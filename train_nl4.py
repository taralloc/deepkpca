import copy
from datetime import datetime
import socket
import time
from argparse import Namespace
import pandas
import torch
import os
import definitions
from definitions import OUT_DIR
import utils
from dataloader import get_dataloader, get_dataset
from definitions import device
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from kernels import kernel_factory
import numpy as np
import hydra
from omegaconf import DictConfig
import logging, sys
from utils import get_params
from train_lin import terminatingcondition_factory, optimizer_factory, initialize, params_factory, load_model

def train_deepkpca(xtrain, levels, args_optimizer, model_to_load=None):
    N = xtrain.shape[0]
    kernels = [level['kernel'] for level in levels]
    s = [level['s'] for level in levels]
    etas = [level['eta'] for level in levels]

    H4_tilde = torch.empty((N, s[3]), device=definitions.device)
    H4_tilde = initialize(H4_tilde, args_optimizer.init, xtrain, kernels, s, 3)
    H4_tilde.requires_grad_()

    H3_tilde = torch.empty((N, s[2]), device=definitions.device)
    H3_tilde = initialize(H3_tilde, args_optimizer.init, xtrain, kernels, s, 2)
    H3_tilde.requires_grad_()

    H2_tilde = torch.empty((N, s[1]), device=definitions.device)
    H2_tilde = initialize(H2_tilde, args_optimizer.init, xtrain, kernels, s, 1)
    H2_tilde.requires_grad_()

    H1_tilde = torch.empty((N, s[0]), device=definitions.device)
    H1_tilde = initialize(H1_tilde, args_optimizer.init, xtrain, kernels, s, 0)
    H1_tilde.requires_grad_()

    L1_tilde = torch.empty((s[0],), device=definitions.device)
    L2_tilde = torch.empty((s[1],), device=definitions.device)
    L1_tilde = initialize(L1_tilde, args_optimizer.init, xtrain, kernels, s, 0)
    L2_tilde = initialize(L2_tilde, args_optimizer.init, xtrain, kernels, s, 1)
    L1_tilde.requires_grad_()
    L2_tilde.requires_grad_()
    L3_tilde = torch.empty((s[1],), device=definitions.device)
    L4_tilde = torch.empty((s[2],), device=definitions.device)
    L3_tilde = initialize(L3_tilde, args_optimizer.init, xtrain, kernels, s, 2)
    L4_tilde = initialize(L4_tilde, args_optimizer.init, xtrain, kernels, s, 3)
    L3_tilde.requires_grad_()
    L4_tilde.requires_grad_()
    params = params_factory([H1_tilde, H2_tilde, H3_tilde, H4_tilde], [L1_tilde, L2_tilde, L3_tilde, L4_tilde], [], args_optimizer)
    optimizer = optimizer_factory(params, args_optimizer)
    if 'lr_scheduler' in args_optimizer and args_optimizer.lr_scheduler.factor < 1:
        from utils import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, **args_optimizer.lr_scheduler)
    else:
        lr_scheduler = None

    Kx = kernels[0](xtrain.t())

    def khatri_rao(A, B):
        assert A.shape[1] == B.shape[1]
        return torch.vstack([torch.kron(A[i], B[i]) for i in range(A.shape[1])]).t()

    def rkm4(x):
        eta1, eta2 = etas[0], etas[1]
        eta3, eta4 = etas[2], etas[3]

        K2 = kernels[1](H1_tilde.t())
        K3 = kernels[2](H2_tilde.t())
        K4 = kernels[3](H3_tilde.t())

        KR1 = khatri_rao(torch.eye(N, N, device=device), (H2_tilde@H2_tilde.t()))
        J1 = torch.vstack([torch.vstack([-2./kernels[1].sigma2 * (H1_tilde[i]-H1_tilde[j]) * kernels[1](H1_tilde[i].unsqueeze(dim=-1), H1_tilde[j].unsqueeze(dim=-1)) for j in range(N)]) for i in range(N)])
        G1 = KR1.t() @ J1

        KR2 = khatri_rao(torch.eye(N, N, device=device), (H3_tilde@H3_tilde.t()))
        J2 = torch.vstack([torch.vstack([-2./kernels[2].sigma2 * (H2_tilde[i]-H2_tilde[j]) * kernels[2](H2_tilde[i].unsqueeze(dim=-1), H2_tilde[j].unsqueeze(dim=-1)) for j in range(N)]) for i in range(N)])
        G2 = KR2.t() @ J2

        KR3 = khatri_rao(torch.eye(N, N, device=device), (H4_tilde@H4_tilde.t()))
        J3 = torch.vstack([torch.vstack([-2./kernels[3].sigma2 * (H3_tilde[i]-H3_tilde[j]) * kernels[3](H3_tilde[i].unsqueeze(dim=-1), H3_tilde[j].unsqueeze(dim=-1)) for j in range(N)]) for i in range(N)])
        G3 = KR3.t() @ J3

        f4 = 0.5 * torch.norm(1. / eta4 * K4 @ H4_tilde - H4_tilde @ torch.diag(L4_tilde), 'fro') ** 2
        f1 = 0.5 * torch.norm(1. / eta1 * Kx @ H1_tilde + 1. / eta2 * G1 - H1_tilde @ torch.diag(L1_tilde), 'fro') ** 2
        f2 = 0.5 * torch.norm(1. / eta2 * K2 @ H2_tilde + 1. / eta3 * G2 - H2_tilde @ torch.diag(L2_tilde), 'fro') ** 2
        f3 = 0.5 * torch.norm(1. / eta3 * K3 @ H3_tilde + 1. / eta4 * G3 - H3_tilde @ torch.diag(L3_tilde), 'fro') ** 2
        return f1 + f2 + f3 + f4, \
               [[None, None], [None, None]]

    cost, grad_q, t, train_table, ortos, best_cost = np.inf, np.nan, 0, pandas.DataFrame(), {'orto1': np.inf,
                                                                                             'orto2': np.inf}, np.inf  # Initialize

    def log_epoch(train_table, log_dict):
        train_table = pandas.concat([train_table, pandas.DataFrame(log_dict, index=[0])])
        logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).to_string(header=(t == 0), index=False,
                                                                                         justify='right', col_space=15,
                                                                                         float_format=utils.float_format,
                                                                                         formatters={'mu': lambda
                                                                                             x: "%.2f" % x}))
        return train_table

    # Optimization loop
    train_table = log_epoch(train_table, {'i': t, 'j': float(cost), 'grad_j': float(cost), 'orto1': float(
        utils.orto(H1_tilde.t() / torch.linalg.norm(H1_tilde.t(), 2, dim=0))),
                                          'orto2': float(utils.orto(H2_tilde / torch.linalg.norm(H2_tilde, 2, dim=0))),
                                          'lr': optimizer.param_groups[0]['lr']})
    terminating_condition = terminatingcondition_factory(args_optimizer)
    start = datetime.now()
    while cost > 0.0 and t < args_optimizer.maxepochs and terminating_condition(cost, rkm4,
                                                                                optimizer):  # run epochs until convergence or cut-off
        loss, [[f1_H, f1_L], [f2_H, f2_L]] = rkm4(xtrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: rkm4(xtrain)[0])
        if lr_scheduler is not None:
            lr_scheduler.step(loss)
        t += 1
        cost = float(loss.detach().cpu())
        # Logging
        grad_q = float(sum([torch.linalg.norm(p.grad) for p in get_params(params)]).detach().cpu())
        ortos = {f'orto1': float(utils.orto(H1_tilde / torch.linalg.norm(H1_tilde, 2, dim=0))),
                 f'orto2': float(utils.orto(H2_tilde / torch.linalg.norm(H2_tilde, 2, dim=0)))}
        log_dict = {'i': t, 'j': float(loss.detach().cpu()), 'grad_j': grad_q, 'lr': optimizer.param_groups[0]['lr']}
        log_dict = utils.merge_dicts([log_dict, ortos])
        train_table = log_epoch(train_table, log_dict)

    elapsed_time = datetime.now() - start
    logging.info("Training complete in: " + str(elapsed_time))

    return {"train_time": elapsed_time.total_seconds(), 'h2tilde-initial_plot': None,
            'H2_tilde': H2_tilde.detach().cpu(), 'H1_tilde': H1_tilde.detach().cpu(),
            'L2_tilde': L2_tilde.detach().cpu(), 'L1_tilde': L1_tilde.detach().cpu(),
            "eigs": [L1_tilde.detach().cpu().numpy(), L2_tilde.detach().cpu().numpy(), L3_tilde.detach().cpu().numpy(), L4_tilde.detach().cpu().numpy()],
            "optimizer": optimizer.state_dict()}

@hydra.main(config_path='configs', config_name='config_rkm', version_base=None)
def main(args: DictConfig):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    label = "model002"
    created_timestamp = int(time.time())
    model_dir = OUT_DIR.joinpath(label)
    model_dir.mkdir()

    # Load Training Data
    def get_data(d):
        loader = get_dataloader(Namespace(**d))
        x, y = get_dataset(loader)
        x = x.view(x.shape[0], -1)
        return x, y

    args_dataset = copy.copy(args.dataset)
    if args.denoising.enabled:
        args_dataset.post_transformations.append(args.denoising.noise)
    x_train, y_train = get_data(utils.merge_two_dicts(args_dataset, {"train": True}))
    x_test, y_test = get_data(utils.merge_two_dicts(args_dataset, {"train": False}))

    # Define level
    levels = [dict(level) for level in args.levels.values()]
    for level in levels:
        kernel = level['kernel']
        level['kernel'] = kernel_factory(kernel['name'], kernel['args'])

    # Train
    training_dict = train_deepkpca(x_train, levels, args.optimizer, load_model(args.saved_model))

    # Evaluate
    eval_dict = {}
    # Eigs of KPCA
    eigs_kpca = torch.svd(levels[0]['kernel'](x_train.t())).S.cpu()[:levels[0]['s']]
    ratio = f"DKPCA ({training_dict['eigs'][0].max() / training_dict['eigs'][0].sum():.5f}) vs. KPCA ({eigs_kpca.max() / eigs_kpca.sum():.5f})"

    eval_dict = utils.merge_dicts([{"train_time": training_dict["train_time"], "eigs": training_dict["eigs"],
                                    "eigs_kpca": eigs_kpca, "ratio": ratio},
                                   {}, eval_dict])

    [eval_dict.pop(key) for key in
     ['h2tilde-initial_plot', 'H2_tilde', 'i', '_runtime', '_timestamp', '_step', "xhat_train", "xden_train",
      "xden_train_pca", "xhat_test", "xden_test"] if key in eval_dict]
    eval_dict.update({"timestamp": created_timestamp, "hostname": socket.getfqdn()})
    logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
    logging.info(f"Saved label: {label}")

    # Save model
    W1, W2 = (x_train.t() @ training_dict["H1_tilde"].to(device)).cpu(), (
                training_dict["H1_tilde"].t().to(device) @ training_dict["H2_tilde"].to(device)).cpu()
    torch.save({'H1': training_dict["H1_tilde"], 'H2': training_dict["H2_tilde"],
                'L1': training_dict["L1_tilde"], 'L2': training_dict["L2_tilde"],
                'W1': W1, 'W2': W2,
                'optimizer': training_dict["optimizer"],
                'args': args, "ot_train_mean": 0.0, "ot_train_var": 1,
                "eigs_kpca": eigs_kpca,
                }, str(model_dir.joinpath("model.pt")))
    import json
    with open(str(model_dir.joinpath("config.yaml")), "w") as outfile:
        from omegaconf import OmegaConf
        OmegaConf.save(args, outfile, resolve=True)

    return eval_dict


if __name__ == '__main__':
    main()
