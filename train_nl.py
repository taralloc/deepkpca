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
    params = params_factory([H1_tilde, H2_tilde], [L1_tilde, L2_tilde], [], args_optimizer)
    optimizer = optimizer_factory(params, args_optimizer)
    if 'lr_scheduler' in args_optimizer and args_optimizer.lr_scheduler.factor < 1:
        from utils import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, **args_optimizer.lr_scheduler)
    else:
        lr_scheduler = None

    # Load saved model
    if model_to_load is not None:
        H1_tilde.data.copy_(model_to_load["H1"])
        H2_tilde.data.copy_(model_to_load["H2"])
        L1_tilde.data.copy_(model_to_load["L1"])
        L2_tilde.data.copy_(model_to_load["L2"])

    Kx = kernels[0](xtrain.t())

    def rkm2(x):
        eta1, eta2 = etas[0], etas[1]
        G = torch.empty((N, s[0], N), device=device)
        K2 = kernels[1](H1_tilde.t())
        for i in range(N):
            G[i] = -2. / kernels[1].sigma2 * (H1_tilde[i] - H1_tilde).t() @ torch.diag(K2[i])
        GH2 = torch.bmm(G, H2_tilde.unsqueeze(0).repeat(N, 1, 1))
        return 0.5 * torch.norm(1. / eta2 * kernels[1](H1_tilde.t()) - H2_tilde @ torch.diag(L2_tilde) @ H2_tilde.t(), 'fro') ** 2 + \
               0.5 * torch.norm(1. / eta1 * Kx @ H1_tilde + 1. / eta2 * torch.bmm(GH2, H2_tilde.unsqueeze(2)).squeeze(2) - H1_tilde @ torch.diag(L1_tilde), 'fro') ** 2, \
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
    while cost > 0.0 and t < args_optimizer.maxepochs and terminating_condition(cost, rkm2,
                                                                                optimizer):  # run epochs until convergence or cut-off
        loss, [[f1_H, f1_L], [f2_H, f2_L]] = rkm2(xtrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: rkm2(xtrain)[0])
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
            "eigs": [L1_tilde.detach().cpu().numpy(), L2_tilde.detach().cpu().numpy()],
            "optimizer": optimizer.state_dict()}

@hydra.main(config_path='configs', config_name='config_rkm', version_base=None)
def main(args: DictConfig):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    label = "model001"
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
