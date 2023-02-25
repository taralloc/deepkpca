import copy
from datetime import datetime
import socket
import time
from argparse import Namespace
from typing import List
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
from torch import nn
from utils import get_params
import geoopt

def terminatingcondition_factory(optimizer_args):
    name = optimizer_args["name"]
    assert name in ["adam", "pg", "geoopt"]
    from functools import reduce
    import operator
    if name == "adam":
        def terminating_condition(loss, model, optimizer):
            params = reduce(operator.concat, [g['params'] for g in optimizer.param_groups]) #list of all variables
            grad_q = [torch.norm(p.grad) for p in params if p.grad is not None]
            grad_q = sum(grad_q) if len(grad_q) > 0 else np.inf

            if not hasattr(optimizer, "prev_params"):
                optimizer.best_loss = float(loss)
                optimizer.patience = optimizer_args["patience"]
                return True
            else:
                diff_loss = optimizer.best_loss - float(loss)
                if diff_loss < optimizer_args["beta"] and (optimizer.param_groups[0]['lr'] == optimizer_args.torch.lr or optimizer.param_groups[0]['lr'] == optimizer_args.lr_scheduler.min_lr):
                    optimizer.patience -= 1
                else:
                    optimizer.patience = optimizer_args["patience"]
                if float(loss) < optimizer.best_loss:
                    optimizer.best_loss = float(loss)
                if not optimizer.patience > 0:
                    print("Stopping because diff loss")

            return grad_q > 1e-7 and loss > 1e-10 and optimizer.patience > 0
    elif name == "pg" or name == "geoopt":
        def terminating_condition(loss, model, optimizer):
            params = reduce(operator.concat, [g['params'] for g in optimizer.param_groups]) #list of all variables
            if not hasattr(optimizer, "prev_params"):
                optimizer.prev_params = [torch.clone(param) for param in params]
                optimizer.best_loss = float(loss)
                optimizer.patience = optimizer_args["patience"]
                return True
            else:
                alpha = optimizer.alpha if hasattr(optimizer, "alpha") else optimizer.param_groups[0]['lr']
                res = [float(torch.max(torch.abs(params[i] - optimizer.prev_params[i]))) / alpha > optimizer_args["epsilon"] for i in range(len(params))]
                res = reduce(lambda x,y: x or y, res)
                optimizer.prev_params = [torch.clone(param) for param in params]
                diff_loss = optimizer.best_loss - float(loss)
                if diff_loss < optimizer_args["beta"] and (optimizer.param_groups[0]['lr'] == optimizer_args.torch.lr or optimizer.param_groups[0]['lr'] == optimizer_args.lr_scheduler.min_lr):
                    optimizer.patience -= 1
                else:
                    optimizer.patience = optimizer_args["patience"]
                if float(loss) < optimizer.best_loss:
                    optimizer.best_loss = float(loss)
                if not res:
                    print("Stopping because diff param")
                if not optimizer.patience > 0:
                    print("Stopping because diff loss")
                return res and optimizer.patience > 0
    return terminating_condition

def params_factory(stiefel_params: List, pos_params: List, other_params: List, optimizer_args: DictConfig):
    name = optimizer_args["name"]
    assert name in ["adam", "pg", "geoopt"]
    if name == "adam":
        return stiefel_params + pos_params + other_params
    elif name == "pg":
        # Divide differentiable parameters in 2 groups: 1. Manifold parameters 2. Kernel parameters
        return [{'params': stiefel_params, 'stiefel': True, 'lr': optimizer_args["torch"]["lr"]},
                {'params': pos_params + other_params, 'stiefel': False, 'lr': optimizer_args["torch"]["lr"]}]
    elif name == "geoopt":
        # Divide differentiable parameters in 2 groups: 1. Manifold parameters 2. Kernel parameters
        return stiefel_params + pos_params + other_params

def optimizer_factory(parameters, optimizer_args: DictConfig):
    name = optimizer_args["name"]
    optimizer = None
    assert name in ["adam", "pg", "geoopt"]
    if name == "adam":
        optimizer = torch.optim.Adam(parameters, **optimizer_args["torch"])
    elif name == "pg":
        from opt_algorithms import st_optimizers
        optimizer = st_optimizers.ProjectedGradient(parameters)
    elif name == "geoopt":
        from geoopt.optim import RiemannianAdam
        optimizer = RiemannianAdam(parameters, **optimizer_args["torch"])
    return optimizer

def initialize(parameter, init_args, xtrain, kernels, s, level_index):
    name = init_args["name"]
    shape = parameter.shape
    assert name in ["random", "levelwise", "unsupervised"]
    if name == "random":
        if len(parameter.shape) >= 2:
            parameter.data.copy_(torch.randn(shape).data)
            torch.nn.init.orthogonal_(parameter)
        elif len(parameter.shape) == 1:
            parameter.data.copy_(torch.rand(shape).data)
    elif name == "levelwise":
        from utils import kPCA
        h = xtrain
        for i, kernel in enumerate(kernels):
            h, s_kpca = kPCA(h, h_n=s[i], k=kernel)
            if i == level_index:
                break
        if len(parameter.shape) == 2:
            parameter.data.copy_(h.data)
        elif len(parameter.shape) == 1:
            parameter.data.copy_(s_kpca.data[:s[level_index]])
        else:
            raise NameError()
    elif name == "unsupervised":
        from opt_algorithms import st_optimizers
        def constr_drkm(h):
            H1 = h[:, :s[0]]
            H2 = h[:, s[0]:]
            K1 = kernels[0](xtrain.t())
            K2 = kernels[1](H1.t())
            return - 0.5 * torch.trace(H1.t() @ K1 @ H1) - 0.5 * torch.trace(H2.t() @ K2 @ H2)

        h = torch.randn((xtrain.shape[0], sum(s)), device=parameter.device, requires_grad=True)
        optimizer = st_optimizers.ProjectedGradient([{'params': [h], 'stiefel': True, 'lr': 0.01}], lr=0.01)
        t = 1
        while t < 1000:
            def closure():
                return constr_drkm(h)

            loss = constr_drkm(h)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure)
            t += 1
        parameter.data.copy_(h[:, s[0]:].data)
    return parameter

def train_deepkpca(xtrain, levels, args_optimizer, model_to_load=None, svdopt=False):
    N = xtrain.shape[0]
    kernels = [level['kernel'] for level in levels]
    s = [level['s'] for level in levels]
    etas = [level['eta'] for level in levels]

    H2_tilde = initialize(torch.empty((N, s[1]), device=definitions.device, requires_grad=True), args_optimizer.init, xtrain, kernels, s, 1)
    H1_tilde = initialize(torch.empty((N, s[0]), device=definitions.device, requires_grad=True), args_optimizer.init, xtrain, kernels, s, 0)
    if args_optimizer["name"] == "geoopt":
        with torch.no_grad():
           H1_tilde = geoopt.ManifoldParameter(H1_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
           H2_tilde = geoopt.ManifoldParameter(H2_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
    L1_tilde = initialize(torch.empty((s[0],), device=definitions.device, requires_grad=True), args_optimizer.init, xtrain, kernels, s, 0)
    L2_tilde = initialize(torch.empty((s[1],), device=definitions.device, requires_grad=True), args_optimizer.init, xtrain, kernels, s, 1)
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
    def rkm_loss_residual(x):
        eta1, eta2 = etas[0], etas[1]
        return 0.5 * torch.norm(1./eta2 * kernels[1](H1_tilde.t()) - H2_tilde @ torch.diag(L2_tilde) @ H2_tilde.t(), 'fro') ** 2 + \
               0.5 * torch.norm(1./eta1 * Kx @ H1_tilde + 1./eta2 * kernels[1](H2_tilde.t()) @ H1_tilde - H1_tilde @ torch.diag(L1_tilde), 'fro') ** 2, \
               [[None, None], [None, None]]

    def rkm_loss_svd(x):
        return 0.5 * torch.norm(kernels[1](H1_tilde.t()) - H2_tilde @ torch.diag(L2_tilde) @ H2_tilde.t(), 'fro') ** 2 + \
               0.5 * torch.norm(Kx + kernels[1](H2_tilde.t()) - H1_tilde @ torch.diag(L1_tilde) @ H1_tilde.t(), 'fro') ** 2, \
               [[None, None], [None, None]]

    rkm_loss = rkm_loss_residual if not svdopt else rkm_loss_svd

    cost, grad_q, t, train_table, ortos, best_cost = np.inf, np.nan, 0, pandas.DataFrame(), {'orto1': np.inf, 'orto2': np.inf}, np.inf  # Initialize

    def log_epoch(train_table, log_dict):
        train_table = pandas.concat([train_table, pandas.DataFrame(log_dict, index=[0])])
        logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).to_string(header=(t == 0), index=False, justify='right', col_space=15, float_format=utils.float_format, formatters={'mu': lambda x: "%.2f" % x}))
        return train_table

    # Optimization loop
    train_table = log_epoch(train_table, {'i': t, 'j': float(cost), 'grad_j': float(cost), 'orto1': float(utils.orto(H1_tilde.t() / torch.linalg.norm(H1_tilde.t(), 2, dim=0))), 'orto2': float(utils.orto(H2_tilde/ torch.linalg.norm(H2_tilde, 2, dim=0))), 'lr': optimizer.param_groups[0]['lr']})
    terminating_condition = terminatingcondition_factory(args_optimizer)
    start = datetime.now()
    while cost > 0.0 and t < args_optimizer.maxepochs and terminating_condition(cost, rkm_loss, optimizer):  # run epochs until convergence or cut-off
        loss, [[f1_H, f1_L], [f2_H, f2_L]] = rkm_loss(xtrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: rkm_loss(xtrain)[0])
        if lr_scheduler is not None:
            lr_scheduler.step(loss)
        t += 1
        cost = float(loss.detach().cpu())
        # Logging
        grad_q = float(sum([torch.linalg.norm(p.grad) for p in get_params(params)]).detach().cpu())
        ortos = {f'orto1': float(utils.orto(H1_tilde / torch.linalg.norm(H1_tilde, 2, dim=0))), f'orto2': float(utils.orto(H2_tilde / torch.linalg.norm(H2_tilde, 2, dim=0)))}
        log_dict = {'i': t, 'j': float(loss.detach().cpu()), 'grad_j': grad_q, 'lr': optimizer.param_groups[0]['lr']}
        log_dict = utils.merge_dicts([log_dict, ortos])
        train_table = log_epoch(train_table, log_dict)

    elapsed_time = datetime.now() - start
    logging.info("Training complete in: " + str(elapsed_time))

    return {"train_time": elapsed_time.total_seconds(), 'h2tilde-initial_plot': None, 'H2_tilde': H2_tilde.detach().cpu(), 'H1_tilde': H1_tilde.detach().cpu(),
            'L2_tilde': L2_tilde.detach().cpu(), 'L1_tilde': L1_tilde.detach().cpu(), "eigs": [L1_tilde.detach().cpu().numpy(), L2_tilde.detach().cpu().numpy()],
            "optimizer": optimizer.state_dict()}

def eval_reconstruction(training_dict, x_train, levels, x_train_clean):
    phis_inv = [level['kernel'].phi_inv for level in levels]
    H2, L2 = training_dict["H2_tilde"].to(device), training_dict["L2_tilde"].to(device)
    H1, L1 = training_dict["H1_tilde"].to(device), training_dict["L1_tilde"].to(device)
    W1, W2 = x_train.t() @ H1, H1.t() @ H2
    x_hat = phis_inv[0]((W1 @ phis_inv[1](W2 @ H2.t())).t())
    loss = nn.MSELoss()
    return float(loss(x_hat, x_train_clean)), x_hat

def eval_reconstruction_oos(training_dict, x_train, levels, x_test, x_test_clean):
    phis_inv = [level['kernel'].phi_inv for level in levels]
    phis = [level['kernel'].phi for level in levels]
    H2, L2 = training_dict["H2_tilde"].to(device), training_dict["L2_tilde"].to(device)
    H1, L1 = training_dict["H1_tilde"].to(device), training_dict["L1_tilde"].to(device)
    W1, W2 = x_train.t() @ H1, H1.t() @ H2
    W2inv = torch.linalg.solve(W2.t() @ W2, W2.t())
    H2_hat = phis[0](x_test) @ (torch.inverse(torch.max(L1)*torch.max(L2)*torch.eye(W2.shape[1]).to(device)-W2.t()@W2) @ W2.t() @ W1.t()).t()
    x_hat = phis_inv[0]((W1 @ phis_inv[1](torch.max(L2) * W2inv.t() @ H2_hat.t())).t())
    loss = nn.MSELoss()
    return float(loss(x_hat, x_test_clean)), x_hat

def load_model(label):
    if label is None:
        return None
    model_dir = OUT_DIR.joinpath(label)
    sd_mdl = torch.load(str(model_dir.joinpath("model.pt")), map_location=torch.device('cpu'))
    return {"H1": sd_mdl["H1"], "H2": sd_mdl["H2"], "L1": sd_mdl["L1"], "L2": sd_mdl["L2"],
            "optimizer": sd_mdl["optimizer"]}


def final_compute(Kx, H1, L1, H2, L2, s1, s2, eta2):
    H1final, L1final, _ = torch.svd(Kx+1/eta2*H2@H2.t())
    H2final, L2final, _ = torch.svd(1/eta2*H1final@H1final.t())
    return H1final, L1final, H2final, L2final


@hydra.main(config_path='configs', config_name='config_rkm', version_base=None)
def main(args: DictConfig):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    label = "model000"
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
    training_dict = train_deepkpca(x_train, levels, args.optimizer, load_model(args.saved_model), svdopt=args.model.svdopt)

    # Approximation bounds
    if args.bounds:
        s1, s2, eta2 = levels[0]['s'], levels[1]['s'], levels[1]['eta']
        Kx = levels[0]['kernel'](x_train.t())
        _, lambda_tilde, _ = torch.svd(Kx)
        H1, H2, L1, L2 = training_dict["H1_tilde"], training_dict["H2_tilde"].to(device), torch.sort(training_dict["L1_tilde"], descending=True).values, training_dict["L2_tilde"]
        H1final, L1final, H2final, L2final = final_compute(Kx, H1, L1, H2, L2, s1, s2, eta2)
        error = float(torch.linalg.norm(Kx-H1final[:,:s1]@torch.diag(L1final[:s1])@H1final[:,:s1].t(), 'fro'))
        lb = float(torch.sqrt(torch.sum(L1final[s1:]**2))-torch.sqrt(torch.tensor(s2))/abs(eta2))
        if eta2 > 0:
            ub = float(torch.sqrt(torch.sum(L1final[s1:]**2)-(1./eta2-2*s1*torch.sum(L1final[:s1]))*s2/eta2))
        else:
            ub = float(torch.sqrt(torch.sum(L1final[s1:]**2)-(s2/eta2+2*torch.sum(lambda_tilde[:s2]))*1./eta2))
        lower_bound = error >= lb
        upper_boud = error <= ub
        logging.info(f"Lower bound is {lower_bound} ({lb}) and upper bound is {upper_boud} ({ub}) with error ({error})")
        exit()

    # Evaluate
    eval_dict = {}
    recon_train = eval_reconstruction(training_dict, x_train, levels, x_train)
    eval_dict.update({"recerr_train": recon_train[0], "xhat_train": recon_train[1]})
    recon_test = eval_reconstruction_oos(training_dict, x_train, levels, x_test, x_test)
    eval_dict.update({"recerr_test": recon_test[0], "xhat_test": recon_test[1]})

    # Eigs of KPCA
    from sklearn.decomposition import KernelPCA
    pca = KernelPCA(n_components=levels[0]['s'], kernel='linear')
    pca.fit(x_train.cpu().numpy())
    eigs_kpca = torch.svd(x_train@x_train.t()).S.cpu()

    eval_dict = utils.merge_dicts([{"train_time": training_dict["train_time"], "eigs": training_dict["eigs"], "eigs_kpca": eigs_kpca},
                                   {}, eval_dict])

    # Save model
    W1, W2 = (x_train.t() @ training_dict["H1_tilde"].to(device)).cpu(), (training_dict["H1_tilde"].t().to(device) @ training_dict["H2_tilde"].to(device)).cpu()
    torch.save({'H1': training_dict["H1_tilde"], 'H2': training_dict["H2_tilde"],
                'L1': training_dict["L1_tilde"], 'L2': training_dict["L2_tilde"],
                'W1': W1, 'W2': W2,
                'optimizer': training_dict["optimizer"],
                'args': args, "ot_train_mean": 0.0, "ot_train_var": 1,
                }, str(model_dir.joinpath("model.pt")))
    with open(str(model_dir.joinpath("config.yaml")), "w") as outfile:
        from omegaconf import OmegaConf
        OmegaConf.save(args, outfile, resolve=True)

    # Finish
    [eval_dict.pop(key) for key in ['h2tilde-initial_plot', 'H2_tilde', 'i', '_runtime', '_timestamp', '_step', "xhat_train", "xden_train", "xden_train_pca", "xhat_test", "xden_test"] if key in eval_dict]
    eval_dict.update({"timestamp": created_timestamp, "hostname": socket.getfqdn()})
    logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
    logging.info(f"Saved label: {label}")

    return eval_dict


if __name__ == '__main__':
    main()
