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
import numpy as np
import hydra
from omegaconf import DictConfig
import logging, sys
from torch import nn
from utils import get_params, Lin_View
import geoopt
from train_lin import terminatingcondition_factory, optimizer_factory, initialize, params_factory, load_model

class Net1(nn.Module):
    """ Encoder - network architecture """
    def __init__(self, nChannels, capacity, x_fdim1, x_fdim2, cnn_kwargs):
        super(Net1, self).__init__()  # inheritance used here.
        self.main = nn.Sequential(
            nn.Conv2d(nChannels, capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(capacity, capacity * 2, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(capacity * 2, capacity * 4, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Flatten(),
            nn.Linear(capacity * 4 * cnn_kwargs[2] ** 2, x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(x_fdim1, x_fdim2),
        )

    def forward(self, x):
        return self.main(x)


class Net3(nn.Module):
    """ Decoder - network architecture """
    def __init__(self, nChannels, capacity, x_fdim1, x_fdim2, cnn_kwargs):
        super(Net3, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(x_fdim2, x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(x_fdim1, capacity * 4 * cnn_kwargs[2] ** 2),
            nn.LeakyReLU(negative_slope=0.2),
            Lin_View(capacity * 4, cnn_kwargs[2], cnn_kwargs[2]),  # Unflatten

            nn.ConvTranspose2d(capacity * 4, capacity * 2, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(capacity * 2, capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose2d(capacity, nChannels, **cnn_kwargs[0]),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Level():
    def __init__(self, phi, psi, s):
        self.phi = phi
        self.psi = psi
        self.s = s

def train_deepkpca(xtrain, levels: List[Level], args_optimizer, ae_weight=None, model_to_load=None):
    if ae_weight is None:
        ae_weight = 10.0
    from kernels import LinearKernel

    N = xtrain.shape[0]
    s = [level.s for level in levels]

    H2_tilde = torch.randn((N, s[1]), device=definitions.device)
    H1_tilde = torch.randn((N, s[0]), device=definitions.device)
    if args_optimizer.name == "geoopt":
        with torch.no_grad():
            H1_tilde = geoopt.ManifoldParameter(H1_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
            H2_tilde = geoopt.ManifoldParameter(H2_tilde, manifold=geoopt.Stiefel(), requires_grad=True).proj_()
    L1_tilde = torch.randn((s[0],), device=definitions.device, requires_grad=True)
    L2_tilde = torch.randn((s[1],), device=definitions.device, requires_grad=True)
    params = params_factory([H1_tilde, H2_tilde], [L1_tilde, L2_tilde], [], args_optimizer)
    optimizer = optimizer_factory(params, args_optimizer)
    if 'lr_scheduler' in args_optimizer and args_optimizer.lr_scheduler.factor < 1:
        from utils import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, **args_optimizer.lr_scheduler)
    else:
        lr_scheduler = None

    # Explicit feature map
    phi1, psi1 = levels[0].phi, levels[0].psi
    optimizer2 = torch.optim.Adam(list(phi1.parameters()) + list(psi1.parameters()), lr=args_optimizer.lr_nn, weight_decay=0)

    # Load saved model
    if model_to_load is not None:
        H1_tilde.data.copy_(model_to_load["H1"])
        H2_tilde.data.copy_(model_to_load["H2"])
        L1_tilde.data.copy_(model_to_load["L1"])
        L2_tilde.data.copy_(model_to_load["L2"])
        phi1.load_state_dict(model_to_load["phi1"])
        phi1.eval()
        psi1.load_state_dict(model_to_load["psi1"])
        psi1.eval()

    lin_kernel = LinearKernel()
    def rkm2(x):
        op1 = phi1(x)
        op1 = op1 - torch.mean(op1, dim=0)
        Kx = op1 @ op1.t()

        f1 = 0.5 * torch.norm(lin_kernel(H1_tilde.t()) - H2_tilde @ torch.diag(L2_tilde) @ H2_tilde.t(), 'fro') ** 2 + \
             0.5 * torch.norm(Kx + lin_kernel(H2_tilde.t()) - H1_tilde @ torch.diag(L1_tilde) @ H1_tilde.t(), 'fro') ** 2

        x_tilde = psi1(torch.mm(torch.mm(H1_tilde, H1_tilde.t()), op1))
        loss = nn.MSELoss(reduction='sum')
        f2 = 0.5 * loss(x_tilde.view(-1, np.prod(x.shape[1:])), x.view(-1, np.prod(x.shape[1:]))) / x.shape[0]  # Recons_loss

        return 1 * f1 + ae_weight * f2, float(f1.detach().cpu()), float(f2.detach().cpu())


    def log_epoch(train_table, log_dict):
        train_table = pandas.concat([train_table, pandas.DataFrame(log_dict, index=[0])])
        logging.info((train_table.iloc[len(train_table) - 1:len(train_table)]).to_string(header=(t == 0), index=False, justify='right', col_space=15, float_format=utils.float_format, formatters={'mu': lambda x: "%.2f" % x}))
        return train_table

    # Optimization loop
    cost, grad_q, t, train_table, ortos, best_cost = np.inf, np.nan, 0, pandas.DataFrame(), {'orto1': np.inf, 'orto2': np.inf}, np.inf  # Initialize
    train_table = log_epoch(train_table, {'i': t, 'j': float(cost), 'kpca': float(np.inf), 'ae': float(np.inf), 'orto1': float(utils.orto(H1_tilde.t() / torch.linalg.norm(H1_tilde.t(), 2, dim=0))), 'orto2': float(utils.orto(H2_tilde/ torch.linalg.norm(H2_tilde, 2, dim=0))), 'lr': optimizer.param_groups[0]['lr']})
    terminating_condition = terminatingcondition_factory(args_optimizer)
    start = datetime.now()
    while cost > 1e-10 and t < args_optimizer.maxepochs and terminating_condition(cost, rkm2, optimizer):  # run epochs until convergence or cut-off
        loss, f1, f2 = rkm2(xtrain)
        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        optimizer.step(lambda: rkm2(xtrain)[0])
        if lr_scheduler is not None:
            lr_scheduler.step(f1)
        t += 1
        cost = float(loss.detach().cpu())
        # Logging
        ortos = {f'orto1': float(utils.orto(H1_tilde / torch.linalg.norm(H1_tilde, 2, dim=0))), f'orto2': float(utils.orto(H2_tilde / torch.linalg.norm(H2_tilde, 2, dim=0)))}
        log_dict = {'i': t, 'j': float(loss.detach().cpu()), 'kpca': f1, 'ae': f2,  'lr': optimizer.param_groups[0]['lr']}
        log_dict = utils.merge_dicts([log_dict, ortos])
        train_table = log_epoch(train_table, log_dict)
    elapsed_time = datetime.now() - start
    logging.info("Training complete in: " + str(elapsed_time))

    return {"train_time": elapsed_time.total_seconds(), 'H2_tilde': H2_tilde.detach().cpu(), 'H1_tilde': H1_tilde.detach().cpu(),
            'L2_tilde': L2_tilde.detach().cpu(), 'L1_tilde': L1_tilde.detach().cpu(), "eigs": [L1_tilde.detach().cpu().numpy(), L2_tilde.detach().cpu().numpy()],
            "phi1": phi1, "psi1": psi1}

def reconstruct_h2(W1, W2, H2, psi1):
    H1 = (W2 @ H2.t()).t()
    x_hat = reconstruct_h1(W1, H1, psi1)
    return x_hat

def reconstruct_h1(W1, H1, psi1):
    x_hat = psi1((W1 @ H1.t()).t())
    return x_hat

def encode_oos(x_test, L1, L2, W1, W2, phi1, ot_train_mean):
    op1 = phi1(x_test)
    op1 = op1 - ot_train_mean
    H1 = torch.inverse(torch.diag(L1) - W2 @ torch.diag(1./L2) @ W2.t()) @ W1.t() @ op1.t()
    h1 = H1.t()
    H2 = torch.diag(1./L2) @ W2.t() @ h1.t()
    h2 = H2.t()
    return h1, h2

@hydra.main(config_path='configs', config_name='config_rkm', version_base=None)
def main(args: DictConfig):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    label = "model004"
    created_timestamp = int(time.time())
    model_dir = OUT_DIR.joinpath(label)
    model_dir.mkdir()

    # Load Training Data
    def get_data(d):
        loader = get_dataloader(Namespace(**d))
        x, y = get_dataset(loader)
        return x, y
    args_dataset = copy.copy(args.dataset)
    x_train, y_train = get_data(utils.merge_two_dicts(args_dataset, {"train": True}))
    x_test, y_test = get_data(utils.merge_two_dicts(args_dataset, {"train": False}))

    # Define explicit maps
    nChannels = x_train.shape[1]
    cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
    if np.prod(x_train.shape[1:]) <= 28 * 28 * 3:
        cnn_kwargs = (cnn_kwargs, dict(kernel_size=3, stride=1), 5)
    else:
        cnn_kwargs = cnn_kwargs, cnn_kwargs, 8
    phi1 = Net1(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
    psi1 = Net3(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to(device)
    levels = [Level(phi1, psi1, args.levels.j1.s), Level(lambda x: x, lambda x: x, args.levels.j2.s)]

    # Train
    training_dict = train_deepkpca(x_train, levels, args.optimizer, ae_weight=args.ae_weight, model_to_load=load_model(args.saved_model))
    op1 = phi1(x_train)
    H2, L2 = training_dict["H2_tilde"].to(device), training_dict["L2_tilde"].to(device)
    H1, L1 = training_dict["H1_tilde"].to(device), training_dict["L1_tilde"].to(device)
    W1, W2 = op1.view(-1, np.prod(op1.shape[1:])).t() @ H1, H1.t() @ H2
    ot_train_mean = torch.mean(op1, dim=0)

    # Evaluate
    eval_dict = {}
    loss = nn.MSELoss()
    x_train_hat = reconstruct_h2(W1, W2, H2, psi1)
    x_test_hat = reconstruct_h2(W1, W2, encode_oos(x_test, L1, L2, W1, W2, phi1, ot_train_mean)[1], psi1)
    eval_dict.update({"recerr_train": float(loss(x_train, x_train_hat)), "xhat_train": x_train_hat})
    eval_dict.update({"recerr_test": float(loss(x_test, x_test_hat)), "xhat_test": x_test_hat})
    eval_dict = utils.merge_dicts([{"train_time": training_dict["train_time"], "eigs": training_dict["eigs"]},
                                   {}, eval_dict])

    [eval_dict.pop(key) for key in ['h2tilde-initial_plot', 'H2_tilde', 'i', '_runtime', '_timestamp', '_step', "xhat_train", "xden_train", "xden_train_pca", "xhat_test", "xden_test"] if key in eval_dict]
    eval_dict.update({"timestamp": created_timestamp, "hostname": socket.getfqdn()})
    logging.info("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))
    logging.info(f"Saved label: {label}")

    # Save model
    torch.save({'H1': training_dict["H1_tilde"], 'H2': training_dict["H2_tilde"],
                'L1': training_dict["L1_tilde"], 'L2': training_dict["L2_tilde"],
                'W1': W1, 'W2': W2,
                'args': args, "ot_train_mean": ot_train_mean, "ot_train_var": 1,
                'phi1': training_dict["phi1"].state_dict(), 'psi1': training_dict["psi1"].state_dict()
                }, str(model_dir.joinpath("model.pt")))
    return eval_dict


if __name__ == '__main__':
    main()
