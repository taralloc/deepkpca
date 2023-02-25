import copy
import json
from functools import partial
import torch
from disent.dataset import DisentDataset
from disent.dataset.data import Shapes3dData, Cars3d64Data, SmallNorb64Data
from disent.dataset.transform import ToImgTensorF32
from disent.metrics import metric_irs
import argparse
import utils
from definitions import OUT_DIR
import numpy as np
from dataloader import get_dataloader, get_dataset
from kernels import GaussianKernelTorch
from train_cnn import reconstruct_h2, Net1, Net3, encode_oos, reconstruct_h1
#import tensorflow_hub as hub #uncomment to evaluate disentanglement_lib models

def get_available_y(dataset_name, N, seed, classes):
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load Training Data
    def get_data(d):
        loader = get_dataloader(argparse.Namespace(**d))
        x, y = get_dataset(loader)
        return x, y
    x_train, y_train = get_data({"name": dataset_name, "train": True, "N": N,
                                                          "classes": classes, "mb_size": 64,
                                                          "shuffle": False, "workers": 0})
    return y_train.cpu().numpy().astype(int)

def get_encoders_dkpca(model_label, sigma_h=True):
    model_dir = OUT_DIR.joinpath(model_label)
    sd_mdl = torch.load(str(model_dir.joinpath("model.pt")), map_location=torch.device('cpu'))
    args = sd_mdl["args"]

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Training Data
    def get_data(d):
        loader = get_dataloader(argparse.Namespace(**d))
        x, y = get_dataset(loader)
        return x, y
    args_dataset = copy.copy(args.dataset)
    x_train, y_train = get_data(utils.merge_two_dicts(args_dataset, {"train": True}))
    x_train = x_train.cpu()

    # Define get_repr for Deep KPCA
    H1, H2, L1, L2, W1, W2 = sd_mdl["H1"].to("cpu"), sd_mdl["H2"].to("cpu"), sd_mdl["L1"].to("cpu"), sd_mdl["L2"].to("cpu"), sd_mdl["W1"].to("cpu"), sd_mdl["W2"].to("cpu")
    img_shape = list(x_train.shape[1:])
    nChannels = x_train.shape[1]
    if "phi1" in sd_mdl:
        cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)
        if np.prod(x_train.shape[1:]) <= 28 * 28 * 3:
            cnn_kwargs = (cnn_kwargs, dict(kernel_size=3, stride=1), 5)
        else:
            cnn_kwargs = cnn_kwargs, cnn_kwargs, 8
        phi1 = Net1(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to("cpu")
        psi1 = Net3(nChannels, capacity=args.levels.j1.phi.capacity, x_fdim1=args.levels.j1.phi.x_fdim1, x_fdim2=args.levels.j1.phi.x_fdim2, cnn_kwargs=cnn_kwargs).to("cpu")
        phi1.load_state_dict(sd_mdl["phi1"])
        psi1.load_state_dict(sd_mdl["psi1"])
        op1 = phi1(x_train)
    else:
        phi1 = lambda x: x.view(-1, np.prod(x.shape[1:]))
        op1 = phi1(x_train)

    def get_repr(mode="weighted"):
        def get_repr_helper(x, mode="weighted", sigma_h=True):
            assert mode in ["weighted", "oos"]

            if mode == "oos":
                T = H2.t() @ H1 @ torch.diag(1./L1) @ H1.t()
                h2 = (1. / (args.levels.j1.eta * args.levels.j2.eta) * torch.inverse(torch.diag(L2) - 1./args.levels.j2.eta**2 * T @ H2) @ T @ op1 @ phi1(x).t()).t()
                h1 = (torch.diag(1./L1) @ (1./args.levels.j1.eta * H1.t() @ op1 @ phi1(x).t() + 1./args.levels.j2.eta * H1.t()@H2@h2.t())).t()
                h = torch.cat((h1, h2), dim=1)
            elif mode == "weighted":
                hh = torch.cat((H1, H2), dim=1)
                sigma2 = 40000.
                if sigma_h:
                    sigmas = [(sigma, GaussianKernelTorch(sigma2=sigma)(x.reshape(x.shape[0], -1).t(), x_train.reshape(x_train.shape[0], -1).t()).min()) for sigma in np.linspace(1.0, 1000.0, 15)]
                    for i in range(len(sigmas)):
                        if sigmas[i][1] > 1e-7 and i > 0:
                            sigma2 = sigmas[i-1][0]
                            break
                kernel = GaussianKernelTorch(sigma2=sigma2)
                K = kernel(x.reshape(x.shape[0], -1).t(), x_train.reshape(x_train.shape[0], -1).t()).cpu()
                hh = hh.cpu()
                h = 0.5 * torch.matmul(K, hh)

            assert len(h.shape) == 2
            assert h.shape[0] == x.shape[0]
            return h
        f = lambda x: get_repr_helper(x, mode=mode, sigma_h=sigma_h)
        f.method = f"-{mode}"
        return f

    return [get_repr(mode=mode) for z in ["cat"] for mode in ["weighted"]]

def get_encoders_disentanglement_lib(model_label):
    model_dir = OUT_DIR.joinpath(model_label)
    if model_dir.joinpath("postprocess/tfhub").exists():
        module_path = model_dir.joinpath("postprocess/tfhub")
    elif model_dir.joinpath("postprocessed/mean/tfhub").exists():
        module_path = model_dir.joinpath("postprocessed/mean/tfhub")
    else:
        raise NameError(f"{model_label} for disentanglement_lib has not been postprocessed")

    def f(x):
        with hub.eval_function_for_module(str(module_path)) as h:
            """Computes representation vector for input images."""
            output = h(dict(images=x.permute(0,2,3,1).numpy()), signature="representation", as_dict=True)
            return np.array(output["default"])
    f.method = "vae"

    return [f]

def get_model_type(model_label):
    model_dir = OUT_DIR.joinpath(model_label)
    if model_dir.joinpath("config.json").exists():
        return "disentanglement_lib"
    elif model_dir.joinpath("postprocessed").exists():
        return "disentanglement_lib-pretrained"
    elif model_dir.joinpath("model.pt").exists():
        sd_mdl = torch.load(str(model_dir.joinpath("model.pt")), map_location=torch.device('cpu'))
        if "args" in sd_mdl:
            return "dkpca"
    else:
        raise NameError("Model type not recognized")

def metric_helper(f, available_y=None, N=100, seed=0):
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = metric_irs(dataset, f, num_train=N, available_y=available_y, batch_size=256)
    return {f"{key}_{f.method}_{N}": value for key, value in d.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='2ihzl9wb', help='Enter model label')
    parser.add_argument('--n', type=int, default=100, help='Enter number of training points to use for evaluation')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite metrics.json')
    opt_gen = parser.parse_args()
    model_dir = OUT_DIR.joinpath(opt_gen.model)

    type = get_model_type(opt_gen.model)
    get_reprs, dataset_name, available_y, N, seed, size = [], None, None, None, None, None
    if type == "dkpca":
        args = torch.load(str(model_dir.joinpath("model.pt")), map_location=torch.device('cpu'))["args"]
        dataset_name = args.dataset.name
        get_reprs = get_encoders_dkpca(opt_gen.model, sigma_h=(dataset_name=="cars3d"))
        available_y = get_available_y(dataset_name, opt_gen.n, args.seed, args.dataset.classes)
        seed = args.seed
        size = 28 if dataset_name in ["3dshapes", "norb"] else None
    elif type == "disentanglement_lib":
        get_reprs = get_encoders_disentanglement_lib(opt_gen.model)
        args = json.load(model_dir.joinpath("config.json").open())
        args["dataset"] = {"cars3d": "cars3d", "smallnorb": "norb", "shapes3d": "3dshapes"}[args["dataset"]]
        dataset_name = args["dataset"]
        available_y = get_available_y(args["dataset"], opt_gen.n, args["seed"], -1)
        seed = args["seed"]
    elif type == "disentanglement_lib-pretrained":
        get_reprs = get_encoders_disentanglement_lib(opt_gen.model)
        args = json.load(model_dir.joinpath("model/results/json/train_config.json").open())
        args["dataset.name"] = {"cars3d": "cars3d", "smallnorb": "norb", "shapes3d": "3dshapes"}[args["dataset.name"].replace('\'', '')]
        dataset_name = args["dataset.name"]
        seed = int(args["model.random_seed"].replace('\'', ''))
    else:
        raise NameError("Model type not recognized")

    # Load dataset for Disent library
    data = None
    if dataset_name == "3dshapes":
        data = Shapes3dData(prepare=True)
    elif dataset_name == "cars3d":
        data = Cars3d64Data(prepare=True)
    elif dataset_name == "norb":
        data = SmallNorb64Data(prepare=True)
    else:
        raise NameError("Not known dataset")
    dataset = DisentDataset(data, transform=ToImgTensorF32(size=size), augment=None)

    metrics = [
               partial(metric_helper, available_y=available_y, N=opt_gen.n, seed=seed),
              ]
    eval_dict = utils.merge_dicts([metric(get_repr) for metric in metrics for get_repr in get_reprs])

    print("\n".join("{}\t{}".format(k, str(v)) for k, v in eval_dict.items()))

    if opt_gen.overwrite or not model_dir.joinpath("metrics.json").exists():
        with open(str(model_dir.joinpath("metrics.json")), 'w') as f:
            json.dump(eval_dict, f)
    else:
        old_metrics = json.load(model_dir.joinpath("metrics.json").open())
        for key in old_metrics:
            eval_dict[key] = old_metrics[key]
        with open(str(model_dir.joinpath("metrics.json")), 'w') as f:
            json.dump(eval_dict, f)

