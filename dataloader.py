import functools
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from definitions import DATA_DIR, device, TensorType, Tensor
import random
import struct
import torch.utils.data as data

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset(loader):
    x, y = [], []
    for data, labels in loader:
        if type(data) == Tensor:
            data = data.to(device=device)
        elif type(data) == list:
            data = list(map(lambda x: x.to(device), data))
        labels = labels.to(device=device)
        x.append(data)
        y.append(labels)
    if type(x[0]) == Tensor:
        x = torch.cat(x)
    elif type(x[0]) == list:
        x = [torch.cat([x[i][v] for i in range(len(x))]).numpy() for v in range(len(x[0]))]
    y = torch.cat(y)
    N = y.shape[0]
    return x.type(TensorType).to(device), y.to(device)

def get_dataloader(args):
    args = transformation_factory(args)
    loader = get_dataloader_helper(args)
    loader = get_dataloader_subset(loader, args)
    return loader

def transformation_factory(args):
    def transformation_factory_helper(transformation):
        assert len(transformation.items()) == 1
        d = {'normalize': transforms.Normalize}
        name, args = list(transformation.items())[0]
        t = d[name](**args)
        return t
    if "post_transformations" in args:
        args.post_transformations = list(map(transformation_factory_helper, args.post_transformations))
    return args

def get_dataloader_subset(loader, args):
    N = args.N if args.train else args.Ntest
    if N >= 0:
        rng = np.random.default_rng(torch.initial_seed())
        indices = list(rng.choice(range(len(loader.dataset)), N, shuffle=False, replace=N>len(loader.dataset)))
        loader = DataLoader(torch.utils.data.Subset(loader.dataset, indices), batch_size=args.mb_size, pin_memory=False, num_workers=args.workers, shuffle=args.shuffle, worker_init_fn=seed_worker)
    return loader

def get_dataloader_helper(args):

    args_dict = vars(args)

    if "post_transformations" not in args_dict:
        args_dict["post_transformations"] = []
    if "pre_transformations" not in args_dict or args_dict["pre_transformations"] == []:
        args_dict["pre_transformations"] = []
    if "train" not in args_dict:
        args_dict["train"] = False
    if "test" not in args_dict:
        args_dict["test"] = False
    if "val" not in args_dict:
        args_dict["val"] = False
    if "dataset_name" not in args_dict:
        args_dict["dataset_name"] = args_dict["name"]

    print(f'Loading data for {args_dict["dataset_name"]}...')

    if args.dataset_name == 'mnist':
        return get_mnist_dataloader(args=args)

    elif args.dataset_name == "square":
        return get_square_dataloader(args=args)

    elif args.dataset_name == "complex6":
        return get_complex6_dataloader(args=args)

    elif args.dataset_name == "multivariatenormal":
        return get_multivariatenormal_dataloader(args=args)

    elif args.dataset_name == "3dshapes":
        return get_3dshapes_dataloader(args=args)

    elif args.dataset_name == 'cars3d':
        return get_cars3d_dataloader(args=args)

    elif args.dataset_name == 'norb':
        return get_norb_dataloader(args=args)

    elif args.dataset_name == "diabetes":
        return get_diabetes_dataloader(args=args)

    elif args.dataset_name == "ionosphere":
        return get_ionosphere_dataloader(args=args)

    elif args.dataset_name == "liver":
        return get_liver_dataloader(args=args)

    elif args.dataset_name == "cholesterol":
        return get_cholesterol_dataloader(args=args)

    elif args.dataset_name == "yacht":
        return get_yacht_dataloader(args=args)


def get_mnist_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    """MNIST dataloader with (28, 28) images."""

    print("Loading MNIST.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    train_data = datasets.MNIST(path_to_data, train=args.train, download=True, transform=all_transforms)
    if args.classes != -1:
        idx = [(train_data.targets == label) for label in args.classes]
        idx = functools.reduce(lambda x,y: x | y, idx)
        train_data.targets = train_data.targets[idx]
        train_data.data = train_data.data[idx]
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    # _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader

def spiral(N: int, rand_state: np.random.RandomState, std, center=(0,0), b=1.0):
    """
    Returns N 2d points drawing an Archimedean spiral of center and b specified from 0 to 8pi.
    :param N: an even number of points
    :param state: random seed
    :param std: standard deviation of noise
    :return: a pair of (N,2) matrices, the true dataset and the noisy one
    """
    #assert N % 2 == 0
    theta = np.linspace(0, 8*np.pi, N)
    x = b * theta * np.cos(theta) + center[0]
    y = b * theta * np.sin(theta) + center[1]
    noise = rand_state.normal(0, std, 2*N)
    x_n = x + noise[:N]
    y_n = y + noise[N:]
    return (np.vstack([x,y]).T, np.vstack([x_n,y_n]).T)

class Square(Dataset):
    """
    Pytorch dataset for a 2D square. Returns a pair of points.
    """
    def __init__(self, N:int, rand_state: np.random.RandomState, std: float):
        self.x, self.x_n = square(N, rand_state, std)
        self.x = torch.from_numpy(self.x)
        self.x_n = torch.from_numpy(self.x_n)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x_n[i], 1

def circle(N: int, rand_state: np.random.RandomState, std, center=(0.0,0.0), radius=1.0):
    """
    Returns N 2d points drawing a cirlce.
    :param N: an even number of points
    :param state: random seed
    :param std: standard deviation of noise
    :return: a pair of (N,2) matrices, the true dataset and the noisy one
    """
    points = np.linspace(0, 2*np.pi, N)
    x = radius * np.cos(points) + center[0]
    y = radius * np.sin(points) + center[1]
    noise = rand_state.normal(0, std, 2*N)
    x_n = x + noise[:N]
    y_n = y + noise[N:]
    return (np.vstack([x,y]).T, np.vstack([x_n,y_n]).T)

class Complex6(Dataset):
    """
    Pytorch dataset of one square, two spirals and a ring. Returns a pair of points.
    """
    def __init__(self, N:int, rand_state: np.random.RandomState, std: float):
        N_each = N // 5 + 1
        if N_each % 2 != 0:
            N_each += 1
        self.N_each = N_each
        square1, square1_n = square(N_each, rand_state, std, (-5,0.5), 4)
        spiral2, spiral2_n = spiral(N_each, rand_state, std, (-4,-2), b=0.15)
        spiral1, spiral1_n = spiral(N_each, rand_state, std, (5,0), b=0.2)
        circle1, circle1_n = circle(N_each, rand_state, std, (14,0), 2.8)
        circle2, circle2_n = circle(N_each, rand_state, std, (14,0), 1.2)

        self.x, self.x_n = np.concatenate([square1, spiral1, circle1, circle2, spiral2])[:N], \
                           np.concatenate([square1_n, spiral1_n, circle1_n, circle2_n, spiral2_n])[:N]
        self.x = torch.from_numpy(self.x)
        self.x_n = torch.from_numpy(self.x_n)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x_n[i], i // self.N_each

def get_square_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    """Square dataloader."""

    print("Loading Square.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = Square(10000, np.random.RandomState(0), args.std)
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader

def get_complex6_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    """Square dataloader."""

    print("Loading Complex6.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = Complex6(10000, np.random.RandomState(0), args.std)
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader

def get_multivariatenormal_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    """Multivariatenormal dataloader."""

    print("Loading Multivariatenormal.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = MultivariateGaussian(10000, np.random.RandomState(0), args.std)
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader

def get_norb_dataloader(args, path_to_data=DATA_DIR.joinpath('norb')):
    """SmallNORB dataloader with (64, 64, 3) images."""

    if not path_to_data.exists():
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system(f" mkdir {str(path_to_data)};")
        os.system(f" wget -O {str(path_to_data.joinpath('nips2015-analogy-data.tar.gz'))} http://www.scottreed.info/files/nips2015-analogy-data.tar.gz ;")
        os.system(f"tar xzf {str(path_to_data.joinpath('nips2015-analogy-data.tar.gz'))}")

    #from utils import Resize
    from torchvision.transforms import CenterCrop, Resize
    all_transforms = transforms.Compose([Resize((28)), transforms.ToTensor()])

    train_data = SmallNORB(root=path_to_data, transform=all_transforms, train=True, download=True) #test set has different types

    if args.classes != -1:
        assert len(args.classes) == 5
        idx = [functools.reduce(lambda x,y:x|y, [np.isclose(train_data.infos[:,i], label) for label in classs]) for i, classs in enumerate(args.classes) if classs != -1]
        idx2 = [functools.reduce(lambda x,y:x|y, [np.isclose(train_data.infos[:,i], label) for label in np.unique(train_data.infos[:,i])]) for i, classs in enumerate(args.classes) if classs == -1]
        idx4 = functools.reduce(lambda x,y: x & y, idx+idx2)
        train_data.infos = train_data.infos[idx4]
        train_data.data = train_data.data[idx4]
        train_data.labels = train_data.labels[idx4]

    norb_loader = DataLoader(train_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(norb_loader))[0].size()
    return norb_loader#, c*x*y, c

def get_cars3d_dataloader(args, path_to_data=DATA_DIR.joinpath('cars3d')):
    """Cars3D dataloader with (64, 64, 3) images."""

    if not path_to_data.exists():
        print('Data at the given path doesn\'t exist. Downloading now...')
        os.system(f" mkdir {str(path_to_data)};")
        os.system(f" wget -O {str(path_to_data.joinpath('nips2015-analogy-data.tar.gz'))} http://www.scottreed.info/files/nips2015-analogy-data.tar.gz ;")
        os.system(f"tar xzf {str(path_to_data.joinpath('nips2015-analogy-data.tar.gz'))}")

    class cars3dDataset(Dataset):
        """Cars3D dataloader class
        The data set was first used in the paper "Deep Visual Analogy-Making"
        (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
        downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.
        The ground-truth factors of variation are:
        0 - elevation (4 different values) [0,3]
        1 - azimuth (24 different values) [0,23]
        2 - object type (183 different values) [0,182]
        Reference: Code adapted from
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/cars3d.py
        """
        lat_names = ('elevation', 'azimuth', 'object_type')
        lat_sizes = np.array([4, 24, 183])

        def __init__(self, path_to_data, subsample=1, transform=None):
            """
            Parameters
            ----------
            subsample : int
                Only load every |subsample| number of images.
            """
            from sklearn.utils.extmath import cartesian
            self.imgs = self._load_data()[::subsample]
            self.lv = cartesian([np.array(list(range(i))) for i in self.lat_sizes])[::subsample]
            self.transform = transform

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            if self.transform:
                sample = self.transform(self.imgs[idx])
            return sample.float(), self.lv[idx]

        def _load_data(self):
            dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
            for i, filename in enumerate(path_to_data.joinpath("data/cars").glob("*.mat")):
                data_mesh = self._load_mesh(filename)
                factor1 = np.array(list(range(4)))
                factor2 = np.array(list(range(24)))
                all_factors = np.transpose([
                    np.tile(factor1, len(factor2)),
                    np.repeat(factor2, len(factor1)),
                    np.tile(i,
                            len(factor1) * len(factor2))
                ])
                dataset[np.arange(i, 24 * 4 * 183, 183)] = data_mesh
            return dataset

        def _load_mesh(self, filename):
            """Parses a single source file and rescales contained images."""
            import scipy.io as sio
            import PIL
            mesh = np.einsum("abcde->deabc", sio.loadmat(filename)["im"])
            flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
            rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
            for i in range(flattened_mesh.shape[0]):
                pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
                pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
                rescaled_mesh[i, :, :, :] = np.array(pic)
            return rescaled_mesh * 1. / 255

    all_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = cars3dDataset(path_to_data, transform=all_transforms)

    if args.classes != -1:
        assert len(args.classes) == 3
        idx = [functools.reduce(lambda x,y:x|y, [np.isclose(train_data.lv[:,i], label) for label in classs]) for i, classs in enumerate(args.classes) if classs != -1]
        idx2 = [functools.reduce(lambda x,y:x|y, [np.isclose(train_data.lv[:,i], label) for label in np.unique(train_data.lv[:,i])]) for i, classs in enumerate(args.classes) if classs == -1]
        idx4 = functools.reduce(lambda x,y: x & y, idx+idx2)
        train_data.lv = train_data.lv[idx4]
        train_data.imgs = train_data.imgs[idx4]

    cars3d_loader = DataLoader(train_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(cars3d_loader))[0].size()
    return cars3d_loader#, c*x*y, c


def get_3dshapes_dataloader(args, path_to_data=DATA_DIR.joinpath('3dshapes')):
    """3dshapes dataloader with images rescaled to (28,28,3)"""

    name = '{}/3dshapes.h5'.format(path_to_data)
    if not os.path.exists(name):
        print('Data at the given path doesn\'t exist. ')
        os.system("  mkdir ~/data/3dshapes;"
                  "  wget -O ~/data/3dshapes/3dshapes.h5 https://storage.googleapis.com/3d-shapes/3dshapes.h5")
    from utils import Resize
    import h5py
    class d3shapesDataset(Dataset):
        """3dshapes dataloader class adapted from disentanglement_lib"""

        lat_names = ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
        lat_sizes = np.array([10, 10, 10, 8, 4, 15])

        def __init__(self, path_to_data, subsample=1, transform=None):
            """
            Parameters
            ----------
            subsample : int
                Only load every |subsample| number of images.
            """
            dataset = h5py.File(path_to_data, 'r')
            self.imgs = dataset['images'][::subsample]
            self.lv = dataset['labels'][::subsample]
            import inspect
            if "irs" in ''.join([str(f.filename) for f in inspect.stack()]):
                # The following three lines are to be used only for disentanglement evaluation
                # because the disent library uses different identifiers for the same generative factors
                for dim in range(self.lv.shape[1]):
                   u = list(np.unique(self.lv[:,dim]))
                   self.lv[:,dim] = np.array([u.index(self.lv[i, dim]) for i in range(self.lv.shape[0])])
            self.transform = transform

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            sample = self.imgs[idx] / 255
            if self.transform:
                sample = self.transform(sample)
            return sample, self.lv[idx]

    transform = transforms.Compose([Resize(28), transforms.ToTensor()])

    train_data = d3shapesDataset(name, transform=transform)

    if args.classes != -1:
        assert len(args.classes) == 6
        idx = [functools.reduce(lambda x,y:x|y, [np.isclose(train_data.lv[:,i], label) for label in classs]) for i, classs in enumerate(args.classes) if classs != -1]
        idx2 = [functools.reduce(lambda x,y:x|y, [np.isclose(train_data.lv[:,i], label) for label in np.unique(train_data.lv[:,i])]) for i, classs in enumerate(args.classes) if classs == -1]
        idx4 = functools.reduce(lambda x,y: x & y, idx+idx2)
        train_data.lv = train_data.lv[idx4]
        train_data.imgs = train_data.imgs[idx4]

    d3shapes_loader = DataLoader(train_data, batch_size=args.mb_size,
                                 shuffle=args.shuffle, pin_memory=True, num_workers=args.workers)
    _, c, x, y = next(iter(d3shapes_loader))[0].size()
    return d3shapes_loader#, c*x*y, c

def spiral(N: int, rand_state: np.random.RandomState, std, center=(0,0), b=1.0):
    """
    Returns N 2d points drawing an Archimedean spiral of center and b specified from 0 to 8pi.
    :param N: an even number of points
    :param state: random seed
    :param std: standard deviation of noise
    :return: a pair of (N,2) matrices, the true dataset and the noisy one
    """
    #assert N % 2 == 0
    theta = np.linspace(0, 8*np.pi, N)
    x = b * theta * np.cos(theta) + center[0]
    y = b * theta * np.sin(theta) + center[1]
    noise = rand_state.normal(0, std, 2*N)
    x_n = x + noise[:N]
    y_n = y + noise[N:]
    return (np.vstack([x,y]).T, np.vstack([x_n,y_n]).T)

class Spiral(Dataset):
    """
    Pytorch dataset for a spiral. Returns a pair of points.
    """
    def __init__(self, N:int, rand_state: np.random.RandomState, std: float):
        self.x, self.x_n = spiral(N, rand_state, std, b=0.2)
        self.x = torch.from_numpy(self.x)
        self.x_n = torch.from_numpy(self.x_n)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x_n[i], 1

class MultivariateGaussian(Dataset):
    """
    Pytorch dataset for a spiral. Returns a pair of points.
    """
    def __init__(self, N:int, rand_state: np.random.RandomState, std: float):
        cov = rand_state.normal(scale=std, size=(140,140))
        cov = cov @ cov.T
        self.x = rand_state.multivariate_normal([0.0 for _ in range(140)], cov, size=(N,))
        self.x = torch.from_numpy(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], 1


def square(N: int, rand_state: np.random.RandomState, std, bottom_left=(-1.0,-1.0), side=2.0):
    """
    Returns N 2d points drawing a square of bottom left vertex and side specified.
    :param N: an even number of points
    :param state: random seed
    :param std: standard deviation of noise
    :param bottom_left: coordinates of the bottom left vertex
    :param side: length of side
    :return: a pair of (N,2) matrices, the true dataset and the noisy one
    """
    assert N % 2 == 0
    points_x = np.linspace(bottom_left[0], bottom_left[0] + side, N//2)
    points_y = points_x - bottom_left[0] + bottom_left[1]
    a_x = np.empty((N//2,))
    a_x[::2] = bottom_left[0] + side #right
    a_x[1::2] = bottom_left[0] #left
    a_y = np.empty((N//2,))
    a_y[::2] = bottom_left[1] #bottom

    a_y[1::2] = bottom_left[1] + side #top
    x = np.concatenate([points_x, a_x])
    y = np.concatenate([a_y, points_y])
    noise = rand_state.normal(0, std, 2*N)
    x_n = x + noise[:N]
    y_n = y + noise[N:]
    return (np.vstack([x,y]).T, np.vstack([x_n,y_n]).T)

class SmallNORB(data.Dataset):
    """`SmallNORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small//>`_ Dataset.
    Args:
        root (string): Root directory of dataset where processed folder and
            and  raw folder exist.
        train (bool, optional): If True, creates dataset from the training files,
            otherwise from the test files.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If the dataset is already processed, it is not processed
            and downloaded again. If dataset is only already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        info_transform (callable, optional): A function/transform that takes in the
            info and transforms it.
        mode (string, optional): Denotes how the images in the data files are returned. Possible values:
            - all (default): both left and right are included separately.
            - stereo: left and right images are included as corresponding pairs.
            - left: only the left images are included.
            - right: only the right images are included.

    Code adapted from disentanglement_lib.
    """

    dataset_root = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    data_files = {
        'train': {
            'dat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
                "md5_gz": "66054832f9accfe74a0f4c36a75bc0a2",
                "md5": "8138a0902307b32dfa0025a36dfa45ec"
            },
            'info': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
                "md5_gz": "51dee1210a742582ff607dfd94e332e3",
                "md5": "19faee774120001fc7e17980d6960451"
            },
            'cat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
                "md5_gz": "23c8b86101fbf0904a000b43d3ed2fd9",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
        'test': {
            'dat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
                "md5_gz": "e4ad715691ed5a3a5f138751a4ceb071",
                "md5": "e9920b7f7b2869a8f1a12e945b2c166c"
            },
            'info': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
                "md5_gz": "a9454f3864d7fd4bb3ea7fc3eb84924e",
                "md5": "7c5b871cc69dcadec1bf6a18141f5edc"
            },
            'cat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
                "md5_gz": "5aa791cd7e6016cf957ce9bdb93b8603",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
    }

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_image_file = 'train_img'
    train_label_file = 'train_label'
    train_info_file = 'train_info'
    test_image_file = 'test_img'
    test_label_file = 'test_label'
    test_info_file = 'test_info'
    extension = '.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, info_transform=None, download=False,
                 mode="left"):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.info_transform = info_transform
        self.train = train  # training set or test set
        self.mode = mode

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # load test or train set
        image_file = self.train_image_file if self.train else self.test_image_file
        label_file = self.train_label_file if self.train else self.test_label_file
        info_file = self.train_info_file if self.train else self.test_info_file

        # load labels
        self.labels = self._load(label_file)

        # load info files
        self.infos = self._load(info_file)
        self.infos[:,2] = torch.LongTensor([int(x/2) for x in self.infos[:,2]])
        self.infos[:,0] = torch.LongTensor([int(x/2) for x in self.infos[:,0]])
        self.infos = torch.cat([self.labels.view(-1,1), self.infos], dim=1)

        # load right set
        if self.mode == "left":
            self.data = self._load("{}_left".format(image_file))

        # load left set
        elif self.mode == "right":
            self.data = self._load("{}_right".format(image_file))

        elif self.mode == "all" or self.mode == "stereo":
            left_data = self._load("{}_left".format(image_file))
            right_data = self._load("{}_right".format(image_file))

            # load stereo
            if self.mode == "stereo":
                self.data = torch.stack((left_data, right_data), dim=1)

            # load all
            else:
                self.data = torch.cat((left_data, right_data), dim=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            mode ``all'', ``left'', ``right'':
                tuple: (image, target, info)
            mode ``stereo'':
                tuple: (image left, image right, target, info)
        """
        target = self.labels[index % 24300] if self.mode == "all" else self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        info = self.infos[index % 24300] if self.mode == "all" else self.infos[index]
        if self.info_transform is not None:
            info = self.info_transform(info)

        if self.mode == "stereo":
            img_left = self._transform(self.data[index, 0])
            img_right = self._transform(self.data[index, 1])
            return img_left, img_right, target, info

        img = self._transform(self.data[index])
        return img, info

    def __len__(self):
        return len(self.data)

    def _transform(self, img):
        # doing this so that it is consistent with all other data sets
        # to return a PIL Image
        from PIL import Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load(self, file_name):
        return torch.load(os.path.join(self.root, self.processed_folder, file_name + self.extension))

    def _save(self, file, file_name):
        with open(os.path.join(self.root, self.processed_folder, file_name + self.extension), 'wb') as f:
            torch.save(file, f)

    def _check_exists(self):
        """ Check if processed files exists."""
        files = (
            "{}_left".format(self.train_image_file),
            "{}_right".format(self.train_image_file),
            "{}_left".format(self.test_image_file),
            "{}_right".format(self.test_image_file),
            self.test_label_file,
            self.train_label_file
        )
        fpaths = [os.path.exists(os.path.join(self.root, self.processed_folder, f + self.extension)) for f in files]
        return False not in fpaths

    def _flat_data_files(self):
        return [j for i in self.data_files.values() for j in list(i.values())]

    def _check_integrity(self):
        """Check if unpacked files have correct md5 sum."""
        from torchvision.datasets.utils import download_url, check_integrity
        root = self.root
        for file_dict in self._flat_data_files():
            filename = file_dict["name"]
            md5 = file_dict["md5"]
            fpath = os.path.join(root, self.raw_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        """Download the SmallNORB data if it doesn't exist in processed_folder already."""
        import gzip
        import errno
        from torchvision.datasets.utils import download_url, check_integrity

        if self._check_exists():
            return

        # check if already extracted and verified
        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            # download and extract
            for file_dict in self._flat_data_files():
                url = self.dataset_root + file_dict["name"] + '.gz'
                filename = file_dict["name"]
                gz_filename = filename + '.gz'
                md5 = file_dict["md5_gz"]
                fpath = os.path.join(self.root, self.raw_folder, filename)
                gz_fpath = fpath + '.gz'

                # download if compressed file not exists and verified
                download_url(url, os.path.join(self.root, self.raw_folder), gz_filename, md5)

                print('# Extracting data {}\n'.format(filename))

                with open(fpath, 'wb') as out_f, \
                        gzip.GzipFile(gz_fpath) as zip_f:
                    out_f.write(zip_f.read())

                os.unlink(gz_fpath)

        # process and save as torch files
        print('Processing...')

        # create processed folder
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # read train files
        left_train_img, right_train_img = self._read_image_file(self.data_files["train"]["dat"]["name"])
        train_info = self._read_info_file(self.data_files["train"]["info"]["name"])
        train_label = self._read_label_file(self.data_files["train"]["cat"]["name"])

        # read test files
        left_test_img, right_test_img = self._read_image_file(self.data_files["test"]["dat"]["name"])
        test_info = self._read_info_file(self.data_files["test"]["info"]["name"])
        test_label = self._read_label_file(self.data_files["test"]["cat"]["name"])

        # save training files
        self._save(left_train_img, "{}_left".format(self.train_image_file))
        self._save(right_train_img, "{}_right".format(self.train_image_file))
        self._save(train_label, self.train_label_file)
        self._save(train_info, self.train_info_file)

        # save test files
        self._save(left_test_img, "{}_left".format(self.test_image_file))
        self._save(right_test_img, "{}_right".format(self.test_image_file))
        self._save(test_label, self.test_label_file)
        self._save(test_info, self.test_info_file)

        print('Done!')

    @staticmethod
    def _parse_header(file_pointer):
        # Read magic number and ignore
        struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        return dimensions

    def _read_image_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300, 2, 96, 96]
            num_samples, _, height, width = dimensions

            left_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)
            right_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

            for i in range(num_samples):

                # left and right images stored in pairs, left first
                left_samples[i, :, :] = self._read_image(f, height, width)
                right_samples[i, :, :] = self._read_image(f, height, width)

        return torch.ByteTensor(left_samples), torch.ByteTensor(right_samples)

    @staticmethod
    def _read_image(file_pointer, height, width):
        """Read raw image data and restore shape as appropriate. """
        image = struct.unpack('<' + height * width * 'B', file_pointer.read(height * width))
        image = np.uint8(np.reshape(image, newshape=(height, width)))
        return image

    def _read_label_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300]
            num_samples = dimensions[0]

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            labels = np.zeros(shape=num_samples, dtype=np.int32)
            for i in range(num_samples):
                category, = struct.unpack('<i', f.read(4))
                labels[i] = category
            return torch.LongTensor(labels)

    def _read_info_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:

            dimensions = self._parse_header(f)
            assert dimensions == [24300, 4]
            num_samples, num_info = dimensions

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            infos = np.zeros(shape=(num_samples, num_info), dtype=np.int32)

            for r in range(num_samples):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    infos[r, c] = info

        return torch.LongTensor(infos)

# From https://stackoverflow.com/a/55593757/3830367
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transforms=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transforms is not None:
            x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]

def get_diabetes_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    import openml
    """Diabetes dataloader."""

    print("Loading Pima Indians Diabetes Database.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = openml.datasets.get_dataset(37)
    x, _, _, _ = dataset.get_data(dataset_format="array")
    y = torch.from_numpy(x[:,-1])
    x = torch.from_numpy(x[:,:-1])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    dataset = CustomTensorDataset([X_train, y_train] if args.train else [X_test, y_test])
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader

def get_ionosphere_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    import openml
    """Ionosphere dataloader."""

    print("Loading Ionosphere.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = openml.datasets.get_dataset(59)
    x, _, _, _ = dataset.get_data(dataset_format="array")
    y = torch.from_numpy(x[:,-1])
    x = torch.from_numpy(x[:,:-1])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    if args.train and args.val:
        data = [X_val, y_val]
    elif args.test:
        data = [X_test, y_test]
    else:
        data = [X_train, y_train]
    dataset = CustomTensorDataset(data)
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader

def get_cholesterol_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    import openml
    """Cholesterol dataloader."""

    print("Loading Cholesterol.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = openml.datasets.get_dataset(204)
    x, _, _, _ = dataset.get_data(dataset_format="array")
    y = torch.from_numpy(x[:,-1])
    x = torch.nan_to_num(torch.from_numpy(x[:,:-1]))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    if args.train and args.val:
        data = [X_val, y_val]
    elif args.test:
        data = [X_test, y_test]
    else:
        data = [X_train, y_train]
    dataset = CustomTensorDataset(data)
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader


def get_yacht_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    import openml
    """Yacht dataloader."""

    print("Loading Yacht.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = openml.datasets.get_dataset(42370)
    x, _, _, _ = dataset.get_data(dataset_format="array")
    y = torch.from_numpy(x[:,-1])
    x = torch.nan_to_num(torch.from_numpy(x[:,:-1]))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    if args.train and args.val:
        data = [X_val, y_val]
    elif args.test:
        data = [X_test, y_test]
    else:
        data = [X_train, y_train]
    dataset = CustomTensorDataset(data)
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader

def get_liver_dataloader(args, path_to_data=DATA_DIR.joinpath("mnist")):
    import openml
    """Liver dataloader."""

    print("Loading Liver.")
    all_transforms = transforms.Compose(args.pre_transformations + [
                                         transforms.ToTensor()] + args.post_transformations)
    dataset = openml.datasets.get_dataset(1480)
    x, _, _, _ = dataset.get_data(dataset_format="array")
    y = torch.from_numpy(x[:,-1])
    x = torch.from_numpy(x[:,:-1])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    if args.train and args.val:
        data = [X_val, y_val]
    elif args.test:
        data = [X_test, y_test]
    else:
        data = [X_train, y_train]
    dataset = CustomTensorDataset(data)
    train_loader = DataLoader(dataset, batch_size=args.mb_size, shuffle=args.shuffle,
                              pin_memory=True, num_workers=args.workers, worker_init_fn=seed_worker)
    return train_loader
