import collections
import geoopt
import torch
from torch.optim import Optimizer
import numpy as np
from definitions import device
from torch import nn

def merge_two_dicts(x, y):
    # In case of same key, it keeps the value of y
    return {**x, **y}


def merge_dicts(list_of_dicts):
    from functools import reduce
    return reduce(merge_two_dicts, list_of_dicts)


def flatten_dict(d, parent_key='', sep='_', prefix='eval_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep, prefix=prefix).items())
        else:
            items.append((prefix + new_key, v))
    return dict(items)


def float_format(f: float) -> str:
    return "%+.4e" % f


def orto(h, full=False):
    o = torch.mm(h.t(), h)
    o = o - torch.eye(*o.shape, device=o.device)
    n = torch.linalg.norm(o, "fro")
    return torch.pow(n, 2)

def kPCA(X, h_n, k=None):
    a = k(X.t())
    nh1 = X.shape[0]
    oneN = torch.div(torch.ones(nh1, nh1).cpu(), nh1).to(device)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)
    h, s, _ = torch.svd(a, some=False)
    return h[:, : h_n], s


def get_params(p):
    if type(p) != list:
        return p
    if len(p) == 0:
        return p
    if type(p[0]) == torch.Tensor or type(p[0]) == geoopt.tensor.ManifoldParameter:
        return p
    else:
        from functools import reduce
        aa = [temp["params"] for temp in p]
        return reduce(lambda x, y: x+y, aa)

# From https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
class ReduceLROnPlateau():
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()
        self.epochs_since_reduction = 0

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.epochs_since_reduction = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            # print(f"Not better: now is {current} and best is {self.best}")
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.epochs_since_reduction = 0
        else:
            self.epochs_since_reduction += 1

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            if self.threshold < 0:
                rel_epsilon = 1. - self.threshold * 1.02 ** (-self.epochs_since_reduction)
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        from torch._six import inf
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

class Lin_View(nn.Module):
    """ Unflatten linear layer to be used in Convolution layer"""

    def __init__(self, c, a, b):
        super(Lin_View, self).__init__()
        self.c, self.a, self.b = c, a, b

    def forward(self, x):
        try:
            return x.view(x.size(0), self.c, self.a, self.b)
        except:
            return x.view(1, self.c, self.a, self.b)

class Resize:
    def __init__(self, size):
        from collections.abc import Iterable
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        import skimage
        from skimage.transform import resize
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float32 array
        return skimage.util.img_as_float32(resize_image)
