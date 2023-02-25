import logging
import torch
from torch.optim import Optimizer
from enum import IntEnum
import utils
from functools import reduce
import numpy as np

class ProjectedGradient(Optimizer):
    def __init__(self, params, lr=1.0, beta=0.5):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError("Invalid beta: {}".format(lr))
        defaults = dict(lr=lr,beta=beta)
        super(ProjectedGradient, self).__init__(params, defaults)

        if len(self.param_groups) > 2:
            raise ValueError("ProjectedGradient doesn't support more than two per-parameter options "
                             "(parameter groups)")
        # for i in range(len(self.param_groups)):
        #     if len(self.param_groups[i]['params']) > 1:
        #         raise ValueError("ProjectedGradient doesn't support multiple parameters")

        i = 0 if self.param_groups[0]['stiefel'] else 1
        # _param = self.param_groups[i]['params'][0]
        # assert _param.shape[0] < _param.shape[1] # it's s x N

        self.alpha = 1000
        self.u, self.s, self.v = None, None, None

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        lr_stiefel = 1
        for group in self.param_groups:
            stiefel = group['stiefel']
            for j in range(len(group['params'])):
                if stiefel:
                    p = torch.clone(group['params'][j]).t()  # use H as N x s
                    d, N = p.shape
                    g = torch.clone(group['params'][j].grad).t()  # use H as N x s
                    lr = group['lr']
                    beta = group['beta']
                    loss = float(closure())

                    # Backtracking line-search
                    p1 = None
                    if self.u is None:
                        self.u, self.s, self.v = torch.empty((d, d), device=p.device), torch.empty((d,), device=p.device), torch.empty((N, d), device=p.device)
                    u, s, v = self.u, self.s, self.v
                    i = 0
                    while i < 1000:
                        p1 = self.st_project(p - lr * g, out=(u,s,v))
                        group['params'][j][:] = p1.t()[:]
                        loss1 = float(closure())
                        condition = loss1 <= loss + torch.trace(torch.mm(g.t(), (p1 - p))) + torch.trace(torch.mm(p1 - p, (p1 - p).t())) / (2 * lr)
                        if condition:
                            break
                        else:
                            lr *= beta
                        i += 1
                    lr_stiefel = lr
                    group['params'][j][:] = p1.t()[:]
                else:
                    # continue
                    p = torch.clone(group['params'][j]).t()  # use H as N x s
                    g = torch.clone(group['params'][j].grad).t()  # use H as N x s
                    lr = group['lr']
                    beta = group['beta']
                    loss = float(closure())

                    # Backtracking line-search
                    p1 = None
                    i = 0
                    while i < 1000:
                        p1 = p - lr * g
                        group['params'][j][:] = p1.t()[:]
                        loss1 = float(closure())
                        def trace(x):
                            if len(x.shape) > 1:
                                return torch.trace(x)
                            else:
                                assert x.flatten().shape[0] == 1
                                return torch.sum(x)
                        condition = loss1 <= loss + trace(torch.matmul(g.t(), (p1 - p))) + trace(torch.matmul(p1 - p, (p1 - p).t())) / (2 * lr)
                        if condition:
                            break
                        else:
                            lr *= beta
                        i += 1
                    p1 = torch.maximum(p1, torch.zeros(p1.shape, device=p1.device))
                    group['params'][j][:] = p1.t()[:]

        loss = float(closure())
        self.alpha = lr_stiefel
        return loss, lr_stiefel

    @staticmethod
    def st_project(x, out=None):
        u, s, v = torch.svd(x)
        return torch.mm(u, v.t())
