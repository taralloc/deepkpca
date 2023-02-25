import torch
from torch import nn
from torch.nn import Parameter

Tensor = torch.Tensor

def kernel_factory(name: str, param: dict):
    assert name in ["rbf", "linear", "poly"]
    kernel = None
    if name == "rbf":
        kernel = GaussianKernelTorch(**param)
    elif name == "poly":
        kernel = PolyKernelTorch(**param)
    elif name == "linear":
        kernel = LinearKernel()
    return kernel

class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        return torch.mm(X.t(), Y)

    def phi_inv(self, x):
        return x

    def phi(self, x):
        return x

class GaussianKernelTorch(nn.Module):
    def __init__(self, sigma2=50.0):
        super(GaussianKernelTorch, self).__init__()
        if type(sigma2) == float:
            self.sigma2 = Parameter(torch.tensor(float(sigma2)), requires_grad=False)
            self.register_parameter("sigma2", self.sigma2)
        else:
            self.sigma2 = sigma2

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        def my_cdist(x1, x2):
            """
            Computes a matrix of the norm of the difference.
            """
            x1 = torch.t(x1)
            x2 = torch.t(x2)
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
            res = res.clamp_min_(1e-30).sqrt_()
            return res

        D = my_cdist(X,Y)

        return torch.exp(- torch.pow(D, 2) / (self.sigma2))

    def phi_inv(self, x):
        raise NotImplementedError()

class PolyKernelTorch(nn.Module):
    def __init__(self, d: int, t=1.0) -> None:
        super().__init__()
        self.d = d
        self.c = t

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        return torch.pow(torch.matmul(X.t(), Y), self.d)

    def phi(self, x):
        raise NotImplementedError()

    def phi_inv(self, x):
        raise NotImplementedError()
