import torch
from torch.nn.functional import softplus


class LipschitzNorm(torch.nn.Module):
    name: str
    dim: int

    def __init__(self, name, dim: int, weight) -> None:
        super().__init__()
        self.name = name
        self.dim = dim
        self.register_parameter(
            "lipschitz_constant",
            torch.nn.Parameter(torch.max(torch.sum(torch.abs(weight), dim))),
        )

    def compute_weight(self, module):
        W = getattr(module, self.name)
        absrowsum = torch.sum(torch.abs(W), dim=self.dim)
        softplus_c = softplus(self.lipschitz_constant)
        scale = torch.minimum(torch.Tensor([1.0]).to(W.device), softplus_c / absrowsum)
        return torch.nn.Parameter(W * scale[:, None])

    @staticmethod
    def apply(module, name: str, dim: int = -1) -> "LipschitzNorm":
        weight = getattr(module, name)
        fn = LipschitzNorm(name, dim, weight)
        setattr(module, name, fn.compute_weight(module))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def lipschitz_norm(module, name: str = "weight", dim: int = 1):
    lipschitz_norm_instance = LipschitzNorm.apply(module, name, dim)
    return module, lipschitz_norm_instance
