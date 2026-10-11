import math
import torch


def expmap0(x, c=1.0):
    if c <= 0:
        raise ValueError('c must be positive')
    s = math.sqrt(c)
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    result = torch.tanh(s * n) * x / (s * n)
    radius = result.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    limit = (1 - 1e-5) / s
    return result * (limit / radius).clamp_max(1.0)


def logmap0(x, c=1.0):
    if c <= 0:
        raise ValueError('c must be positive')
    s = math.sqrt(c)
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    clipped = n.clamp_max((1 - 1e-5) / s)
    return torch.atanh(s * clipped) * x / (s * n)


def poincare_distance(x, y, c=1.0):
    if c <= 0:
        raise ValueError('c must be positive')
    x2, y2 = (x * x).sum(-1), (y * y).sum(-1)
    eps = torch.finfo(x.dtype).eps
    denom = ((1 - c * x2).clamp_min(eps) * (1 - c * y2).clamp_min(eps))
    norm = torch.linalg.vector_norm(x - y, dim=-1)
    return 2 * torch.asinh(math.sqrt(c) * norm / torch.sqrt(denom)) / math.sqrt(c)
