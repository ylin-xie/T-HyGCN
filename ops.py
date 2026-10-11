import torch


def index_points(points, idx):
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view(-1, *([1] * (idx.ndim - 1))).expand_as(idx)
    return points[batch, idx]


def knn_indices(xyz, k, query_xyz=None, chunk_size=512):
    if k < 1 or xyz.shape[1] < 1:
        raise ValueError('k and reference point count must be positive')
    same = query_xyz is None or (query_xyz.shape == xyz.shape and query_xyz.data_ptr() == xyz.data_ptr())
    query_xyz = xyz if query_xyz is None else query_xyz
    count = min(k, xyz.shape[1] - int(same))
    if count == 0:
        return torch.empty((*query_xyz.shape[:2], 0), dtype=torch.long, device=xyz.device)
    pieces = []
    with torch.no_grad():
        for start in range(0, query_xyz.shape[1], chunk_size):
            end = min(start + chunk_size, query_xyz.shape[1])
            distance = torch.cdist(query_xyz[:, start:end], xyz)
            if same:
                distance[:, torch.arange(end-start, device=xyz.device),
                         torch.arange(start, end, device=xyz.device)] = float('inf')
            pieces.append(distance.topk(count, largest=False).indices)
    return torch.cat(pieces, 1)


def farthest_point_sample(xyz, npoint):
    b, n, _ = xyz.shape
    if n < 1 or npoint < 1:
        raise ValueError('FPS requires a nonempty cloud and a positive sample size')
    npoint = min(npoint, n)
    out = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, n), float('inf'), device=xyz.device)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    used = torch.zeros(b, n, dtype=torch.bool, device=xyz.device)
    with torch.no_grad():
        for i in range(npoint):
            out[:, i] = farthest
            used[batch, farthest] = True
            centroid = xyz[batch, farthest].unsqueeze(1)
            dist = torch.minimum(dist, ((xyz - centroid) ** 2).sum(-1))
            farthest = dist.masked_fill(used, -1).max(-1).indices
    return out


def three_nn_interpolate(source_xyz, source_feat, target_xyz, chunk_size=512):
    k = min(3, source_xyz.shape[1])
    pieces = []
    for start in range(0, target_xyz.shape[1], chunk_size):
        d, idx = torch.cdist(target_xyz[:, start:start+chunk_size], source_xyz).topk(k, largest=False)
        exact = d <= 1e-8
        inverse = d.clamp_min(1e-8).reciprocal()
        w = inverse / inverse.sum(-1, keepdim=True)
        w = torch.where(exact.any(-1, keepdim=True), exact.to(w.dtype) / exact.sum(-1, keepdim=True).clamp_min(1), w)
        pieces.append((index_points(source_feat, idx) * w.unsqueeze(-1)).sum(-2))
    return torch.cat(pieces, 1)
