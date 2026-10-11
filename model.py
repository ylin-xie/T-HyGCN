import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .hyperbolic import expmap0, logmap0, poincare_distance
from .ops import farthest_point_sample, index_points, knn_indices, three_nn_interpolate


def _kd_layout(count):

    queue = [(list(range(count)), 0, 1)]
    parents, depths = [-1], [0]
    intervals = []
    while queue:
        ids, parent, depth = queue.pop(0)
        if not ids:
            continue
        node = len(parents)
        parents.append(parent); depths.append(depth)
        intervals.append(ids)
        middle = len(ids) // 2
        queue.extend([(ids[:middle], node, depth + 1),
                      (ids[middle + 1:], node, depth + 1)])
    return parents, depths


def _recursive_kd_order(neighbor_xyz):
    b, n, k, _ = neighbor_xyz.shape
    values = neighbor_xyz.detach().cpu().numpy()
    orders = np.empty((b, n, k), dtype=np.int64)
    for bi in range(b):
        for pi in range(n):
            queue, ordered = [list(range(k))], []
            while queue:
                ids = queue.pop(0)
                if not ids:
                    continue
                subset = values[bi, pi, ids]
                axis = int(np.argmax(np.ptp(subset, axis=0)))
                ids = sorted(ids, key=lambda i: (values[bi, pi, i, axis], i))
                mid = len(ids) // 2
                ordered.append(ids[mid])
                queue.extend([ids[:mid], ids[mid + 1:]])
            orders[bi, pi] = ordered
    return torch.as_tensor(orders, device=neighbor_xyz.device), *_kd_layout(k)


def _rank_layout(count):
    parents = [-1] + [(i - 1) // 2 for i in range(1, count + 1)]
    depths = [0]
    for p in parents[1:]:
        depths.append(depths[p] + 1)
    return parents, depths


class TreeGCN(nn.Module):
    def __init__(self, in_channels, channels, iterations=2, heads=4, hyperbolic=True):
        super().__init__()
        self.iterations, self.hyperbolic = iterations, hyperbolic
        heads = max(d for d in range(1, heads + 1) if channels % d == 0)
        self.input_project = nn.Linear(in_channels, channels)
        self.level_gru = nn.GRU(channels, channels, batch_first=True)
        self.level_gate = nn.Linear(channels, channels)
        self.child_transform = nn.Linear(channels, channels, bias=False)
        self.bottom_norm = nn.LayerNorm(channels)
        self.root_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.root_norm = nn.LayerNorm(channels)

    def forward(self, root, children, tree_order, parents, depths):
        b, n, k, _ = children.shape
        projected = self.input_project(torch.cat((root.unsqueeze(2), children), 2))
        c = projected.shape[-1]
        ordered = torch.gather(projected[:, :, 1:], 2,
                               tree_order.unsqueeze(-1).expand(-1, -1, -1, c))
        z = torch.cat((projected[:, :, :1], ordered), 2).reshape(b * n, k + 1, c)
        e = expmap0(z) if self.hyperbolic else z
        for iteration in range(self.iterations):
            z = logmap0(e) if self.hyperbolic else e
            child_ids = [[q for q in range(1, k + 1) if parents[q] == v]
                         for v in range(k + 1)]
            weights = {}
            for v, ids in enumerate(child_ids):
                if ids:
                    distance = (poincare_distance(e[:, v:v + 1], e[:, ids])
                                if self.hyperbolic else
                                torch.linalg.vector_norm(e[:, v:v + 1] - e[:, ids], dim=-1))
                    weights[v] = torch.softmax(-distance, -1)
            updated = [z[:, 0]] + [None] * k
            levels = []
            for depth in range(1, max(depths) + 1):
                ids = [v for v in range(1, k + 1) if depths[v] == depth]
                u, h = self.level_gru(z[:, ids])
                levels.append(torch.sigmoid(self.level_gate(h[-1])) * h[-1])
                for j, v in enumerate(ids):
                    updated[v] = u[:, j]
            aggregated = [None] * (k + 1)
            for v in sorted(range(k + 1), key=lambda v: depths[v], reverse=True):
                ids = child_ids[v]
                if not ids:
                    aggregated[v] = updated[v]
                else:
                    messages = self.child_transform(torch.stack([aggregated[q] for q in ids], 1))
                    aggregate = (weights[v].unsqueeze(-1) * messages).sum(1)
                    aggregated[v] = self.bottom_norm(F.relu(updated[v] + aggregate))
            q = aggregated[0].unsqueeze(1)
            kv = torch.cat((q, torch.stack(levels, 1)), 1) if levels else q
            attention, _ = self.root_attention(q, kv, kv, need_weights=False)
            output_root = self.root_norm(F.relu(attention[:, 0]))
            z = torch.stack([output_root] + aggregated[1:], 1)
            if iteration + 1 < self.iterations:
                e = expmap0(z) if self.hyperbolic else z
        return output_root.reshape(b, n, c)


class StandardGCN(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_channels, channels), nn.Linear(channels, channels)])

    def forward(self, root, children, order, parents, depths):
        c = children.shape[-1]
        ordered = torch.gather(children, 2, order.unsqueeze(-1).expand(-1, -1, -1, c))
        x = torch.cat((root.unsqueeze(2), ordered), 2)
        size = len(parents)
        a = torch.eye(size, device=x.device, dtype=x.dtype)
        for v in range(1, size):
            a[v, parents[v]] = a[parents[v], v] = 1
        d = a.sum(-1).rsqrt()
        a = d[:, None] * a * d[None, :]
        for layer in self.layers:
            x = F.relu(layer(torch.einsum('ij,bnjc->bnic', a, x)))
        return x[:, :, 0]


VARIANTS = {
    'full': (True, True, True, True),
    'baseline': (False, False, False, False),
    'A1': (False, True, False, False),
    'A2': (False, True, True, False),
    'A3': (True, False, False, False),
    'A4': (True, True, True, False),
}


class DTHA(nn.Module):
    def __init__(self, in_channels, out_channels, neighbours=32, iterations=2, variant='full'):
        super().__init__()
        self.neighbours = neighbours
        self.dual, tree, hyperbolic, self.adaptive = VARIANTS[variant]
        def branch():
            return (TreeGCN(in_channels, out_channels, iterations, hyperbolic=hyperbolic)
                    if tree else StandardGCN(in_channels, out_channels))
        self.kd_tree = branch()
        self.kd_residual = nn.Linear(out_channels, out_channels)
        if self.dual:
            self.rank_tree = branch()
            self.rank_residual = nn.Linear(out_channels, out_channels)
        if self.adaptive:
            self.gate = nn.Linear(out_channels * 2, 1, bias=False)

    def forward(self, xyz, features):
        idx = knn_indices(xyz, self.neighbours)
        neighbor_f, neighbor_xyz = index_points(features, idx), index_points(xyz, idx)
        kd_order, parents, depths = _recursive_kd_order(neighbor_xyz)
        kd = self.kd_tree(features, neighbor_f, kd_order, parents, depths)
        kd = kd + F.relu(self.kd_residual(kd))
        if not self.dual:
            return kd
        numerator = (features.unsqueeze(2) * neighbor_f).sum(-1)
        denominator = features.norm(dim=-1, keepdim=True) * neighbor_f.norm(dim=-1) + 1e-8
        order = torch.argsort(numerator / denominator, dim=-1, descending=True, stable=True)
        parents, depths = _rank_layout(idx.shape[-1])
        rank = self.rank_tree(features, neighbor_f, order, parents, depths)
        rank = rank + F.relu(self.rank_residual(rank))
        alpha = torch.sigmoid(self.gate(torch.cat((kd, rank), -1))) if self.adaptive else 0.5
        return alpha * kd + (1 - alpha) * rank


class DTHAP(nn.Module):
    def __init__(self, in_channels, out_channels, neighbours, iterations, variant='full'):
        super().__init__()
        self.dtha = DTHA(in_channels, out_channels, neighbours, iterations, variant)

    def forward(self, xyz, features):
        f = self.dtha(xyz, features)
        ids = farthest_point_sample(xyz, max(1, xyz.shape[1] // 4))
        return index_points(xyz, ids), index_points(f, ids)


class THyGCN(nn.Module):
    def __init__(self, input_channels=6, classes=2, neighbours=32, iterations=2, variant='full'):
        super().__init__()
        self.dtha1 = DTHA(input_channels, 64, neighbours, iterations, variant)
        self.dtha2 = DTHA(64, 64, neighbours, iterations, variant)
        self.pool1 = DTHAP(64, 128, neighbours, iterations, variant)
        self.pool2 = DTHAP(128, 256, neighbours, iterations, variant)
        self.pool3 = DTHAP(256, 512, neighbours, iterations, variant)
        self.global_mlp = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.LayerNorm(1024))
        self.embedding = nn.Sequential(nn.Linear(1984, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.classifier = nn.Linear(256, classes)

    def forward(self, points):
        xyz = points[..., :3]
        f1 = self.dtha1(xyz, points)
        f2 = self.dtha2(xyz, f1)
        x1, f3 = self.pool1(xyz, f2)
        x2, f4 = self.pool2(x1, f3)
        x3, f5 = self.pool3(x2, f4)
        global_feature = self.global_mlp(f5).amax(1, keepdim=True).expand(-1, xyz.shape[1], -1)
        fused = torch.cat((f2, three_nn_interpolate(x1, f3, xyz),
                           three_nn_interpolate(x2, f4, xyz),
                           three_nn_interpolate(x3, f5, xyz), global_feature), -1)
        embedding = self.embedding(fused)
        return self.classifier(embedding), embedding
