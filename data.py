from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


def _dimension_map(las):
    return {str(name).lower(): str(name) for name in las.point_format.dimension_names}


def _get_dimension(las, names, required=False):
    dims = _dimension_map(las)
    for name in names:
        if name.lower() in dims:
            return np.asarray(las[dims[name.lower()]])
    if required:
        raise ValueError(f"Missing LAS dimension; expected one of: {', '.join(names)}")
    return None


def read_las(path):
    las = laspy.read(str(path))
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    channels = [_get_dimension(las, (n,)) for n in ("red", "green", "blue")]
    if all(c is not None for c in channels):
        rgb = np.column_stack(channels).astype(np.float32)
        if rgb.max(initial=0) > 255:
            rgb /= 65535.0
        else:
            rgb /= 255.0
    else:
        raise ValueError("Input requires RGB dimensions; supply colorized LAS/LAZ")
    rgb = np.clip(rgb, 0, 1)
    return las, xyz, rgb


def read_labels(las, semantic_name="semantic", instance_name="instance_id", head_value=0):
    semantic = _get_dimension(las, (semantic_name,), required=True)
    instance = _get_dimension(las, (instance_name,), required=True)
    classes = np.where(semantic == head_value, 0, 1).astype(np.int64)
    instance = instance.astype(np.int64)
    instance[classes != 0] = -1
    return classes, instance


def estimate_normals(xyz, k=20):
    k = min(k, len(xyz))
    _, ids = cKDTree(xyz).query(xyz, k=k)
    if ids.ndim == 1:
        ids = ids[:, None]
    local = xyz[ids] - xyz[ids].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", local, local) / max(ids.shape[1] - 1, 1)
    normal = np.linalg.eigh(cov)[1][:, :, 0]
    normal *= np.where(normal[:, 2:3] < 0, -1, 1)
    return normal.astype(np.float32)


def make_features(xyz, rgb):
    return np.concatenate((xyz.astype(np.float64), rgb.astype(np.float64)), axis=1)


def normalize(points):
    if len(points) == 0:
        raise ValueError('Cannot normalize an empty cloud')
    points = np.asarray(points, dtype=np.float64)
    center = points[:, :3].mean(0)
    scale = np.linalg.norm(points[:, :3] - center, axis=1).max()
    out = points.copy()
    out[:, :3] = (out[:, :3] - center) / max(float(scale), 1e-8)
    return out.astype(np.float32), center.astype(np.float64), float(scale)


def fixed_point_indices(xyz, count, rng):
    n = len(xyz)
    if n == 0 or count < 1:
        raise ValueError('FPS requires a nonempty cloud and positive sample size')
    if n < count:
        base = np.arange(n, dtype=np.int64)
        extra = rng.choice(n, count - n, replace=True)
        return np.concatenate((base, extra))
    if n == count:
        return np.arange(n, dtype=np.int64)
    selected = np.empty(count, dtype=np.int64)
    min_dist = np.full(n, np.inf, dtype=np.float64)
    current = int(rng.integers(n))
    xyz64 = xyz.astype(np.float64, copy=False)
    used = np.zeros(n, dtype=bool)
    for i in range(count):
        selected[i] = current
        min_dist = np.minimum(min_dist, np.sum((xyz64 - xyz64[current]) ** 2, axis=1))
        used[current] = True
        current = int(np.argmax(np.where(used, -1, min_dist)))
    return selected


class LasSemanticDataset(Dataset):
    def __init__(self, data_dir, points_per_block=8192, semantic_name="semantic",
                 instance_name="instance_id", head_value=0, augment=False, seed=42):
        self.files = sorted([*Path(data_dir).glob("*.las"), *Path(data_dir).glob("*.laz")])
        if not self.files:
            raise FileNotFoundError(f"No LAS/LAZ blocks found in {data_dir}")
        self.points_per_block = int(points_per_block)
        self.semantic_name, self.instance_name = semantic_name, instance_name
        self.head_value, self.augment, self.seed = head_value, augment, seed
        self.epoch = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        las, xyz, rgb = read_las(path)
        semantic, instance = read_labels(las, self.semantic_name, self.instance_name, self.head_value)
        rng = np.random.default_rng(self.seed + index)
        ids = fixed_point_indices(xyz, self.points_per_block, rng)
        points, semantic, instance = make_features(xyz[ids], rgb[ids]), semantic[ids], instance[ids]
        if self.augment:
            angle = rng.uniform(-np.pi / 4, np.pi / 4)
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
            points[:, :3] = points[:, :3] @ rot.T
            points[:, :3] += np.clip(rng.normal(0, .01, (len(points), 3)), -.05, .05)
        normalized, center, scale = normalize(points)
        return normalized.astype(np.float32), semantic, instance, str(path), center.astype(np.float64), np.float64(scale)
