import csv
from pathlib import Path

import laspy
import numpy as np
from kneed import KneeLocator
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans


def choose_k(xy, max_k=30):
    if max_k < 1:
        raise ValueError("max_k must be positive")
    upper = min(max_k, len(np.unique(xy, axis=0)))
    if upper < 2:
        return 1
    ks = list(range(1, upper + 1))
    wcss = [KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(xy).inertia_ for k in ks]
    knee = KneeLocator(ks, wcss, curve="convex", direction="decreasing").knee
    return int(knee) if knee is not None else 1


def convex_hull_volume(xyz):
    if len(xyz) < 4:
        return 0.0
    try:
        hull = ConvexHull(xyz)
    except QhullError:
        return 0.0
    vertices = xyz[hull.vertices]
    origin = vertices.mean(axis=0)
    triangles = xyz[hull.simplices] - origin
    return float(np.abs(np.einsum("ij,ij->i", triangles[:, 0],
                                  np.cross(triangles[:, 1], triangles[:, 2]))).sum() / 6.0)


def segment_instances_and_volume(xyz, semantic, max_k=30, min_fragment_points=100):
    result = np.full(len(xyz), -1, np.int64)
    foreground = np.flatnonzero(semantic == 0)
    if len(foreground) < 4:
        return result, []
    head_xyz = xyz[foreground]
    k = choose_k(head_xyz[:, :2], max_k)
    local = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit_predict(head_xyz[:, :2])
    counts = np.bincount(local, minlength=k)
    valid = np.flatnonzero(counts >= min_fragment_points)
    if len(valid) == 0:
        return result, []
    centers = np.stack([head_xyz[local == i, :2].mean(0) for i in range(k)])
    remap = np.full(k, -1, dtype=np.int64)
    for new_id, old_id in enumerate(valid):
        remap[old_id] = new_id
    for old_id in range(k):
        if remap[old_id] < 0:
            remap[old_id] = int(np.argmin(np.linalg.norm(centers[valid] - centers[old_id], axis=1)))
    labels = remap[local]
    result[foreground] = labels
    rows = []
    for instance_id in range(len(valid)):
        cluster = xyz[result == instance_id]
        rows.append({"instance_id": instance_id, "points": int(len(cluster)),
                     "volume_m3": convex_hull_volume(cluster)})
    return result, rows


def write_results(output_dir, source_path, original_xyz, rgb, semantic, instances, rows,
                  semantic_dimension="pred_semantic", instance_dimension="pred_instance"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", xyz=original_xyz, rgb=rgb,
                        semantic=semantic, instance=instances)
    with (output / "instances.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0]) if rows else ["instance_id", "points", "volume_m3"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    las = laspy.read(str(source_path))
    names = {str(n).lower() for n in las.point_format.dimension_names}
    if semantic_dimension.lower() not in names:
        las.add_extra_dim(laspy.ExtraBytesParams(name=semantic_dimension, type=np.uint8,
                                                 description="Head=0, background=1"))
    if instance_dimension.lower() not in names:
        las.add_extra_dim(laspy.ExtraBytesParams(name=instance_dimension, type=np.int32,
                                                 description="Head instance; background=-1"))
    dims = {str(n).lower(): str(n) for n in las.point_format.dimension_names}
    las[dims[semantic_dimension.lower()]] = semantic.astype(np.uint8)
    las[dims[instance_dimension.lower()]] = instances.astype(np.int32)
    las.write(output / f"{Path(source_path).stem}_predicted.las")


def instance_metrics(gt_instance, pred_instance, iou_threshold=0.5):
    gt_ids = np.unique(gt_instance[gt_instance >= 0])
    pred_ids = np.unique(pred_instance[pred_instance >= 0])
    if not len(gt_ids) or not len(pred_ids):
        return {"mPrec": 0.0, "mRec": 0.0, "mCov": 0.0, "mWCov": 0.0}
    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for i, gid in enumerate(gt_ids):
        g = gt_instance == gid
        for j, pid in enumerate(pred_ids):
            p = pred_instance == pid
            intersection = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            iou[i, j] = intersection / max(union, 1)
    pairs = sorted(((iou[i, j], i, j) for i in range(len(gt_ids)) for j in range(len(pred_ids))), reverse=True)
    used_g, used_p, matched = set(), set(), []
    for score, i, j in pairs:
        if score < iou_threshold:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); matched.append((i, j, score))
    tp = len(matched)
    max_iou = iou.max(axis=1)
    weights = np.array([(gt_instance == gid).sum() for gid in gt_ids], dtype=np.float64)
    weights /= max(weights.sum(), 1)
    return {"mPrec": tp / len(pred_ids), "mRec": tp / len(gt_ids),
            "mCov": float(max_iou.mean()), "mWCov": float(np.sum(max_iou * weights))}


def volume_metrics(pred_rows, reference_by_instance):
    pairs = [(float(row["volume_m3"]), float(reference_by_instance[row["instance_id"]]))
             for row in pred_rows if row["instance_id"] in reference_by_instance]
    if not pairs:
        return {"R2": float("nan"), "RMSE": float("nan"), "matched_heads": 0}
    estimate, reference = np.asarray(pairs).T
    ss_res = np.sum((estimate - reference) ** 2)
    ss_tot = np.sum((reference - reference.mean()) ** 2)
    return {"R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "RMSE": float(np.sqrt(np.mean((estimate - reference) ** 2))), "matched_heads": len(pairs)}


def match_instances(gt_instance, pred_instance, iou_threshold=0.5):
    gt_ids = np.unique(gt_instance[gt_instance >= 0])
    pred_ids = np.unique(pred_instance[pred_instance >= 0])
    if not len(gt_ids) or not len(pred_ids):
        return {}
    scores = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for gi, gid in enumerate(gt_ids):
        g = gt_instance == gid
        for pi, pid in enumerate(pred_ids):
            p = pred_instance == pid
            union = np.logical_or(g, p).sum()
            scores[gi, pi] = np.logical_and(g, p).sum() / max(union, 1)
    pairs = sorted(((scores[i, j], i, j) for i in range(len(gt_ids)) for j in range(len(pred_ids))), reverse=True)
    matched = {}
    used_gt, used_pred = set(), set()
    for score, gi, pi in pairs:
        if score < iou_threshold:
            break
        if gi not in used_gt and pi not in used_pred:
            matched[int(pred_ids[pi])] = int(gt_ids[gi])
            used_gt.add(gi)
            used_pred.add(pi)
    return matched
