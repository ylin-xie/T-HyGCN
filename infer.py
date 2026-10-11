import argparse
from pathlib import Path

import numpy as np
import torch

from .data import fixed_point_indices, make_features, normalize, read_las
from .checkpoint import load_model
from .pipeline import segment_instances_and_volume, write_results


@torch.no_grad()
def predict_block(model, points, num_points, device):
    ids = fixed_point_indices(points[:, :3], num_points, np.random.default_rng(42))
    sampled, _, _ = normalize(points[ids])
    logits = model(torch.from_numpy(sampled).unsqueeze(0).to(device))[0][0].cpu().numpy()
    unique, inverse = np.unique(ids, return_inverse=True)
    sums = np.zeros((len(unique), logits.shape[-1]), np.float64)
    counts = np.zeros(len(unique), np.float64)
    np.add.at(sums, inverse, logits); np.add.at(counts, inverse, 1)
    return unique, (sums / counts[:, None]).argmax(-1)


@torch.no_grad()
def predict_full_cloud(model, points, num_points, device):

    from scipy.spatial import cKDTree
    ids, labels = predict_block(model, points, num_points, device)
    if len(ids) == len(points):
        return labels
    nearest = cKDTree(points[ids, :3]).query(points[:, :3])[1]
    return labels[nearest]


def main():
    p = argparse.ArgumentParser(description="Predict cabbage head masks, instances, and volumes")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-points", type=int, default=8192)
    p.add_argument("--dense-output", action="store_true", help="Optional nearest-point lifting beyond paper protocol")
    p.add_argument("--max-clusters", type=int, default=30)
    p.add_argument("--min-fragment-points", type=int, default=100)
    p.add_argument("--semantic-dim", default="pred_semantic")
    p.add_argument("--instance-dim", default="pred_instance")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.checkpoint, device)
    las, xyz, rgb = read_las(a.input)
    features = make_features(xyz, rgb)
    if a.dense_output:
        semantic = predict_full_cloud(model, features, a.num_points, device)
        ids = np.arange(len(xyz))
    else:
        ids, semantic = predict_block(model, features, a.num_points, device)
    xyz, rgb = xyz[ids], rgb[ids]
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'sampled_input.las'
    las.points = las.points[ids]
    las.write(source)
    instances, rows = segment_instances_and_volume(xyz, semantic, a.max_clusters, a.min_fragment_points)
    for row in rows:
        row["reference_volume_m3"] = ""
        row["volume_error_m3"] = ""
    output = Path(a.output_dir)
    write_results(output, source, xyz, rgb, semantic, instances, rows,
                  a.semantic_dim, a.instance_dim)
    print(f"points={len(xyz)} predicted_heads={len(rows)} output={output}")


if __name__ == "__main__":
    main()
