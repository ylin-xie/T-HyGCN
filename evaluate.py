import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from .data import make_features, normalize, read_labels, read_las
from .checkpoint import load_model
from .split import validate_manifest
from .infer import predict_block
from .pipeline import instance_metrics, match_instances, segment_instances_and_volume, volume_metrics, convex_hull_volume


def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on labeled LAS/LAZ blocks")
    p.add_argument("--checkpoint", required=True); p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--semantic-dim", default="semantic")
    p.add_argument("--instance-dim", default="instance_id"); p.add_argument("--head-value", type=int, default=0)
    p.add_argument("--num-points", type=int, default=8192); p.add_argument("--max-clusters", type=int, default=30)
    p.add_argument("--min-fragment-points", type=int, default=100)
    p.add_argument("--reference-table")
    p.add_argument("--split-manifest", help="Validate test membership and provide T1--T4 grouping")
    p.add_argument("--id-column", default="instance_id")
    p.add_argument("--block-column", default="block")
    p.add_argument("--length-column", default="length")
    p.add_argument("--width-column", default="width")
    p.add_argument("--height-column", default="height")
    p.add_argument("--dimension-unit", choices=["m", "cm", "mm"], default="mm")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.checkpoint, device)
    output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    volume_csv = output / 'ground_truth_instance_volumes.csv'
    if volume_csv.exists():
        volume_csv.unlink()
    files = sorted([*Path(a.data_dir).glob("*.las"), *Path(a.data_dir).glob("*.laz")])
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ blocks found in {a.data_dir}")
    dates_by_block = {}
    if a.split_manifest:
        manifest = validate_manifest(a.split_manifest)
        test_rows = [r for r in manifest if r['split'] == 'test']
        if {str(f.resolve()) for f in files} != {r['path'] for r in test_rows}:
            raise ValueError('Evaluation directory must match the test subset in manifest')
        dates_by_block = {Path(r['path']).stem: r['date'] for r in test_rows}
    volume_pairs_by_date = {}
    instance_totals = dict(matches=0.0, predicted=0, truth=0, coverage_sum=0.0, weighted_sum=0.0, truth_points=0)
    rows, semantic_counts, volume_pairs, volume_reference_values = [], np.zeros((2, 2), dtype=np.int64), [], {}
    reference_by_id = {}
    if a.reference_table:
        ref_path = Path(a.reference_table)
        table = pd.read_excel(ref_path) if ref_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(ref_path)
        factor = {"m": 1.0, "cm": 0.01, "mm": 0.001}[a.dimension_unit]
        for _, row in table.iterrows():
            l, w, h = (float(row[c]) * factor for c in (a.length_column, a.width_column, a.height_column))
            if a.block_column and a.block_column in table.columns:
                key = f"{row[a.block_column]}:{int(row[a.id_column])}"
            else:
                key = str(int(row[a.id_column]))
            if key in reference_by_id:
                raise ValueError(f'Duplicate reference key {key}; use unique block and instance identities')
            reference_by_id[key] = float(np.pi * l * w * h / 6)
    for path in files:
        las, xyz, rgb = read_las(path)
        truth, gt_instances = read_labels(las, a.semantic_dim, a.instance_dim, a.head_value)
        ids, pred = predict_block(model, make_features(xyz, rgb), a.num_points, device)
        xyz, truth, gt_instances = xyz[ids], truth[ids], gt_instances[ids]
        for t, q in zip(truth, pred): semantic_counts[t, q] += 1
        predicted_instances, volume_rows = segment_instances_and_volume(
            xyz, pred, a.max_clusters, a.min_fragment_points)
        matched_instances = match_instances(gt_instances, predicted_instances)
        volume_rows_by_instance = []
        for pred_id, gt_id in matched_instances.items():
            mask = predicted_instances == pred_id
            if mask.sum() >= 4:
                estimated = convex_hull_volume(xyz[mask])
                reference_key = f"{path.stem}:{int(gt_id)}"
                reference_value = reference_by_id.get(reference_key, reference_by_id.get(str(int(gt_id)), ""))
                item = {"block": path.stem, "predicted_instance_id": int(pred_id),
                        "matched_ground_truth_id": int(gt_id), "points": int(mask.sum()),
                        "date": dates_by_block.get(path.stem, "unknown"),
                        "estimated_volume_m3": estimated,
                        "reference_volume_m3": reference_value}
                if reference_value != "":
                    key = reference_key
                    volume_pairs.append({"instance_id": key, "volume_m3": estimated})
                    volume_reference_values[key] = float(reference_value)
                    date = dates_by_block.get(path.stem, 'unknown')
                    volume_pairs_by_date.setdefault(date, []).append({'instance_id': key, 'volume_m3': estimated})
                item["instance_key"] = reference_key
                volume_rows_by_instance.append(item)
        if volume_rows_by_instance:
            volume_csv = output / "ground_truth_instance_volumes.csv"
            write_header = not volume_csv.exists()
            with volume_csv.open("a", newline="", encoding="utf-8") as vf:
                writer = csv.DictWriter(vf, fieldnames=list(volume_rows_by_instance[0]))
                if write_header: writer.writeheader()
                writer.writerows(volume_rows_by_instance)
        im = instance_metrics(gt_instances, predicted_instances)
        ng = len(np.unique(gt_instances[gt_instances >= 0]))
        npred = len(np.unique(predicted_instances[predicted_instances >= 0]))
        ngpoints = int((gt_instances >= 0).sum())
        instance_totals['matches'] += im['mPrec'] * npred
        instance_totals['predicted'] += npred
        instance_totals['truth'] += ng
        instance_totals['coverage_sum'] += im['mCov'] * ng
        instance_totals['weighted_sum'] += im['mWCov'] * ngpoints
        instance_totals['truth_points'] += ngpoints
        rows.append({"block": path.name, **im, "head_iou": float(
            ((pred == 0) & (truth == 0)).sum() / max(((pred == 0) | (truth == 0)).sum(), 1)),
            "predicted_instances": len(volume_rows), "gt_instances": len(np.unique(gt_instances[gt_instances >= 0]))})
    tp = semantic_counts[0, 0]; fp = semantic_counts[:, 0].sum() - tp; fn = semantic_counts[0].sum() - tp
    summary = {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
               "f1": float(2*tp / max(2*tp + fp + fn, 1)), "iou": float(tp / max(tp + fp + fn, 1)),
               "blocks": len(files), "instance_macro": {key: float(np.mean([r[key] for r in rows]))
                                                          for key in ("mPrec", "mRec", "mCov", "mWCov")}}
    summary['instance_global'] = {
        'mPrec': instance_totals['matches'] / max(instance_totals['predicted'], 1),
        'mRec': instance_totals['matches'] / max(instance_totals['truth'], 1),
        'mCov': instance_totals['coverage_sum'] / max(instance_totals['truth'], 1),
        'mWCov': instance_totals['weighted_sum'] / max(instance_totals['truth_points'], 1),
    }
    summary['instance_counts'] = instance_totals
    summary["volume_by_date"] = {date: volume_metrics(items, volume_reference_values)
                                   for date, items in volume_pairs_by_date.items()}
    summary["volume"] = volume_metrics(volume_pairs, volume_reference_values) if volume_reference_values else None
    with (output / "evaluation.json").open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    with (output / "blocks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
