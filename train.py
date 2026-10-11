import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import LasSemanticDataset
from .losses import segmentation_loss
from .model import THyGCN, VARIANTS
from .split import validate_manifest


@torch.no_grad()
def evaluate(model, loader, device, head_class=0):
    model.eval()
    cm = torch.zeros(2, 2, dtype=torch.int64)
    losses = []
    for points, labels, _, _, _, _ in loader:
        points, labels = points.to(device), labels.to(device)
        logits, _ = model(points)
        pred = logits.argmax(-1)
        cm += torch.bincount((labels * 2 + pred).flatten(), minlength=4).reshape(2, 2).cpu()
        losses.append(torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels).item())
    tp, fp, fn = cm[head_class, head_class].item(), cm[:, head_class].sum().item() - cm[head_class, head_class].item(), cm[head_class, :].sum().item() - cm[head_class, head_class].item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"loss": float(np.mean(losses)), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "iou": tp / max(tp + fp + fn, 1)}


def main():
    parser = argparse.ArgumentParser(description="Train and validate T-HyGCN")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", default="runs/thy_gcn")
    parser.add_argument("--split-manifest", required=True, help="CSV with path, spatial_id, date, split")
    parser.add_argument("--variant", choices=list(VARIANTS), default="full")
    parser.add_argument("--augment", action="store_true", help="Optional; not described in the paper")
    parser.add_argument("--semantic-dim", default="semantic")
    parser.add_argument("--instance-dim", default="instance_id")
    parser.add_argument("--head-value", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = LasSemanticDataset(args.train_dir, args.num_points, args.semantic_dim,
                                   args.instance_dim, args.head_value, args.augment, args.seed)
    val_set = LasSemanticDataset(args.val_dir, args.num_points, args.semantic_dim,
                                 args.instance_dim, args.head_value, False, args.seed)
    validate_manifest(args.split_manifest, train_set.files, val_set.files)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)
    model = THyGCN(input_channels=6, neighbours=args.neighbors, variant=args.variant).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_set.epoch = epoch
        model.train(); total_loss = 0.0
        for points, labels, instance_ids, _, _, _ in train_loader:
            points, labels = points.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, embedding = model(points)
            loss = segmentation_loss(logits, embedding, labels)
            loss.backward(); optimizer.step(); total_loss += loss.item()
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        state = {"format_version": 2, "variant": args.variant, "model": model.state_dict(), "neighbors": args.neighbors, "iterations": 2,
                 "input_channels": 6, "num_classes": 2, "epoch": epoch,
                 "metrics": metrics, "args": vars(args)}
        torch.save(state, output / "last.pt")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]; torch.save(state, output / "best.pt")
        with (output / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": total_loss / max(len(train_loader), 1),
                                **metrics}, ensure_ascii=False) + "\n")
        print(f"epoch={epoch:03d} train_loss={total_loss / max(len(train_loader), 1):.4f} "
              f"val_f1={metrics['f1']:.4f} val_iou={metrics['iou']:.4f}")


if __name__ == "__main__":
    main()
