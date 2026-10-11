import torch
import torch.nn.functional as F


def batch_hard_triplet(embedding, labels, margin=0.2, chunk_size=256):

    b, n, _ = embedding.shape
    total = embedding.sum() * 0.0
    all_ids = torch.arange(n, device=labels.device)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        anchor = embedding[:, start:end]
        with torch.no_grad():
            distances = ((anchor.square().sum(-1, keepdim=True)
                          + embedding.square().sum(-1).unsqueeze(1)
                          - 2 * anchor @ embedding.transpose(1, 2)).clamp_min(0))
            same = labels[:, start:end, None].eq(labels[:, None, :])
            self_mask = all_ids[start:end, None] == all_ids[None, :]
            positive, pos_ids = distances.masked_fill(~(same & ~self_mask), -1).max(-1)
            negative, neg_ids = distances.masked_fill(same, float('inf')).min(-1)
            valid = (positive >= 0) & torch.isfinite(negative)
        batch = torch.arange(b, device=labels.device)[:, None]
        positive = (anchor - embedding[batch, pos_ids]).square().sum(-1)
        negative = (anchor - embedding[batch, neg_ids]).square().sum(-1)
        total = total + F.relu(positive[valid] - negative[valid] + margin).sum()
    return total / (b * n)


def segmentation_loss(logits, embedding, labels):
    ce = F.cross_entropy(logits.transpose(1, 2), labels)
    return 0.8 * ce + 0.2 * batch_hard_triplet(embedding, labels)
