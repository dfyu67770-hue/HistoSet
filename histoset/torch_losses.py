"""PyTorch losses for HistoSet."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for histoset.torch_losses. Install with `pip install -e .[dl]`."
    ) from exc


EXPLANATION_TO_PATTERN = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=torch.long)


def soft_targets(counts: "torch.Tensor", foreground: "torch.Tensor") -> "torch.Tensor":
    denom = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (counts / denom) * foreground[:, None].float()


def soft_cross_entropy(logits: "torch.Tensor", target: "torch.Tensor", foreground: "torch.Tensor") -> "torch.Tensor":
    loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    return loss[foreground].mean()


def class_balanced_soft_cross_entropy(
    logits: "torch.Tensor", target: "torch.Tensor", foreground: "torch.Tensor", max_weight: float = 8.0
) -> "torch.Tensor":
    mass = (target * foreground[:, None].float()).sum(dim=(0, 2, 3))
    frequency = mass / mass.sum().clamp_min(1e-6)
    weights = (1.0 / torch.sqrt(frequency.clamp_min(1e-4))).clamp(max=max_weight)
    weights = weights / weights.mean().clamp_min(1e-6)
    loss = -(target * weights[None, :, None, None].to(logits.device) * F.log_softmax(logits, dim=1)).sum(dim=1)
    return loss[foreground].mean()


def soft_dice_loss(
    logits: "torch.Tensor", target: "torch.Tensor", foreground: "torch.Tensor", eps: float = 1e-6
) -> "torch.Tensor":
    probability = F.softmax(logits, dim=1) * foreground[:, None].float()
    target = target * foreground[:, None].float()
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    present = target.sum(dim=(0, 2, 3)) > 0
    dice = (2 * intersection + eps) / (denominator + eps)
    if present.any():
        dice = dice[present]
    return 1.0 - dice.mean()


def remap_explanation_to_pattern(explanation_probability: "torch.Tensor") -> "torch.Tensor":
    mapping = EXPLANATION_TO_PATTERN.to(explanation_probability.device)
    n, _, h, w = explanation_probability.shape
    pattern = torch.zeros(n, 4, h, w, dtype=explanation_probability.dtype, device=explanation_probability.device)
    for explanation_index in range(explanation_probability.shape[1]):
        pattern[:, mapping[explanation_index]] += explanation_probability[:, explanation_index]
    return pattern


def hierarchy_consistency_loss(
    explanation_logits: "torch.Tensor", pattern_logits: "torch.Tensor", foreground: "torch.Tensor"
) -> "torch.Tensor":
    explanation_pattern = remap_explanation_to_pattern(F.softmax(explanation_logits, dim=1))
    direct_pattern = F.softmax(pattern_logits, dim=1)
    return ((explanation_pattern - direct_pattern).abs().sum(dim=1)[foreground]).mean() / 2.0


def tumor_margin_loss(
    explanation_logits: "torch.Tensor",
    explanation_target: "torch.Tensor",
    foreground: "torch.Tensor",
    margin: float = 0.20,
) -> "torch.Tensor":
    target_tumor = (explanation_target[:, 1:].sum(dim=1) > 0).float()
    benign_logit = explanation_logits[:, 0]
    max_tumor_logit = explanation_logits[:, 1:].max(dim=1).values
    tumor_loss = F.relu(margin + benign_logit - max_tumor_logit)
    benign_loss = F.relu(margin + max_tumor_logit - benign_logit)
    positive_rate = target_tumor[foreground].mean().clamp(1e-3, 1 - 1e-3)
    weights = torch.where(target_tumor > 0, 0.5 / positive_rate, 0.5 / (1 - positive_rate))
    return (torch.where(target_tumor > 0, tumor_loss, benign_loss) * weights)[foreground].mean()
