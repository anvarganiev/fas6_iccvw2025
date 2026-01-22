import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss_LS(nn.Module):
    """Binary cross-entropy with label smoothing."""

    def __init__(self, label_smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing must be in [0, 1)"
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = (
                target * positive_smoothed_labels
                + (1 - target) * negative_smoothed_labels
            )
        return self.bce_with_logits(input, target)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets.float())
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BinaryFocalLoss(nn.Module):
    """Binary focal loss variant."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        pt = torch.sigmoid(logits).detach()
        focal = (1 - pt).pow(self.gamma)
        loss = self.bce(logits, targets.float()) * focal
        alpha = self.alpha if targets.sum() else (1 - self.alpha)
        return (alpha * loss).mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        sim = torch.mm(feats, feats.t()) / self.T
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        sim_exp = torch.exp(sim) * (
            ~torch.eye(len(feats), dtype=bool, device=feats.device)
        )
        pos_exp = sim_exp * mask
        loss = -torch.log(pos_exp.sum(1) / sim_exp.sum(1)).mean()
        return loss


def get_loss(labels_pred, labels_true, criterion, aux_pred=None, aux_true=None, criterion_mh=None):
    """Compute the loss dictionary."""
    res = {"BCE_loss": criterion(labels_pred, labels_true)}
    if criterion_mh and aux_pred is not None:
        res["CE_loss"] = criterion_mh(aux_pred, aux_true) * 0.001
    return res
