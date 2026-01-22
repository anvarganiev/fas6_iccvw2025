import timm
import torch.nn as nn
import torch.nn.functional as F


class FeatHead(nn.Module):
    """
    Wrap any timm backbone so `forward(..., return_feats=True)` returns:
        logits  – shape [B, 1]
        feats   – L2-normalised projection, shape [B, proj_dim]
    """

    def __init__(self, backbone_name: str, pretrained: bool = True, proj_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        in_f = self.backbone.num_features
        self.cls = nn.Linear(in_f, 1)
        self.proj = nn.Sequential(
            nn.Linear(in_f, in_f // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_f // 2, proj_dim),
        )

    def forward(self, x, return_feats: bool = False):
        f = self.backbone(x)
        logits = self.cls(f)
        if return_feats:
            feats = F.normalize(self.proj(f), dim=1)
            return logits, feats
        return logits
