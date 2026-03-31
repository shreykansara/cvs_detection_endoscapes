"""
CVS Classifier — ResNet-50 backbone
-------------------------------------------
Architecture: torchvision ResNet-50 pretrained on ImageNet,
with the default classifier replaced by a single binary output head.

The two-phase training design (freeze backbone → unfreeze) is specific
to this implementation and is not derived from any existing CVS codebase.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CVSClassifier(nn.Module):
    """
    Binary CVS classifier built on ResNet-50.

    Phase 1 (freeze_backbone=True): only the new head trains.
              Allows the head to learn meaningful features before
              the pretrained weights are disturbed.
    Phase 2 (unfreeze_backbone()): entire network fine-tunes
              at a lower learning rate with cosine annealing.

    num_outputs: 1 for binary/soft, 3 for per-criterion prediction.
    """

    def __init__(self,
                 dropout: float = 0.4,
                 freeze_backbone: bool = True,
                 num_outputs: int = 1):
        super().__init__()

        self.backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the default classifier with a task-specific head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_outputs)
        )

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.backbone(x)
        # Squeeze only for single-output case to keep shape consistent
        return out.squeeze(1) if out.shape[1] == 1 else out