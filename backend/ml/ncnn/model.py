"""N-CNN model definition.

Clean-room reimplementation of the Zamzmi et al. (IJCNN 2019) neonatal pain
CNN, per NCNN_implementation_spec.md. No external N-CNN repository was read
or copied. Topology follows the published design; specific filter counts,
kernel sizes, and dense width are working defaults flagged in comments with
`TODO: confirm against IJCNN 2019 source` so they can be tuned once the
primary reference is verified.

Research prototype, not a medical device.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Defaults that came from the spec's "star" markers. Anything starred in the
# spec is a working default, not a confirmed value, and needs verifying
# against the IJCNN 2019 source before we report any result.
DEFAULT_INPUT_SIZE = 120  # TODO: confirm against IJCNN 2019 source
DEFAULT_NUM_CLASSES = 2
DEFAULT_DROPOUT = 0.5  # TODO: confirm against IJCNN 2019 source
DEFAULT_DENSE_WIDTH = 128  # TODO: confirm against IJCNN 2019 source


class NCNN(nn.Module):
    """N-CNN: three parallel branches, channel concat, merge conv, classifier head.

    The three branches must share the same spatial size at the concat point.
    Each branch ends in a single 2x2 max pool so a 120x120 input becomes
    60x60 before concat. The classifier uses LazyLinear so the input
    resolution can change via config without recomputing the flatten size.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        dense_width: int = DEFAULT_DENSE_WIDTH,
    ) -> None:
        super().__init__()

        # Generic branch (L): shallow path, medium kernel, broad facial structure.
        # Kernel 5x5. TODO: confirm against IJCNN 2019 source.
        self.generic = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Deep branch (C): two stacked 3x3 convs, finer higher-level features.
        # TODO: confirm against IJCNN 2019 source.
        self.deep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Prominent branch (R): large kernel, wide receptive field for the
        # salient pain regions (brow, eyes, nasolabial fold).
        # Kernel 7x7. TODO: confirm against IJCNN 2019 source.
        self.prominent = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Merge head: concat is 32 + 64 + 32 = 128 channels along dim=1.
        # Kernel 3x3 merge conv. TODO: confirm against IJCNN 2019 source.
        self.merge = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Classifier head. LazyLinear infers the flatten size from the first
        # forward pass; replace with nn.Linear and the explicit feature count
        # once the architecture is frozen.
        # Dense width 128. TODO: confirm against IJCNN 2019 source.
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.LazyLinear(dense_width)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(dense_width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run input through all three branches and the head. Returns raw logits."""
        l = self.generic(x)
        c = self.deep(x)
        r = self.prominent(x)
        merged = torch.cat([l, c, r], dim=1)
        merged = self.merge(merged)
        flat = self.flatten(merged)
        flat = self.dropout1(flat)
        h = self.relu_fc(self.fc1(flat))
        h = self.dropout2(h)
        return self.fc2(h)

    def branch_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose per-branch outputs for tests and diagnostics. Not used at inference."""
        return {
            "generic": self.generic(x),
            "deep": self.deep(x),
            "prominent": self.prominent(x),
        }
