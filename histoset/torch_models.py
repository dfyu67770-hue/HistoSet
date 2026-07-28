"""PyTorch model definitions for HistoSet training.

The module is intentionally separate from the lightweight NumPy utilities so
that the package can still be imported on systems without PyTorch installed.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch
    raise ModuleNotFoundError(
        "PyTorch is required for histoset.torch_models. Install the deep-learning "
        "dependencies with `pip install -e .[dl]` before training models."
    ) from exc


class TinyHistoSetUNet(nn.Module):
    """Small dual-head U-Net used for reproducible HistoSet baselines.

    The explanation head predicts the morphologic concept hierarchy. The
    optional pattern head predicts Gleason patterns directly, allowing
    hierarchy-consistency training between concept-derived and direct pattern
    probabilities.
    """

    def __init__(
        self,
        n_explanation_classes: int = 10,
        n_pattern_classes: int = 4,
        base_channels: int = 16,
        dual_pattern_head: bool = True,
    ) -> None:
        super().__init__()
        self.dual_pattern_head = dual_pattern_head
        self.encoder1 = self._block(3, base_channels)
        self.encoder2 = nn.Sequential(nn.MaxPool2d(2), self._block(base_channels, base_channels * 2))
        self.encoder3 = nn.Sequential(nn.MaxPool2d(2), self._block(base_channels * 2, base_channels * 4))
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = self._block(base_channels * 2, base_channels)
        self.explanation_head = nn.Conv2d(base_channels, n_explanation_classes, kernel_size=1)
        self.pattern_head = (
            nn.Conv2d(base_channels, n_pattern_classes, kernel_size=1) if dual_pattern_head else None
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        enc1 = self.encoder1(image)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        dec2 = self.decoder2(torch.cat([self.up2(enc3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))
        output = {"explanation_logits": self.explanation_head(dec1)}
        if self.pattern_head is not None:
            output["pattern_logits"] = self.pattern_head(dec1)
        return output
