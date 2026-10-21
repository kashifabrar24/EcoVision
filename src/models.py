"""Neural network architectures for EcoVision waste image classification."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """Small CNN baseline trained from scratch.

    Architecture:
        - 4 Conv Blocks (Conv-BatchNorm-ReLU-MaxPool)
        - Global Average Pooling (AdaptiveAvgPool2d to 1x1)
        - Dropout (p=0.3)
        - Linear classification head (num_features -> num_classes)
    """

    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 224x224 -> 112x112
            ConvBlock(32, 64),   # 112x112 -> 56x56
            ConvBlock(64, 128),  # 56x56 -> 28x28
            ConvBlock(128, 256), # 28x28 -> 14x14
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mobilenetv2_transfer(num_classes: int = 6, dropout_p: float = 0.3, pretrained: bool = True):
    """Instantiate ImageNet-pretrained MobileNetV2 with custom classification head.

    Replaces default 1000-class classifier with Dropout(p=dropout_p) + Linear(1280 -> num_classes).
    """
    import torchvision.models as tv_models

    weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)
    in_features = model.last_channel  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )
    return model


