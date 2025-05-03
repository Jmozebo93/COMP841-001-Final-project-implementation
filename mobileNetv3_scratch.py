import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish Activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

# MBConv Block
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expansion
        self.use_residual = in_channels == out_channels and stride == 1

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            Swish()
        ) if expansion != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            Swish()
        )

        self.se = SEBlock(mid_channels)

        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Expansion
        out = self.expand(x)
        # Depthwise convolution
        out = self.depthwise(out)
        # Squeeze-and-Excitation block
        out = self.se(out)
        # Projection
        out = self.project(out)
        # Residual connection (if applicable)
        if self.use_residual:
            out = out + x
        return out



class MobileNetV3(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Swish(),

            # MBConv blocks
            MBConv(16, 24, expansion=1, stride=1),  # Block 1
            MBConv(24, 40, expansion=6, stride=2),  # Block 2
            MBConv(40, 80, expansion=6, stride=2),  # Block 3
            MBConv(80, 112, expansion=6, stride=1),  # Block 4
            MBConv(112, 160, expansion=6, stride=2),  # Block 5
            MBConv(160, 160, expansion=6, stride=1)
        )

        # Pooling and classification layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(160, 1280),  # Bottleneck layer
            Swish(),
            nn.Linear(1280, num_classes)  # Final classification layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Example usage
if __name__ == "__main__":
    model = MobileNetV3(num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)  # torch.Size([1, 2])