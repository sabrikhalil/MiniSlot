import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),  # 64x64 → 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 32x32 → 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 16x16 → 8x8 (add this layer)
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, 5, stride=2, padding=2),  # 8x8 → 4x4 (optional)
            nn.ReLU()
        )
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 3, H, W]
        Returns:
            A tensor of shape [B, N, feature_dim] where N = (H/4)*(W/4)
        """
        features = self.conv(x)  # [B, feature_dim, H/4, W/4]
        B, C, H, W = features.shape
        # Flatten spatial dimensions (each spatial location becomes one feature vector)
        features = features.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        return features

# For a quick unit-test:
if __name__ == "__main__":
    encoder = Encoder()
    dummy_input = torch.randn(1, 3, 64, 64)  # example input: batch of 1, 64x64 image
    output = encoder(dummy_input)
    print("Output shape:", output.shape)  # Expecting shape [1, 256, 64] because 64/4=16 and 16*16=256.
