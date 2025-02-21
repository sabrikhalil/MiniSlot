import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(4, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # FIX 4: Create 4 directional distance maps with values in [0, 1].
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        left = 1 - grid_x
        right = grid_x
        top = 1 - grid_y
        bottom = grid_y
        pos = torch.stack([left, right, top, bottom], dim=-1)  # [H, W, 4]
        
        # Project to feature dimension.
        pos = self.proj(pos)  # [H, W, dim]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, dim, H, W]
        return x + pos

class Encoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super().__init__()
        # Backbone: Gradual downsampling from 128x128 to 16x16.
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, stride=1, padding=2),  # 128x128 -> 128x128
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),           # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),           # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),           # 32x32 -> 16x16
            nn.ReLU()
        )
        self.pos_embed = PositionEmbedding(dim=64)
        # Per-location MLP to act like a 1x1 convolution.
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape [B, 3, 128, 128]
        Returns:
            Flattened features: [B, N, feature_dim] where N = H * W.
        """
        x = self.backbone(x)         # [B, 64, 16, 16]
        x = self.pos_embed(x)        # Add positional embeddings.
        x = x.permute(0, 2, 3, 1)    # [B, H, W, 64]
        # FIX 5: Apply MLP per location with a residual connection.
        mlp_out = self.mlp(x)        # [B, H, W, feature_dim]
        x = x + mlp_out             # Residual addition (assumes feature_dim == 64)
        return x.flatten(1, 2)      # [B, N, feature_dim]
