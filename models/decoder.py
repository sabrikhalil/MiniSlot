import torch
import torch.nn as nn

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, out_channels=3, resolution=(128, 128)):
        """
        Args:
            slot_dim (int): Dimensionality of the slot vector.
            out_channels (int): Number of output channels (typically 3 for RGB).
            resolution (tuple): Desired spatial resolution (H, W) for decoding.
        """
        super().__init__()
        self.resolution = resolution
        # The decoder takes the broadcasted slot (slot_dim) plus 2 coordinate channels.
        self.decoder = nn.Sequential(
            nn.Conv2d(slot_dim + 2, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Add upsampling
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1),  # 3 RGB + 1 alpha
            nn.Sigmoid()  # Ensure this is present!
        )


    def forward(self, slot):
        """
        Args:
            slot: Tensor of shape [B, slot_dim]
        Returns:
            rgb: Reconstructed image patch, shape [B, out_channels, H, W]
            alpha: Alpha mask, shape [B, 1, H, W]
        """
        B, D = slot.shape
        H, W = self.resolution
        
        # Broadcast the slot vector to a 2D grid: [B, D, H, W]
        slot_broadcast = slot.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)
        
        # Create coordinate grid (2 channels with values in [-1, 1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=slot.device),
            torch.linspace(-1, 1, W, device=slot.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # shape: [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # shape: [B, 2, H, W]
        
        # Concatenate broadcasted slot with coordinate grid: [B, D+2, H, W]
        x = torch.cat([slot_broadcast, grid], dim=1)
        
        # Decode to produce output tensor: [B, out_channels+1, H, W]
        out = self.decoder(x)
        rgb = out[:, :-1, :, :]   # [B, out_channels, H, W]
        alpha = out[:, -1:, :, :]  # [B, 1, H, W]
        return rgb, alpha

# Quick test for decoder:
if __name__ == "__main__":
    dummy_slot = torch.randn(1, 64)
    decoder = SpatialBroadcastDecoder(slot_dim=64, out_channels=3, resolution=(64, 64))
    rgb, alpha = decoder(dummy_slot)
    print("RGB output shape:", rgb.shape)   # Expected: [1, 3, 64, 64]
    print("Alpha mask shape:", alpha.shape)   # Expected: [1, 1, 64, 64]
