import torch
import torch.nn as nn

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, out_channels=3, broadcast_resolution=(16, 16), final_resolution=(128, 128)):
        """
        Args:
            slot_dim (int): Dimensionality of the slot vector.
            out_channels (int): Number of output channels (e.g., 3 for RGB).
            broadcast_resolution (tuple): The spatial grid size (H, W) to which each slot is broadcast.
            final_resolution (tuple): The final output resolution after upsampling.
        """
        super().__init__()
        self.broadcast_resolution = broadcast_resolution  # e.g. (16, 16)
        self.final_resolution = final_resolution          # e.g. (128, 128)

        # Decoder using transposed convolutions to upsample from broadcast resolution to final resolution.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(slot_dim + 2, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),             # 32 -> 64
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),             # 64 -> 128
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),                                 # Final conv: output channels 4
            nn.Sigmoid()
        )

    def forward(self, slots):
        """
        Args:
            slots: Tensor of shape [B, K, slot_dim] where K is the number of slots.
        Returns:
            rgb: Reconstructed image patches, shape [B, K, out_channels, H, W]
            alpha: Alpha masks, shape [B, K, 1, H, W]
        """
        B, K, D = slots.shape
        H, W = self.broadcast_resolution

        # Reshape to process all slots in parallel.
        slots = slots.view(B * K, D)  # [B*K, D]

        # Broadcast the slot vector to a small spatial grid.
        slot_broadcast = slots.unsqueeze(-1).unsqueeze(-1).expand(B * K, D, H, W)

        # Create a coordinate grid with values in [0, 1] to match paper's position embeddings.
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=slots.device),
            torch.linspace(0, 1, W, device=slots.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B * K, -1, -1, -1)  # [B*K, 2, H, W]

        # Concatenate broadcasted slot with coordinate grid.
        x = torch.cat([slot_broadcast, grid], dim=1)  # [B*K, slot_dim + 2, H, W]
        out = self.decoder(x)  # Expected output shape: [B*K, 4, final_H, final_W]

        rgb = out[:, :-1, :, :]  # [B*K, 3, final_H, final_W]
        alpha = out[:, -1:, :, :]  # [B*K, 1, final_H, final_W]

        # Reshape back to [B, K, ...]
        final_H, final_W = self.final_resolution
        rgb = rgb.view(B, K, rgb.shape[1], final_H, final_W)
        alpha = alpha.view(B, K, 1, final_H, final_W)
        return rgb, alpha
