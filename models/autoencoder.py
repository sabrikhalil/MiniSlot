import torch
import torch.nn as nn
from models.encoder import Encoder
from models.slot_attention import SlotAttention
from models.decoder import SpatialBroadcastDecoder

class MiniSlotAutoencoder(nn.Module):
    def __init__(self, num_slots=10, slot_dim=64):
        """
        Args:
            num_slots (int): Number of slots.
            slot_dim (int): Dimensionality of each slot vector.
        """
        super().__init__()
        self.encoder = Encoder(in_channels=3, feature_dim=slot_dim)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim, iters=3, hidden_dim=128)
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            out_channels=3,
            broadcast_resolution=(16, 16),
            final_resolution=(128, 128)
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape [B, 3, 128, 128].
        Returns:
            recon: Reconstructed image [B, 3, 128, 128].
            slots: Final slot representations [B, num_slots, slot_dim].
        """
        features = self.encoder(x)           # [B, N, slot_dim]
        slots = self.slot_attention(features)  # [B, num_slots, slot_dim]

        # Vectorized decoding: decode all slots in parallel.
        rgb_slots, alpha_slots = self.decoder(slots)  # rgb_slots: [B, num_slots, 3, 128, 128], alpha_slots: [B, num_slots, 1, 128, 128]

        # Normalize alpha masks over slots.
        alpha_norm = torch.softmax(alpha_slots, dim=1)
        # Recombine slots using normalized alpha masks.
        recon = (rgb_slots * alpha_norm).sum(dim=1)   # [B, 3, 128, 128]
        return recon, slots
