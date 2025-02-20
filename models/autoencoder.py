import torch
import torch.nn as nn
from models.encoder import Encoder
from models.slot_attention import SlotAttention
from models.decoder import SpatialBroadcastDecoder

class MiniSlotAutoencoder(nn.Module):
    def __init__(self, num_slots=5, slot_dim=64):
        """
        Args:
            num_slots (int): Number of slots.
            slot_dim (int): Dimensionality of slot vectors.
        """
        super().__init__()
        self.encoder = Encoder(in_channels=3, feature_dim=slot_dim)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim, iters=7)
        # Set decoder resolution to (64,64) to match input.
        self.decoder = SpatialBroadcastDecoder(slot_dim=slot_dim, out_channels=3, resolution=(32, 32))

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, 64, 64].
        Returns:
            recon: Reconstructed image [B, 3, 64, 64].
            slots: Final slot representations [B, num_slots, slot_dim].
        """
        features = self.encoder(x)               # [B, N, slot_dim] where N=256 for a 64x64 image.
        slots = self.slot_attention(features)      # [B, num_slots, slot_dim]
        B, K, _ = slots.shape
        # Decode each slot into an image patch and alpha mask.
        decoded_slots = [self.decoder(slots[:, i, :]) for i in range(K)]
        rgb_slots = torch.stack([ds[0] for ds in decoded_slots], dim=1)    # [B, K, 3, 64, 64]
        alpha_slots = torch.stack([ds[1] for ds in decoded_slots], dim=1)  # [B, K, 1, 64, 64]
        # Normalize alpha masks over slots.
        alpha_norm = torch.softmax(alpha_slots, dim=1)  # [B, K, 1, 64, 64]
        recon = (rgb_slots * alpha_norm).sum(dim=1)       # [B, 3, 64, 64]
        return recon, slots

# Quick test for autoencoder:
if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 64, 64)
    model = MiniSlotAutoencoder(num_slots=5, slot_dim=64)
    recon, slots = model(dummy_input)
    print("Input shape:", dummy_input.shape)         # Expected: [1, 3, 64, 64]
    print("Reconstruction shape:", recon.shape)        # Expected: [1, 3, 64, 64]
    print("Slots shape:", slots.shape)                 # Expected: [1, 5, 64]
