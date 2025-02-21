# test_decoder.py
import torch
from models.decoder import SpatialBroadcastDecoder

def main():
    # Set parameters
    batch_size = 1
    slot_dim = 64
    resolution = (16, 16)
    
    # Create a dummy slot tensor with shape [B, slot_dim]
    dummy_slot = torch.randn(batch_size, slot_dim)
    
    # Initialize the decoder
    decoder = SpatialBroadcastDecoder(slot_dim=slot_dim, out_channels=3, resolution=resolution)
    
    # Run a forward pass
    rgb, alpha = decoder(dummy_slot)
    
    # Print output shapes for verification
    print("Dummy slot shape:", dummy_slot.shape)   # Expected: [1, 64]
    print("RGB output shape:", rgb.shape)            # Expected: [1, 3, 16, 16]
    print("Alpha output shape:", alpha.shape)        # Expected: [1, 1, 16, 16]

if __name__ == "__main__":
    main()
