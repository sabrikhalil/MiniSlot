# test_slot_attention.py
import torch
from models.slot_attention import SlotAttention

def main():
    # Assume our encoder produces a set of features with shape [B, N, D]
    # For this test, we use:
    # B (batch size) = 1, N (number of features) = 256, D (feature dimension) = 64
    B, N, D = 1, 256, 64
    dummy_inputs = torch.randn(B, N, D)
    
    # Define number of slots (e.g., 5) and iterations (e.g., 3)
    num_slots = 5
    iters = 3

    # Instantiate the Slot Attention module.
    slot_attn = SlotAttention(num_slots=num_slots, dim=D, iters=iters)

    # Forward pass
    slots = slot_attn(dummy_inputs)
    
    print("Input features shape:", dummy_inputs.shape)  # Expected: [1, 256, 64]
    print("Output slots shape:", slots.shape)           # Expected: [1, 5, 64]

if __name__ == "__main__":
    main()
