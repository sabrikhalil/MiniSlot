# test_autoencoder.py
import torch
from torchvision import transforms
from PIL import Image
from models.autoencoder import MiniSlotAutoencoder

def main():
    # Load the test image (ensure 'data/test.png' exists)
    img = Image.open("data/test.png").convert("RGB")
    
    # Define the transform: resize to 64x64 and convert to tensor.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # [B, 3, 64, 64] with B=1
    
    # Initialize the autoencoder.
    model = MiniSlotAutoencoder(num_slots=5, slot_dim=64)
    
    # Run a forward pass.
    recon, slots = model(img_tensor)
    
    # Print the shapes to verify correctness.
    print("Input image shape:", img_tensor.shape)      # Expected: [1, 3, 64, 64]
    print("Reconstructed image shape:", recon.shape)     # Expected: [1, 3, 16, 16]
    print("Slot representations shape:", slots.shape)    # Expected: [1, 5, 64]

if __name__ == "__main__":
    main()
