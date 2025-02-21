import torch
from torchvision import transforms
from PIL import Image
from models.encoder import Encoder

def main():
    # Load the test image (ensure you have 'data/test.png' available)
    img = Image.open("data/test.png").convert("RGB")
    
    # Define the transform: resize to 64x64 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension: [1, 3, 64, 64]
    
    # Initialize the encoder
    encoder = Encoder(in_channels=3, feature_dim=64)
    
    # Forward pass through the encoder
    output = encoder(img_tensor)
    print("Output shape:", output.shape)
    # Expected output shape: [1, 256, 64] (since 64/2/2 = 16 and 16x16=256)

if __name__ == "__main__":
    main()
