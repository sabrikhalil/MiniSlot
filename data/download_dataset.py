import os
import requests
import zipfile
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping download.")
        return
    print(f"Downloading from {url} to {dest_path} ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KiB
    with open(dest_path, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))
    print("Download complete.")

def extract_zip(zip_path, extract_to):
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"File at {zip_path} is not a valid zip file.")
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def download_clevr():
    # Updated URL for CLEVR v1.0 dataset
    url = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    zip_path = os.path.join("data", "CLEVR_v1.0.zip")
    data_folder = os.path.join("data", "CLEVR_v1.0")
    
    # Ensure the data folder exists
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # If dataset is already extracted, skip download and extraction.
    if os.path.exists(data_folder):
        print("Dataset already extracted.")
        return data_folder
    
    # Download the dataset zip file
    download_file(url, zip_path)
    
    # Extract the zip file
    extract_zip(zip_path, "data")
    
    return data_folder

def test_dataset(data_folder, split="train"):
    """
    Tests the dataset by loading one image from a given split ('train', 'val', or 'test')
    and printing its tensor shape.
    """
    # The CLEVR dataset structure: data/CLEVR_v1.0/images/{split}/...
    images_split_dir = os.path.join(data_folder, "images", split)
    if not os.path.exists(images_split_dir):
        print(f"Images folder for split '{split}' not found in the dataset.")
        return
    # List only files (ignore directories)
    files = [f for f in os.listdir(images_split_dir) if os.path.isfile(os.path.join(images_split_dir, f))]
    if len(files) == 0:
        print(f"No images found in the '{split}' split.")
        return
    img_path = os.path.join(images_split_dir, files[0])
    print("Loading image:", img_path)
    img = Image.open(img_path).convert("RGB")
    
    # Apply transformations: resize to 64x64 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    print("Image shape (C x H x W):", img_tensor.shape)

if __name__ == "__main__":
    try:
        data_folder = download_clevr()
        # Test the "train" split by default; you can change to "val" or "test"
        test_dataset(data_folder, split="train")
    except Exception as e:
        print("Error:", e)
