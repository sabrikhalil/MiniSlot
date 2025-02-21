import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CLEVRDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the CLEVR dataset 
                           (expects images under root_dir/images/{split}).
            split (str): Dataset split: 'train', 'val', or 'test'.
            transform: torchvision.transforms to preprocess the images.
        """
        self.root_dir = root_dir
        self.split = split
        self.images_dir = os.path.join(root_dir, "images", split)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.images_dir)
                            if os.path.isfile(os.path.join(self.images_dir, f)) and 
                               f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def get_transform(split='train', resolution=(128, 128)):
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor()
        ])
    return transform

def get_dataloader(root_dir, split='train', batch_size=64, shuffle=True, num_workers=4):
    transform = get_transform(split=split, resolution=(128, 128))
    dataset = CLEVRDataset(root_dir=root_dir, split=split, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
