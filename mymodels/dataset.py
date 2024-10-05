import torch
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
from PIL import Image 
import os 

padding_transform = transforms.Compose([
    # how many pixels apply to each side
    transforms.Pad((8,8)),
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Pytorch supports two types of datasets: map-style datasets and iterable-style datasets
# Map-style dataset implements __getitem__() and __len__()
# iterable-style represents an iterable over data samples. Suitable for cases where random reads are expensive or even improbable and the batch size depends on fetched area.
# Ex: iter(dataset) can fetch streaming data from database, remote server or real time data.
class SegmentationDataset(Dataset):

    def __init__(self, images_dir, masks_dir, transform=padding_transform):

        self.images_dir=images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):

        return len(self.images)
    
    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        #Convert the image to greyscale mode (L = luminance)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    

