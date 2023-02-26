import torch
from torchvision import transforms
from image_dataset import ImageDataset
from pathlib import Path
import numpy as np

torch.manual_seed(689)


# Transformation 
transform = torch.nn.Sequential(
        transforms.RandomRotation(
            degrees=15
            ),

        transforms.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0)
            ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
            ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
            ),

        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.2)
            ),

        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.2
            ),

        transforms.RandomApply(
            [transforms.Grayscale(num_output_channels=3)],
            p=0.1
            ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            ),

        transforms.resize()  # Dimensions tbd
)


xtrain_path = Path("../data/X_train.npy")
ytrain_path = Path("../data/Y_train.npy")

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, dPath, transform=None):
        self.data = np.load(dPath)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            self.transform(x)
        return x




scripted_transformed = torch.jit.script(transform)
