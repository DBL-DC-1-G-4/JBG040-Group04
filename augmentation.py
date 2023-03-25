from image_dataset import ImageDataset
from torchvision import transforms
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset
train_dataset = ImageDataset(Path("/Users/20211922/Documents/DBL1/data/X_train.npy"),
                             Path("/Users/20211922/Documents/DBL1/data/Y_train.npy"))

train_data = train_dataset.imgs
train_labels = train_dataset.targets

torch.manual_seed(689)

train_data_torch = torch.from_numpy(train_data).to(dtype=torch.float32)

augmentation_rules = transforms.Compose([
    transforms.RandomRotation(degrees=(0,25)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10),

        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(1, 3))], p=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.1),
        transforms.RandomAutocontrast()]
)

train_augmented = augmentation_rules(train_data_torch)
train_augmented = train_augmented.numpy()

np.save("/Users/20211922/Documents/DBL1/data/X_train_augmented.npy", train_augmented)
np.save("/Users/20211922/Documents/DBL1/data/Y_train_augmented.npy", train_labels)

train_dataset_augmented = ImageDataset(Path("/Users/20211922/Documents/DBL1/data/X_train_augmented.npy"), 
                                       Path("/Users/20211922/Documents/DBL1/data/Y_train_augmented.npy"))

merged_train = ConcatDataset([train_dataset, train_dataset_augmented])
img, label = merged_train[0]

# print the shape of the image and the label
print(img.shape)
print(label)