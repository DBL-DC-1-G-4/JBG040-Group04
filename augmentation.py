import os
os.chdir("/Users/zygimantaskrasauskas/Desktop/DS1/JBG040-Group04")

from image_dataset import ImageDataset
from torchvision import transforms
from collections import Counter
from pathlib import Path
import numpy as np
import torch

os.chdir("/Users/zygimantaskrasauskas/Desktop/DS1")

train_dataset = ImageDataset(Path("data/X_train.npy"),
                             Path("data/Y_train.npy"))

train_data = train_dataset.imgs
train_labels = train_dataset.targets

unique_data_labels = np.unique(train_labels)

square_dict = {label: train_data[train_labels == label] for label in unique_data_labels}

# Find the biggest amount of disease and its number
max_count = Counter(train_labels.tolist()).most_common(1)[0][1]
max_count_label = Counter(train_labels.tolist()).most_common(1)[0][0]

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


train_augmented = augmentation_rules(train_data_torch).numpy()

train_augmented = np.append(train_data, train_augmented, 0)
train_augmented_labels = np.append(train_labels, train_labels, 0)

np.save("data/X_train_augmented.npy", train_augmented)
np.save("data/Y_train_augmented.npy", train_augmented_labels)

