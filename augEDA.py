import numpy as np
import matplotlib.pyplot as plt #used for the plot
from pathlib import Path
import torchvision.transforms as transforms
from random import randint
import torch
from typing import Tuple
import os
from image_dataset import ImageDataset

directory = "data/"
cwd = os.getcwd()
parDir = os.path.dirname(cwd)
data = directory
torch.manual_seed(689)


train_dataset = ImageDataset(
        os.path.join(data, "X_train.npy"),
        os.path.join(data, "Y_train.npy")
        )

train_data = train_dataset.imgs
train_labels = train_dataset.targets

pipe = torch.nn.Sequential(
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomRotation(5),
        transforms.RandomAdjustSharpness(
            sharpness_factor=1.3,
            p=0.2
            ),
        )

pipe_rotate = torch.nn.Sequential( #sharpness
    transforms.RandomRotation(5)
)

pipe_bcs = torch.nn.Sequential( #birhtness, contrast, saturation
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
)

pipe_sharp = torch.nn.Sequential( #sharpness
    transforms.RandomAdjustSharpness(
            sharpness_factor=1.3,
            p=0.2
            )
)

pipe_rotate_crop = torch.nn.Sequential( #cuts out the black part of the image after rotating
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    transforms.CenterCrop(size=128),
)

pipe_rot_crop_sharp = torch.nn.Sequential(
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.2),
    transforms.CenterCrop(size=128),
)

pipe_rot_crop_sharp_sat = torch.nn.Sequential(
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.2),
    transforms.CenterCrop(size=128),
)
#first 10 images unaugmented
fig = plt.figure(figsize=(5, 2))
for i in range(10):
    ax = plt.subplot2grid((2, 5), (int(i / 5), i - int(i / 5) * 5))
    ax.imshow(train_data[i][0], cmap='gray')
    ax.set_title(f"Label: {train_labels[i]}")
plt.show()

X_train_aug = np.zeros_like(train_data[:10])
for i in range(10):
    img =train_data[i][0]
    img_tensor = transforms.ToTensor()(img)
    img_tensor = pipe_sharp(img_tensor)
    img_aug = transforms.ToPILImage()(img_tensor)
    X_train_aug[i] = np.array(img_aug)

# Display first 10 augmented images with labels
fig = plt.figure(figsize=(5, 2))
for i in range(10):
    ax = plt.subplot2grid((2, 5), (int(i / 5), i - int(i / 5) * 5))
    ax.imshow(X_train_aug[i][0], cmap='gray')
    ax.set_title(f"Label: {train_labels[i]}")
plt.show()
