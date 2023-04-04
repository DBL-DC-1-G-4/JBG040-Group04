import torchvision.transforms as transforms
import random
import numpy as np
import torch
import os
from image_dataset import ImageDataset


def augment(pVersion: int) -> None:
    """
    Applies image augmentation techniques to the training data based on
    the value of `pVersion`, and saves the augmented data and
    corresponding labels to a directory.

    Args:
        pVersion (int): The version of the augmentation pipeline to be used.
        Must be one of the following:
            1: rotation,
            2: brightness, contrast and saturation,
            3: sharpness,
            4: rotation, brightness, contrast and saturation,
            5: rotation and sharpness,
            6: rotation, sharpness, brightness, contrast and saturation,

    Returns:
        None
    """

    cwd = os.getcwd()
    dataDir = os.path.join(cwd, "data")
    torch.manual_seed(689)
    random.seed(689)
    train_dataset = ImageDataset(
            os.path.join(dataDir, "X_train.npy"),
            os.path.join(dataDir, "Y_train.npy"),
    )

    train_data = train_dataset.imgs
    train_labels = train_dataset.targets

    pipe_rotate = torch.nn.Sequential(
        transforms.RandomRotation(5)
    )

    pipe_bcs = torch.nn.Sequential(
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
    )

    pipe_sharp = torch.nn.Sequential(
        transforms.RandomAdjustSharpness(
                sharpness_factor=1.3,
                p=0.2
                )
    )
    pipe_rotate_bcs = torch.nn.Sequential(
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
    )

    pipe_rotate_sharp = torch.nn.Sequential(
        transforms.RandomRotation(5),
        transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.2)
    )

    pipe_rotate_sharp_bcs = torch.nn.Sequential(
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.2),
    )

    pipeDict = {
            1: pipe_rotate,
            2: pipe_bcs,
            3: pipe_sharp,
            4: pipe_rotate_bcs,
            5: pipe_rotate_sharp,
            6: pipe_rotate_sharp_bcs
            }

    scriptPipe = torch.jit.script(pipeDict[pVersion])

    uniqueLabels, frequency = np.unique(
            train_labels,
            return_counts=True
            )
    maxIndex = frequency.argmax()
    #  print(f"Maximum count of a class: {frequency[maxIndex]}")
    base = np.ones(len(frequency))*frequency[maxIndex]
    howMany = base - frequency
    howMany = howMany.astype(int)
    toBeAuged = np.empty((howMany.sum(), 1, 128, 128), dtype=int)
    Y_train_tba = np.empty(howMany.sum(), dtype=int)

    num = 0
    for i in range(len(uniqueLabels)):
        tempOriginal = train_data[train_labels == uniqueLabels[i]]
        for count in range(howMany[i]):
            toBeAuged[num] = tempOriginal[random.randint(0, frequency[i]-1)]
            Y_train_tba[num] = uniqueLabels[i]
            num += 1

    torchBeAuged = torch.from_numpy(toBeAuged).to(dtype=torch.float32)
    train_augmented = scriptPipe(torchBeAuged).numpy()
    balancedAndAuged = np.concatenate((train_data, train_augmented), axis=0)
    Y_balanced_augmented = np.concatenate((train_labels, Y_train_tba), axis=0)
    #  print((balancedAndAuged.shape, Y_balanced_augmented.shape))
    #  Change paths to save augmented datasets for different pipes
    augmentedDir = os.path.join(dataDir, "augmented")
    np.save(os.path.join(augmentedDir, "X_train.npy"), balancedAndAuged)
    np.save(os.path.join(augmentedDir, "Y_train.npy"), Y_balanced_augmented)


