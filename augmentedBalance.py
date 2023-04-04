from image_dataset import ImageDataset
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms


def scriptedPipeline(pVersion: int) -> torch.ScriptModule:
    """
    This function allows for the selection of the augmentation pipeline that is used.
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
        torch.ScriptModule
    """
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
    return torch.jit.script(pipeDict[pVersion])


def augmentedBalance(pVersion: int) -> None:
    """
    This function creates a balanced dataset and augments it.

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
    random.seed(689)
    torch.manual_seed(689)


    cwd = os.getcwd()
    dataDir = os.path.join(cwd, "data")

    train_dataset = ImageDataset(
            os.path.join(dataDir, "X_train.npy"),
            os.path.join(dataDir, "Y_train.npy"),
            )
    train_data = train_dataset.imgs
    train_labels = train_dataset.targets

    labels, frequency = np.unique(
            train_labels,
            return_counts=True
            )
    minIndex = frequency.argmin()

    bottomBalanced = np.empty((len(labels)*frequency[minIndex], 1, 128, 128), dtype=int)
    Y_train_bottom_balanced = np.empty(len(labels)*frequency[minIndex], dtype=int)

    num = 0
    for i, freq in enumerate(frequency):
        tempOriginal = train_data[train_labels == labels[i]]
        for count in range(frequency[minIndex]):
            bottomBalanced[num] = tempOriginal[random.randint(0, freq-1)]
            Y_train_bottom_balanced[num] = labels[i]
            num += 1

    torchBottomBalanced = torch.from_numpy(bottomBalanced).to(dtype=torch.float32)
    pipe = scriptedPipeline(pVersion) 
    balancedAndAuged = np.concatenate(
            (bottomBalanced, pipe(torchBottomBalanced)),
            axis=0)

    Y_balanced_augmented = np.concatenate(
            (Y_train_bottom_balanced, Y_train_bottom_balanced),
            axis=0)

    augmentedDir = os.path.join(dataDir, "balanced_and_augmented")
    np.save(os.path.join(augmentedDir, "X_train.npy"), balancedAndAuged)
    np.save(os.path.join(augmentedDir, "Y_train.npy"), Y_balanced_augmented)


