from image_dataset import ImageDataset
import numpy as np
import os
import random


def balance() -> None:
    """
    This creates a balanced dataset without augmentations 
    i.e. reduces the number of samples of all categories to that of the smallest category
    """
    random.seed(689)
    cwd = os.getcwd()
    parDir = os.path.dirname(cwd)
    dataDir = os.path.join(parDir,"data")

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
    balancedDir = os.path.join(dataDir, "balanced")
    
    np.save(os.path.join(balancedDir, "X_train.npy"), bottomBalanced) #change paths to save augmented datasets for different pipes
    np.save(os.path.join(balancedDir, "Y_train.npy"), Y_train_bottom_balanced)


balance()
