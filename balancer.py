from image_dataset import ImageDataset
import numpy as np
import os
import random



def balance(directory) -> None:
    """
    This creates a balanced dataset without augmentations 
    i.e. reduces the number of samples of all categories to that of the smallest category
    """
    random.seed(689)
   
    train_dataset = ImageDataset(
            f"{directory}X_train.npy",
            f"{directory}Y_train.npy"
            )
    train_data = train_dataset.imgs
    train_labels = train_dataset.targets

    labels, frequency = np.unique(
            train_labels,
            return_counts=True
            )
    minCount = frequency.argmin()

    bottomBalanced = np.empty((len(labels)*minCount, 1, 128, 128), dtype=int)
    Y_train_bottom_balanced = np.empty(len(labels)*minCount, dtype=int)


    num = 0
    for i, freq in enumerate(frequency):
        tempOriginal = train_data[train_labels == labels[i]]
        for count in range(minCount):
            bottomBalanced[num] = tempOriginal[random.randint(0, freq-1)]
            Y_train_bottom_balanced[num] = labels[i]
            num += 1

    np.save( "data/balanced/X_train.npy", bottomBalanced) #change paths to save augmented datasets for different pipes
    np.save("data/balanced/Y_train.npy", Y_train_bottom_balanced)
