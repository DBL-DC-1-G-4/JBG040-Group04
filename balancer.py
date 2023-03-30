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
    parDir = os.path.pardir(cwd)
    data = os.path.join(parDir,"data")
    train_dataset = ImageDataset(
            os.path.join(data,"X_train.npy"),
            os.path.join(data,"Y_train.npy")
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

    np.save(os.path.join(parDir, "X_train_bottom_balanced.npy"), bottomBalanced) #change paths to save augmented datasets for different pipes
    np.save(os.path.join(parDir, "Y_train_bottom_balanced.npy"), Y_train_bottom_balanced)
