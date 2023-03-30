
import torchvision.transforms as transforms
from random import randint
import numpy as np
import torch
from typing import Tuple
from pathlib import Path
import os

class ImageDataset:
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    """

    def __init__(self, x: Path, y: Path) -> None:
        # Target labels
        self.targets = ImageDataset.load_numpy_arr_from_npy(y)
        # Images
        self.imgs = ImageDataset.load_numpy_arr_from_npy(x)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.

        Input:
        path: local path of file

        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)

def augment(pVersion:int)->None:
    cwd = os.getcwd()
    parDir = os.path.dirname(cwd)
    data = os.path.join(parDir, "data")
    torch.manual_seed(689)


    train_dataset = ImageDataset(
            os.path.join(data, "X_train.npy"),
            os.path.join(data, "Y_train.npy")
            )
    train_data = train_dataset.imgs
    train_labels = train_dataset.targets

    # instructions
    # choose pipe you want to use and pass it below to scriptPipe
    # change name of the file you save the augmented data to
    # start with just rotation
    # then check rotation with crop - it should work better if yes - add crop whenever using rotation
    # check other combinations
    # in EDA augmentation you can run and see how augmented images look like or go to google docs report stuff 'data augmentation research' file and see there

    pipe_rotate = torch.nn.Sequential( #rotation
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

    pipeDict = {1: pipe_rotate, 2: pipe_bcs, 3: pipe_sharp, 4: pipe_rotate_bcs, 5: pipe_rotate_sharp, 6: pipe_rotate_sharp_bcs}

    scriptPipe = torch.jit.script(pipeDict[pVersion]) # choose pipe here

    uniqueLabels, frequency = np.unique(
            train_labels,
            return_counts=True
            )
    maxCount = frequency.argmax()

    base = np.ones(len(frequency))*frequency[maxCount]

    # howMany = np.empty(len(frequency))
    # for i in range(len(frequency)):
    #     howMany[i] = base[i] - frequency[i] 

    howMany = base - frequency
    howMany = howMany.astype(int)

    toBeAuged = np.empty((howMany.sum(), 1, 128, 128))

    Y_train = np.empty(howMany.sum())
    Y_train = Y_train.astype(int)

    num = 0
    for i in range(len(uniqueLabels)):
        tempOriginal = train_data[train_labels == uniqueLabels[i]]
        for count in range(howMany[i]):
            toBeAuged[num] = tempOriginal[int(randint(0, frequency[i]-1))]
            Y_train[num] = uniqueLabels[i]
            num += 1


    balanced = np.concatenate((train_data, toBeAuged), axis = 0)
    Y_balanced = np.concatenate((train_labels, Y_train),axis = 0)

    print((Y_balanced.shape, balanced.shape))

    train_torch = torch.from_numpy(balanced).to(dtype=torch.float32)
    train_augmented = scriptPipe(train_torch).numpy()

    np.save(os.path.join(parDir, "X_train_balanced.npy"), train_augmented) #change paths to save augmented datasets for different pipes
    np.save(os.path.join(parDir, "Y_train_balanced.npy"), Y_balanced)
 