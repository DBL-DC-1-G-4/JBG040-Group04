import os
from image_dataset import ImageDataset
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from math import ceil


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
train_torch = torch.from_numpy(train_data).to(dtype=torch.float32)


pipe = torch.nn.Sequential(
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomRotation(5),
        transforms.RandomAdjustSharpness(
            sharpness_factor=1.3,
            p=0.2
            ),
        )

scriptPipe = torch.jit.script(pipe)

uniqueLabels, frequency = np.unique(
        train_labels,
        return_counts=True
        )
maxCount = frequency.argmax()


for labIt in tqdm(range(len(uniqueLabels))):
    tempTorch = torch.from_numpy(train_data[train_labels == uniqueLabels[labIt]]).to(dtype=torch.float32)
    tempArr = np.empty(ceil(frequency[maxCount] / frequency[labIt]))
    label = np.array([uniqueLabels[labIt] for i in range(frequency[maxCount])])
    if labIt == 0:
        Y_auged = label
    else:
        Y_auged = np.concatenate(Y_auged, label)
    
    if labIt == maxCount:
        if labIt == 0:
            X_auged = train_data[train_labels == uniqueLabels[labIt]]
        else:
            X_auged = np.concatenate(
                    X_auged,
                    train_data[train_labels == uniqueLabels[labIt]]
                    )
    else:
        for i in range(ceil(frequency[maxCount]/frequency[labIt])-1):
            if i == 0:
                tempAuged = train_data[train_labels == uniqueLabels[labIt]]
            else:
                transfedNP = scriptPipe(tempTorch).numpy()
                print(transfedNP.shape)
                print(type(transfedNP))
                tempAuged = np.concatenate(tempAuged, transfedNP)

        if labIt == 0:
            X_auged = tempAuged
        else:
            X_auged = np.concatenate(X_auged, tempAuged[:frequency[maxCount]])


    label = np.array([uniqueLabels[labIt] for i in range(frequency[maxCount])])
    Y_auged = np.concatenate(Y_auged, label)

augedLab, augedFreq = np.unique(X_auged)
print(augedFreq)


train_augmented = scriptPipe(train_torch).numpy()
np.save(os.join(parDir,"X_train_auged.npy"), X_auged)
np.save(os.join(parDir, "Y_train_auged.npy"), Y_auged)
