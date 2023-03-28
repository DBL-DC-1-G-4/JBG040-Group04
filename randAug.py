import os
from image_dataset import ImageDataset
import numpy as np
import torch
import torchvision.transforms as transforms
from random import randint


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

howMany = np.ones(len(frequency))*frequency[maxCount] - frequency

toBeAuged = np.empty(len(uniqueLabels)*frequency[maxCount])
Y_train = np.empty(len(uniqueLabels)*frequency[maxCount])
num = 0
for i in range(len(uniqueLabels)):
    tempOriginal = train_torch[train_labels == uniqueLabels[i]]
    for count in range(int(howMany[i])):
        print(type(num))
        toBeAuged[num] = tempOriginal[int(randint(0, frequency[i]))]
        Y_train[num] = uniqueLabels[i]
        num +=1


print(toBeAuged.shape)





#train_augmented = scriptPipe(train_torch).numpy()

#np.save(os.join(parDir,"X_train_auged.npy"), X_auged)
#np.save(os.join(parDir, "Y_train_auged.npy"), Y_auged)



