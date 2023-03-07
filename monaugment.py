import numpy as np
import torch
import os
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadArray,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    RandAffine
)

torch.manual_seed(689)


def getDirs():
    parDir = os.path.dirname(os.getcwd())
    data_path = os.path.join(parDir, "data")
    x_train_path = os.path.join(data_path, "X_train.npy")
    y_train_path = os.path.join(data_path, "Y_train.npy")
    return parDir, x_train_path, y_train_path, x_test_path, y_test_path


parDir, x_train_path, y_train_path, x_test_path, y_test_path = getDirs()


x_train_data = torch.from_numpy(np.load(x_train_path)[:10])

transform = Compose([
    RandRotate(prob=0.3),
    RandZoom(prob=0.3),
    RandAffine(
        prob=0.5,
        shear_range=(np.pi/6, np.pi/6),
        ),
    ])

tranData = transform()






#%%
