from torchvision import transforms
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def get_dirs():
    parDir = os.path.dirname(os.getcwd())
    data_path = "data"
    x_train_path = os.path.join(data_path, "X_train.npy")
    y_train_path = os.path.join(data_path, "Y_train.npy")
    x_test_path = os.path.join(data_path, "Y_test.npy")
    y_test_path = os.path.join(data_path, "Y_test.npy")
    return parDir, x_train_path, y_train_path, x_test_path, y_test_path


def augment_data():
    torch.manual_seed(689)

    parDir, x_train_path, y_train_path, x_test_path, y_test_path = get_dirs()

    x_npy = np.load(x_train_path)

    X_train_torch = torch.from_numpy(x_npy).to(dtype=torch.float32)

    x_tran = transforms.Compose([
        transforms.RandomRotation(degrees=(0,25)),
        transforms.RandomErasing(scale=(0.1, 0.1)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(1, 3))], p=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.1),
        transforms.RandomAutocontrast()
        ]
    )

    x_auged = x_tran(X_train_torch).numpy()
    np.save("data/augmented/X_train_split.npy", x_auged)