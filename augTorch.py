from torchvision import transforms
import torch
import numpy as np
import os

def getDirs():
    parDir = os.path.dirname(os.getcwd())
    data_path = os.path.join(parDir, "data")
    x_train_path = os.path.join(data_path, "X_train.npy")
    y_train_path = os.path.join(data_path, "Y_train.npy")
    return parDir, x_train_path, y_train_path, x_test_path, y_test_path

def convert_npy_to_torch(npy_file_path):
    """
    Converts numpy array into a pytorch object.

    Input:
    path: path for the numpy file.

    Outputs:
    tensor object: creates a pytorch object.
    """

    npy_file = np.load(npy_file_path)

    torch_object = torch.from_numpy(npy_file)

    return torch_object

torch.manual_seed(689)
parDir, x_train_path, y_train_path, x_test_path, y_test_path = getDirs()

x_dst = convert_npy_to_torch(x_train_path)[:10]
X_train_torch = x_dst.to(dtype=torch.float64)


# Affine, Shear, zoom, brightness
x_tran = transforms.Compose([
    transforms.RandomRotation(degrees=(0,25)),
    transforms.RandomResizedCrop(),
    transforms.RandomErasing(),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1),
        scale=(0.9,1.1),
        shear=10
        ),
    transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomAutocontrast([0.2])
    ]
                   )

test = x_tran(X_train_torch)
