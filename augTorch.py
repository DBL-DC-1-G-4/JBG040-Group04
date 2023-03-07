from torchvision import transforms
import torch
import numpy
from monaugment import getDirs
from augment import convert_npy_to_torch
torch.manual_seed
parDir, x_train_path, y_train_path, x_test_path, y_test_path = getDirs()

x_dst = convert_npy_to_torch(x_train_path)
X_train_torch = x_dst.to(dtype=torch.float64)


# Affine, Shear, zoom, brightness
transforms.Compose([
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


