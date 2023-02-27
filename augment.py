import torch
from torchvision import transforms
#from image_dataset import ImageDataset
from pathlib import Path
import numpy as np

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

#xtrain_path = Path("../data/X_train.npy")
ytrain_path = Path("../data/Y_train.npy")

xtrain_path = r"/Users/zygimantaskrasauskas/Desktop/DS1/data/X_train.npy"

X_train_torch = convert_npy_to_torch(xtrain_path)

torch.manual_seed(689)

# Transformation
transform = torch.nn.Sequential(
        transforms.RandomRotation(
            degrees=15
            ),

        transforms.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0)
            ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
            ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
            ),

        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.2)
            ),

        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.2
            ),

        #transforms.RandomApply(
         #   [transforms.Grayscale(num_output_channels=1)],
          #  p=0.1
           # ),

        #transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
            ),

        #transforms.resize()  # Dimensions 1x128x128
)


c = X_train_torch[0].to(dtype = torch.float64)

d = transform(c)


# =============================================================================
# class NumpyDataset(torch.utils.data.Dataset):
#     def __init__(self, dPath, transform=None):
#         self.data = np.load(dPath)
#         self.transform = transform
# 
#     def __len__(self):
#         return len(self.data)
# 
#     def __getitem__(self, index):
#         x = self.data[index]
#         if self.transform:
#             self.transform(x)
#         return x
# 
# 
# scripted_transformed = torch.jit.script(transform)
# 
# =============================================================================
