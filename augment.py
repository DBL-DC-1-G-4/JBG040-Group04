import torch
from torchvision import transforms
#from image_dataset import ImageDataset
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

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

xtrain_path = r"/home/maxwell/Documents/Y2/DC1/data/X_train.npy"

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

# =============================================================================
#         transforms.ColorJitter(
#             brightness=0.2,
#             contrast=0.2,
#             saturation=0.2,
#             hue=0.2
#             ),
# =============================================================================

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

        transforms.RandomApply(
            [transforms.Grayscale(num_output_channels=1)],
            p=0.1
            ),


        transforms.Normalize(
            mean=0.5,
            std=0.5
            ),

)

# scripted_transformed = torch.jit.script(transform)

npy_file = np.load(xtrain_path)

#%%

floatTorch = X_train_torch.to(dtype=torch.float64)

test = transform(floatTorch[0])
#outTens = torch.tensor(data=[], dtype=torch.float64)
# i = 0
#for tens in tqdm(X_train_torch.to(dtype=torch.float64)):
#    outTens = torch.cat((outTens, transform(tens), 0))
#    i += 1

#np.save(
#    "/home/maxwell/Documents/Y2/DC1/data/aug_X_train.npy",
#    outTens.numpy()
#)

before = plt.imshow(npy_file[0].squeeze(), cmap='gray')

#%%
after = plt.imshow(test.squeeze(), cmap='gray')

# =============================================================================
# class NumpyDataset(torch.utils.data.Dataset):
#     def __init__(self, dPath, transform=None):
#         self.data = np.load(dPath)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)
# 
#     def __getitem__(self, index):
#         x = self.data[index]
#         if self.transform:
#             self.transform(x)
#         return x
# 
# =============================================================================
