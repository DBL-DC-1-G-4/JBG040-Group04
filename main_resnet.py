# Custom imports
from batch_sampler import BatchSampler
from image_dataset import ImageDataset
from train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List

from net_resnet2 import * 

#def ResNet50():
#    return ResNet4(Bottleneck, [3,4,6,3], 6)

def main() -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
    test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    # Define the device to be used for training
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    # Define the ResNeXt50 model
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=6)
    # This line is equivalent to the previous
    
    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()
    for i, data in enumerate(training_loader):
        X, y = data
        outputs = model(X)
#    for e in range(2):
#        x, y = train_dataset
#
#        output = model.forward(x)

main()
