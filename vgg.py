import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(VGG, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            # Each layer is convolution, batch normalization applied
            # ReLU, max pooling and dropout

            #c0nv 1

            nn.Conv2d(1, 16, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.1),

            # Defining another 2D convolution layer
            #CONV 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.1),

            #NEW GROUP
            #CONV 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.05),
            #NEW GROUP
            #CONV 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.05),
       
            #NEW GROUP
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.05),


 
        )

        self.linear_layers = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128*4*4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
           
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x=self.linear_layers(x)
        
        return x

