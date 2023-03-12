import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            # Each layer is convolution, batch normalization applied
            # ReLU, max pooling and dropout
            #CONV 1 
            nn.Conv2d(1, 8, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,8,kernel_size=3,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            

            # Defining another 2D convolution layer
            #CONV 2
            nn.Conv2d(8, 16, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            

            # Defining another 2D convolution layer
            #CONV 3
            nn.Conv2d(16, 32, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #NEW GROUP
            #CONV 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
       
            #NEW GROUP
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),


 
        )

        self.linear_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
            #nn.Softmax(dim=1)
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x=self.linear_layers(x)
        
        return x
