import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class conv_layers(nn.Module):
    def __init__(self):
        super(conv_layers, self).__init__()
        # Input Block
        # CONVOLUTION BLOCK 1 -- dilated convolution
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1) 
        ) # output_size = 30, RF=3
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 15, RF = 9

        # CONVOLUTION BLOCK 2 -- Depthwise seperable convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), groups=8, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03) 
        ) # output_size = 13, RF=11

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 6, RF = 22
        # TRANSITION BLOCK 1
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6, RF=22
        
         # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        ) # output_size = 6 RF=24
        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 3, RF = 48

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01)
        ) # output_size = 1

        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        ) 
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.depthwise(x)
        x = self.pool2(x)
        x = self.transition(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.convblock4(x)
        x = self.avg(x)
        x = self.last(x)
        x = x.view(-1, 10)
        return x

def conv_types():
    return (conv_layers())