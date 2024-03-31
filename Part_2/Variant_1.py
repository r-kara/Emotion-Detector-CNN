# An additional convolutional layer (conv2) was added to the architecture

import torch.nn as nn
import torch.nn.functional as F

class Variant1CNN(nn.Module):
    def __init__(self):
        super(Variant1CNN, self).__init__()

        # Convolution Layer 1
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.relu1 = nn.ReLU()

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 25, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        # Additional Convolution Layer
        self.conv3 = nn.Conv2d(25, 30, kernel_size=3)  # New convolution layer
        self.relu3 = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(44 * 44 * 30, 450)  # Adjusted input size
        self.fc2 = nn.Linear(450, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = self.conv3(x)  # New convolution layer
        x = self.relu3(x)

        x = x.view(-1, 44 * 44 * 30)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)
        x = self.fc2(x)

        return x
