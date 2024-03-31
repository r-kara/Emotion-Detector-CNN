# We used larger kernel sizes (7 x 7) in the first convolutional layer
# and smaller kernel sizes (3 x 3) in the second convolutional layer

import torch.nn as nn
import torch.nn.functional as F

class Variant2CNN(nn.Module):
    def __init__(self):
        super(Variant2CNN, self).__init__()

        # Convolution Layer 1 with larger kernel size
        self.conv1 = nn.Conv2d(1, 20, kernel_size=7)  # Larger kernel size
        self.relu1 = nn.ReLU()

        # Convolution Layer 2 with smaller kernel size
        self.conv2 = nn.Conv2d(20, 25, kernel_size=3)  # Smaller kernel size
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(46 * 46 * 25, 450)  # Adjusted input size
        self.fc2 = nn.Linear(450, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        print(x.shape)

        x = x.view(-1, 46 * 46 * 25)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)
        x = self.fc2(x)

        return x
