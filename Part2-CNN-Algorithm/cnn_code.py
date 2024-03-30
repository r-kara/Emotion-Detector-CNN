# this python script is used to create and train a CNN model to determine the emotion class of a given image
import os
import torch
import torch.cuda as cuda
import torch.nn as nn
import matplotlib.pyplot as plt  # for loss and performance graphs

# for the training phase:
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms  # to normalize the dataset

# some useful functionalities
import torch.nn.functional as F

# some important parameters
epochs = 10
batch_size = 50
learning_rate = 0.1


# create data loaders by loading the dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")

# no need to transform because it was done in the previous sprint
train_dir = os.path.join(dataset_dir, "train")
train_dataset = datasets.ImageFolder(root=train_dir, transform=None)

test_dir = os.path.join(dataset_dir, "test")
test_dataset = datasets.ImageFolder(root=test_dir, transform=None)

# now we run the dataLoader to specify how many data to load and learn from at the same time/during same batch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# next, we create the CNN model

# class CNN(nn.Module):









