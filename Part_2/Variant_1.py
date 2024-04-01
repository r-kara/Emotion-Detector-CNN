# An additional convolutional layer (conv2) was added to the architecture

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert PIL Image to Tensor
])


# some variables:
batch_size = 64
lr = 0.001

# Define dataset and dataloaders
train_dataset = ImageFolder(root='../Part_2/NewDataset/training', transform=transform)
val_dataset = ImageFolder(root='../Part_2/NewDataset/validation', transform=transform)
test_dataset = ImageFolder(root='../Part_2/NewDataset/testing', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


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


# Initialize the model, loss function, and optimizer
model = Variant1CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
num_epochs = 20
best_val_loss = float('inf')
patience = 5
early_stopping_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)

    # Print training and validation loss
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'Models/best_model_variant1.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

print("Training completed.")

# Save the trained model
torch.save(model.state_dict(), 'Models/final_model_variant1.pth')

