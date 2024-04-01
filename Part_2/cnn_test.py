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


class MainCNN(nn.Module):
    def __init__(self):
        super(MainCNN, self).__init__()

        # Convolution Layer 1, our input images are: 48 x 48 x 1 -> 1 for grayscale
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)  # 1 for grayscale, 20 for number of kernels -> 44 x 44 x 20
        self.relu1 = nn.ReLU()  # activation function, 44 x 44 x 20

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 25, kernel_size=5)  # dimensions 20 x 20 x 25
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.maxpool2 = nn.MaxPool2d(2)  # the dimensions will drop by 2: 10 x 10 x 25
        self.relu2 = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(46 * 46 * 25, 450)
        self.fc2 = nn.Linear(450, 4)  # out is 4 classes

    def forward(self, x):
        # Convolution layer 1:
        x = self.conv1(x)
        x = self.relu1(x)

        # Convolution layer 2:
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # switch from activation map to vectors
        x = x.view(-1, 46 * 46 * 25)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)
        x = self.fc2(x)

        return x


# Initialize the model, loss function, and optimizer
model = MainCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
num_epochs = 15
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
        torch.save(model.state_dict(), 'Models/best_model_maincnn.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

print("Main CNN: Training completed.")

# Save the trained model
torch.save(model.state_dict(), 'Models/final_model_maincnn.pth')

