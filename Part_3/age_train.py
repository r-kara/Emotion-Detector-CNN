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
    transforms.Resize((100, 100)),  # Resize images to 100x100
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert PIL Image to Tensor
])

# Define batch size and learning rate
batch_size = 64
lr = 0.001

# Define dataset and dataloaders for each age group
train_dataset_middle_aged = ImageFolder(root='../Part_3/Age/Adult/train', transform=transform)
test_dataset_middle_aged = ImageFolder(root='../Part_3/Age/Adult/test', transform=transform)

train_loader_middle_aged = DataLoader(train_dataset_middle_aged, batch_size=batch_size, shuffle=True)
test_loader_middle_aged = DataLoader(test_dataset_middle_aged, batch_size=batch_size, shuffle=False)

train_dataset_senior = ImageFolder(root='../Part_3/Age/Senior/train', transform=transform)
test_dataset_senior = ImageFolder(root='../Part_3/Age/Senior/test', transform=transform)

train_loader_senior = DataLoader(train_dataset_senior, batch_size=batch_size, shuffle=True)
test_loader_senior = DataLoader(test_dataset_senior, batch_size=batch_size, shuffle=False)

train_dataset_young = ImageFolder(root='../Part_3/Age/Young/train', transform=transform)
test_dataset_young = ImageFolder(root='../Part_3/Age/Young/test', transform=transform)

train_loader_young = DataLoader(train_dataset_young, batch_size=batch_size, shuffle=True)
test_loader_young = DataLoader(test_dataset_young, batch_size=batch_size, shuffle=False)

class MainCNN(nn.Module):
    def __init__(self):
        super(MainCNN, self).__init__()

        # Convolution Layer 1
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.relu1 = nn.ReLU()

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 25, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(46 * 46 * 25, 450)
        self.fc2 = nn.Linear(450, 4)  # Output is 4 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = x.view(-1, 46 * 46 * 25)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)
        x = self.fc2(x)

        return x

def train_model(model, train_loader, num_epochs=15, lr=0.001, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_test_loss = float('inf')
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

        # Print training loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Early stopping
        if train_loss < best_test_loss:
            best_test_loss = train_loss
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'Models/age_best_model_maincnn.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered!")
                break

    print("Main CNN: Training completed.")
    # Save the trained model
    torch.save(model.state_dict(), 'Models/age_final_model_maincnn.pth')

if __name__ == "__main__":
    # Initialize the model
    model = MainCNN()

    # Train the model for each age group
    print("Training for Adult...")
    train_model(model, train_loader_middle_aged)
    print("\nTraining for Senior...")
    train_model(model, train_loader_senior)
    print("\nTraining for Young...")
    train_model(model, train_loader_young)
