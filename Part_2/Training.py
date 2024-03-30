import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from Part_2.MainCNN import MainCNN

# Set random seed for reproducibility
torch.manual_seed(42)

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to Tensor
])

# Define dataset and dataloaders
train_dataset = ImageFolder(root='../Part_2/NewDataset/training', transform=transform)
val_dataset = ImageFolder(root='../Part_2/NewDataset/validation', transform=transform)
test_dataset = ImageFolder(root='../Part_2/NewDataset/testing', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize the model, loss function, and optimizer
model = MainCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
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
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

print("Training completed.")

# Save the trained model
torch.save(model.state_dict(), 'final_model.pth')

# Load the model from the .pth file
model.load_state_dict(torch.load('best_model.pth'))
model.load_state_dict(torch.load('final_model.pth'))

