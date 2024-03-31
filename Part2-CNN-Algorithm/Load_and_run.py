import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from Part_2.MainCNN import MainCNN

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to Tensor
])

# Define dataset and dataloaders
test_dataset = ImageFolder(root='../Part_2/NewDataset/testing', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize the model
model = MainCNN()

# Load the saved model from the .pth file
model.load_state_dict(torch.load('best_model.pth'))  # Or load 'final_model.pth' for the final model

# Set model to evaluation mode
model.eval()

# Make predictions on the test dataset
predictions = []
true_labels = []
for images, labels in test_loader:
    with torch.no_grad():  # Disable gradient calculation during inference
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

# Calculate accuracy or other evaluation metrics
accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# INDIVIDUAL IMAGE PREDICTION

# Define a mapping from class index to emotion label
class_to_emotion = {
    0: 'Happy',
    1: 'Neutral',
    2: 'Surprised',
    3: 'Engaged'
}

# Define transforms
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize the image to match the input size of the model
    transforms.ToTensor(),           # Convert PIL Image to Tensor
])

# Provide the path to the individual image
individual_image_path = '../Part_2/Image/example_1.jpg'

# Load individual image using PIL
individual_image = Image.open(individual_image_path)

# Apply transformations to the individual image
individual_image_tensor = transform(individual_image).unsqueeze(0)  # Add batch dimension

# Make predictions on the individual image
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    output = model(individual_image_tensor)
    _, predicted_class = torch.max(output, 1)

# Map the predicted class index to the corresponding emotion label
predicted_emotion = class_to_emotion[predicted_class.item()]

# Print the predicted emotion
print("Predicted Emotion:", predicted_emotion)