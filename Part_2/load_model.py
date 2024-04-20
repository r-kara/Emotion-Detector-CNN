import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# Import the CNN you want to load: MainCNN or Variant_1 or Variant_2
from cnn_test import MainCNN
# from Variant_1 import Variant1CNN
# from Variant_2 import Variant2CNN

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert PIL Image to Tensor
])

# Define dataset and dataloaders
test_dataset = ImageFolder(root='../Part_2/NewDataset/testing', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize the model
model = MainCNN() # Insert the CNN Model (MainCNN() / Variant1CNN() / Variant2CNN())

# Load the saved model from the .pth file
model.load_state_dict(torch.load('Models/best_model_maincnn.pth'))  # Insert 'age_best_model_maincnn.pth' or 'best_model_variant1.pth' or 'best_model_variant2.pth'

# Set model to evaluation mode
model.eval()

# Make predictions on the test dataset
predictions = []
true_labels = []
correct_prediction = 0
total_img = 0
for images, labels in test_loader:
    with torch.no_grad():  # Disable gradient calculation during inference
        # Get prediction for current batch of test data
        outputs = model(images)
        # Get predicted class labels
        _, predicted = torch.max(outputs, 1)

        # total nb of images processed
        total_img += labels.size(0)

        # update counter by comparing predicted and true label
        correct_prediction += (predicted == labels).sum().item()
        # predictions.extend(predicted.tolist())
        # true_labels.extend(labels.tolist())

# Calculate accuracy or other evaluation metrics
print('Accuracy: %d %%' % (100 * correct_prediction / total_img))
# accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(true_labels)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# INDIVIDUAL IMAGE PREDICTION

# Define a mapping from class index to emotion label
class_to_emotion = {
    0: 'Engaged',
    1: 'Happy',
    2: 'Neutral',
    3: 'Surprised'
}

# Define transforms
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize the image to match the input size of the model
    transforms.ToTensor(),           # Convert PIL Image to Tensor
])

# Provide the path to the individual image
individual_image_path = '../Part_2/Image/train_engaged_101.jpg'

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
