import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from Part_2.cnn_test import MainCNN  # Import your CNN model from cnn_test.py

# Define the transformation for image preprocessing
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Define the path to the Age dataset
data_path = '../Part_3/Age'

# Define the classes (emotions)
class_names = ['engaged', 'happy', 'neutral', 'surprised']


# Define a function to evaluate the model on a given data loader
def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    confusion_matrix = np.zeros((len(class_names), len(class_names)))

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            for i in range(len(labels)):
                confusion_matrix[labels[i], predicted[i]] += 1

    accuracy = total_correct / total_samples
    precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


if __name__ == '__main__':
    # Load the model
    model = MainCNN()
    model.load_state_dict(torch.load('Models/age_final_model_maincnn.pth')) ##../Part_2/Models/final_model_maincnn.pth or Models/age_final_model_maincnn.pth

    # Define dictionaries to store evaluation results for each age group
    age_groups = ['Adult', 'Senior', 'Young']
    evaluation_results = {age_group: {} for age_group in age_groups}

    # Lists to store metrics for each group
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    # Evaluate the model on each age group
    for age_group in age_groups:
        # Create dataset and data loaders for the current age group
        test_dataset = ImageFolder(root=f'{data_path}/{age_group}/test', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluate the model
        accuracy, precision, recall, f1_score = evaluate_model(model, test_loader)

        # Store evaluation results
        evaluation_results[age_group]['Accuracy'] = accuracy
        evaluation_results[age_group]['Precision'] = precision
        evaluation_results[age_group]['Recall'] = recall
        evaluation_results[age_group]['F1_Score'] = f1_score

        # Append metrics to lists for averaging
        accuracy_list.append(accuracy)
        precision_list.append(precision.mean())
        recall_list.append(recall.mean())
        f1_score_list.append(f1_score.mean())

    # Calculate averages
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1_score = np.mean(f1_score_list)

    # Print results
    print("Macro-averaged Performance Metrics for Each Demographic Group:")
    for age_group, results in evaluation_results.items():
        print(f"\nAge Group: {age_group}")
        print(f"Accuracy: {results['Accuracy']:.4f}")
        print(f"Precision: {results['Precision'].mean():.4f}")
        print(f"Recall: {results['Recall'].mean():.4f}")
        print(f"F1 Score: {results['F1_Score'].mean():.4f}")

    # Print averages
    print("\nAverage Performance Metrics:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f}")
