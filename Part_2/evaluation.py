import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from cnn_test import MainCNN, test_loader
from Variant_1 import Variant1CNN
from Variant_2 import Variant2CNN


# Computes and outputs a confusion matrix for a given model
def generate_confusion_matrix(model, test_loader, num_classes, class_names):
    # Set model to evaluation mode
    model.eval()

    total_correct = 0
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                confusion_matrix[labels[i].item(), predicted[i]] += 1

    # Add color, labels and ranges to the matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, cmap="Blues", interpolation='nearest')
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Confusion Matrix")
    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_classes), class_names)

    fmt = ".1f"
    thresh = confusion_matrix.max() / 2

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.colorbar()
    plt.show()

    return confusion_matrix

# Calculates the performance metrics for a given confusion matrix
def calculate_performance_metrics(confusion_matrix):
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    f1_measure = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
    return {
        'Accuracy': float(accuracy),
        'Precision': precision,
        'Mean Recall': mean_recall,
        'F1_measure': f1_measure
    }

# Computes and prints metrics needed to fill the table for the report
def compute_and_print_table_metrics(metrics):
    macro_averaged_precision = []
    macro_averaged_recall = []
    macro_averaged_f1_score = []
    micro_averaged_precision = []
    micro_averaged_recall = []
    micro_averaged_f1_score = []

    # Compute the macro-averaged metrics
    macro_precision = sum(metrics['Precision']) / len(metrics['Precision'])
    macro_recall = metrics['Mean Recall']
    macro_f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    # Compute the micro-averaged metrics
    true_positives = sum(metrics['Precision'])
    false_positives = sum(metrics['Precision']) - sum(precision ** 2 for precision in metrics['Precision'])
    false_negatives = len(metrics['Precision']) - true_positives - false_positives
    micro_precision = true_positives / (true_positives + false_positives)
    micro_recall = true_positives / (true_positives + false_negatives)
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    # Append the computed metrics to the corresponding lists
    macro_averaged_precision.append(macro_precision)
    macro_averaged_recall.append(macro_recall)
    macro_averaged_f1_score.append(macro_f1_score)
    micro_averaged_precision.append(micro_precision)
    micro_averaged_recall.append(micro_recall)
    micro_averaged_f1_score.append(micro_f1_score)

    # Print the metrics in a simple list format
    print("\nTable Metrics Data:\n")  
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Macro-averaged Precision: {macro_precision:.4f}")
    print(f"  Macro-averaged Recall: {macro_recall:.4f}")
    print(f"  Macro-averaged F1-score: {macro_f1_score:.4f}")
    print(f"  Micro-averaged Precision: {micro_precision:.4f}")
    print(f"  Micro-averaged Recall: {micro_recall:.4f}")
    print(f"  Micro-averaged F1-score: {micro_f1_score:.4f}\n")

# Loads the 3 models
def load_models():
    model_dict = {
        'main_model': (MainCNN(), 'Models/final_model_maincnn.pth'),
        'variant_1': (Variant1CNN(), 'Models/final_model_variant1.pth'),
        'variant_2': (Variant2CNN(), 'Models/final_model_variant2.pth'),
    }
    models = {}
    for name, (model, model_path) in model_dict.items():
        loaded_state_dict = torch.load(model_path)
        model.load_state_dict(loaded_state_dict)
        model.eval()
        models[name] = model
    return models

models = load_models()
class_names = ['engaged', 'happy', 'neutral', 'surprised']


# Main part: generate the confusion matrix and the metrics for each model
for name, model in models.items():
    # Generate confusion matrix
    print(f"\nConfusion matrix for {name}:\n")
    conf_matrix = generate_confusion_matrix(model, test_loader, num_classes=4, class_names=class_names)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(conf_matrix)

    # Print performance metrics
    print("\nPerformance Metrics:\n")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {' '.join(f'{name}: {value:.4f}' for name, value in zip(class_names, metrics['Precision']))}")
    print(f"  Mean Recall: {metrics['Mean Recall']:.4f}")
    print(f"  F1_measure: {metrics['F1_measure']:.4f}\n")

    # Print table metrics
    compute_and_print_table_metrics(metrics)





'''
OUTPUT OF THIS SCRIPT


Confusion matrix for main_model:

*popup figure*

Performance Metrics:

  Accuracy: 0.6308
  Precision: engaged: 0.8519 happy: 0.4000 neutral: 0.4667 surprised: 0.6667
  Mean Recall: 0.5187
  F1_measure: 0.5548


Table Metrics Data:

  Accuracy: 0.6308
  Macro-averaged Precision: 0.5963
  Macro-averaged Recall: 0.5187
  Macro-averaged F1-score: 0.5548
  Micro-averaged Precision: 0.7402
  Micro-averaged Recall: 0.7542
  Micro-averaged F1-score: 0.7471


Confusion matrix for variant_1:

*popup figure*

Performance Metrics:

  Accuracy: 0.7846
  Precision: engaged: 1.0000 happy: 0.4545 neutral: 0.6667 surprised: 0.8182
  Mean Recall: 0.7294
  F1_measure: 0.7321


Table Metrics Data:

  Accuracy: 0.7846
  Macro-averaged Precision: 0.7348
  Macro-averaged Recall: 0.7294
  Macro-averaged F1-score: 0.7321
  Micro-averaged Precision: 0.8261
  Micro-averaged Recall: 0.8694
  Micro-averaged F1-score: 0.8472


Confusion matrix for variant_2:

*popup figure*

Performance Metrics:

  Accuracy: 0.8000
  Precision: engaged: 1.0000 happy: 0.5833 neutral: 0.8462 surprised: 0.6471
  Mean Recall: 0.7871
  F1_measure: 0.7780


Table Metrics Data:

  Accuracy: 0.8000
  Macro-averaged Precision: 0.7691
  Macro-averaged Recall: 0.7871
  Macro-averaged F1-score: 0.7780
  Micro-averaged Precision: 0.8364
  Micro-averaged Recall: 0.9053
  Micro-averaged F1-score: 0.8695
'''