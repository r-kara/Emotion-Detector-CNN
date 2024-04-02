import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from cnn_test import MainCNN, test_loader
from Variant_1 import Variant1CNN
from Variant_2 import Variant2CNN


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

def generate_confusion_matrix(model, test_loader, num_classes, class_names):
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

    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, cmap="Blues", interpolation='nearest')
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
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

for name, model in models.items():
    # Generate confusion matrix
    print(f"\nConfusion matrix for {name}:\n")
    conf_matrix = generate_confusion_matrix(model, test_loader, num_classes=4, class_names=class_names)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(conf_matrix)

    #Print performance metrics
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {' '.join(f'{name}: {value:.4f}' for name, value in zip(class_names, metrics['Precision']))}")
    # Print mean recall as a single value
    print(f"Mean Recall: {metrics['Mean Recall']:.4f}")
    print(f"F1_measure: {metrics['F1_measure']:.4f}\n")




'''

def generate_confusion_matrix(model, test_loader, num_classes, labels=None, class_names=None):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]

    if labels is not None:
        conf_matrix = confusion_matrix(true_labels, predictions, labels=list(range(num_classes)))
    else:
        conf_matrix = confusion_matrix(predictions, predictions, labels=list(range(num_classes)))

    plt.figure(figsize=figsize)
    plt.imshow(conf_matrix, cmap="Blues")
    plt.xticks(np.arange(conf_matrix.shape[1]), class_names)
    plt.yticks(np.arange(conf_matrix.shape[0]), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    fmt = ".2f"
    thresh = conf_matrix.max() / 2

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.colorbar()
    plt.show()

    return conf_matrix


    
def load_model():
    model_dict = {
        'main_model': 'Models/final_model_maincnn.pth',
        'variant_1': 'Models/final_model_variant1.pth',
        'variant_2': 'Models/final_model_variant2.pth'
    }
    models = {name: MainCNN() for name in model_dict}
    for name, model in models.items():
        model.load_state_dict(torch.load(model_dict[name]))
        model.eval()
    return models
'''

'''
Confusion matrix for main_model:

Accuracy: 0.6000
Precision: engaged: 0.8214 happy: 0.6000 neutral: 0.4194 surprised: 0.0000
Mean Recall: 0.4873
F1_measure: 0.4734


Confusion matrix for variant_1:

Accuracy: 0.5692
Precision: engaged: 0.8462 happy: 0.2667 neutral: 0.5000 surprised: 0.4286
Mean Recall: 0.5101
F1_measure: 0.5102


Confusion matrix for variant_2:

Accuracy: 0.6923
Precision: engaged: 0.8966 happy: 0.5714 neutral: 0.4762 surprised: 0.6250
Mean Recall: 0.6012
F1_measure: 0.6211
'''