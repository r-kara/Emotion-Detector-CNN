import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# from test import kFoldCrossValidation

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
# train_dataset = ImageFolder(root='../Part_2/NewDataset/training', transform=transform)
# val_dataset = ImageFolder(root='../Part_2/NewDataset/validation', transform=transform)
# test_dataset = ImageFolder(root='../Part_2/NewDataset/testing', transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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


def train_model_kfold(model, datas, num_epochs=10, lr=0.001, patience=5, numKfold=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    # Here, evaluation metrics
    accuracy_fold = []
    precision_micro_fold = []
    recall_micro_fold = []
    f1_micro_fold = []

    precision_macro_fold = []
    recall_macro_fold = []
    f1_macro_fold = []

    kf = KFold(n_splits=numKfold, shuffle=True, random_state=0)

    fold_nb = 1
    # Feed the split function the dataset path
    # for each fold, starting with fold nb1:
    for train, test in kf.split(datas):
        print("----------------------------------------------------------------")
        train_data = torch.utils.data.Subset(datas, train)
        test_data = torch.utils.data.Subset(datas, test)
        # Data Loaders:
        train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

        print(f"Fold: {fold_nb}/{numKfold}")
        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for images, labels in train_load:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_load.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            for images, labels in test_load:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
            val_loss /= len(test_load.dataset)

            # Print training and validation loss
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                # Save the best model
                # torch.save(model.state_dict(), 'Models/age_best_model_maincnn.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered!")
                    break

        print(f"Fold {fold_nb}: Training completed.")
        # Save the trained model
        # torch.save(model.state_dict(), 'Models/age_final_model_maincnn.pth')

        # Evaluating the fold
        model.eval()
        predictions = []
        actual = []
        with torch.no_grad():
            for images, labels in test_load:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                actual.extend(labels.cpu().numpy())

        # Next, calculate the evaluation metrics needed for this fold:
        accuracy_fold.append(accuracy_score(actual, predictions))
        # Micro metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(actual, predictions,
                                                                                     average='micro')
        precision_micro_fold.append(precision_micro)
        recall_micro_fold.append(recall_micro)
        f1_micro_fold.append(f1_micro)

        # Macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(actual, predictions,
                                                                                     average='macro')
        precision_macro_fold.append(precision_macro)
        recall_macro_fold.append(recall_macro)
        f1_macro_fold.append(f1_macro)

        # Printing the metrics for each fold one at a time
        print(f" Fold {fold_nb}/{numKfold}:")
        print(f" Accuracy: {accuracy_fold[-1]:.4f}")
        print("Micro values: ")
        print(
            f" Micro Precision: {precision_micro:.4f}, Micro Recall: {recall_micro:.4f}, Micro F1-score: {f1_micro:.4f}")
        print("Macro values: ")
        print(
            f" Macro Precision: {precision_macro:.4f}, Micro Recall: {recall_macro:.4f}, Micro F1-score: {f1_macro:.4f}")
        # Updating the fold number
        fold_nb += 1
    # Calculating average of metrics of all folds
    avg_accuracy = sum(accuracy_fold) / len(accuracy_fold)
    avg_precision_micro = sum(precision_micro_fold) / len(precision_micro_fold)
    avg_recall_micro = sum(recall_micro_fold) / len(recall_micro_fold)

    avg_f1_micro = sum(f1_micro_fold) / len(f1_micro_fold)

    avg_precision_macro = sum(precision_macro_fold) / len(precision_macro_fold)
    avg_recall_macro = sum(recall_macro_fold) / len(recall_macro_fold)
    avg_f1_macro = sum(f1_macro_fold) / len(f1_macro_fold)

    print("Average values:")
    print(f" Average Accuracy: {avg_accuracy:.4f}")
    print("Micro values: ")
    print(f" Average Precision: {avg_precision_micro:.4f}, Average Recall: {avg_recall_micro:.4f}, Average F1: {avg_f1_micro:.4f}")
    print("Macro values: ")
    print(f" Average Precision: {avg_precision_macro:.4f}, Average Recall: {avg_recall_macro:.4f}, Average F1: {avg_f1_macro:.4f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),  # Convert PIL Image to Tensor
    ])

    batch_size = 64
    lr = 0.001
    kfolds = 10

    # Define dataset and dataloaders
    # train_dataset = ImageFolder(root='../Part_2/NewDataset/training', transform=transform)
    # val_dataset = ImageFolder(root='../Part_2/NewDataset/validation', transform=transform)
    # test_dataset = ImageFolder(root='../Part_2/NewDataset/testing', transform=transform)

    # Getting the dataset
    data = ImageFolder(root='../Part_3/Dataset', transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model
    model = MainCNN()

    # Train the model
    train_model_kfold(model, data)
