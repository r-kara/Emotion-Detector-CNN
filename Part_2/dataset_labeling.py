# Import libraries
import os

# Relative path
train_folder = 'NewDataset/training'
test_folder = 'NewDataset/testing'
val_folder = 'NewDataset/validation'

# Rename files with sequential numbering
def rename_files(folder_path, class_name):
    counter = 1
    for filename in sorted(os.listdir(folder_path)):
        base, ext = os.path.splitext(filename)
        new_name = f'{counter}{ext}'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
        counter += 1

    # Rename the files with the desired prefix
    for filename in sorted(os.listdir(folder_path)):
        base, ext = os.path.splitext(filename)
        new_name = f'{class_name}_{base}{ext}'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

# Iterate over each class of emotion
for class_folder in ('engaged', 'happy', 'neutral', 'surprised'):
    # train dataset
    train_class_folder = os.path.join(train_folder, class_folder)
    rename_files(train_class_folder, 'train_' + class_folder)

    # test dataset
    test_class_folder = os.path.join(test_folder, class_folder)
    rename_files(test_class_folder, 'test_' + class_folder)
    
    # validation dataset
    val_class_folder = os.path.join(val_folder, class_folder)
    rename_files(val_class_folder, 'val_' + class_folder)