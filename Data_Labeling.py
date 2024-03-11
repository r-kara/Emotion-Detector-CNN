# This file labels the image files according to their type. 
# For example, images in the test/happy folder are renamed "test_happy_1", "test_happy_2", etc.

# Import libraries
import os

# Folders containing the train and test sets
train_folder = '/Users/kara/Desktop/engaged_dataset/train'
test_folder = '/Users/kara/Desktop/engaged_dataset/test'

# Iterate over the folders containing each class
for class_folder in ('happy', 'surprised', 'engaged', 'neutral'):
    # train dataset
    train_class_folder = os.path.join(train_folder, class_folder)
    for i, filename in enumerate(os.listdir(train_class_folder)):
        base, ext = os.path.splitext(filename)
        new_name = f'train_{class_folder}_{i}{ext}'
        os.rename(os.path.join(train_class_folder, filename), os.path.join(train_class_folder, new_name))

    # test dataset
    test_class_folder = os.path.join(test_folder, class_folder)
    for i, filename in enumerate(os.listdir(test_class_folder)):
        base, ext = os.path.splitext(filename)
        new_name = f'test_{class_folder}_{i}{ext}'
        os.rename(os.path.join(test_class_folder, filename), os.path.join(test_class_folder, new_name))