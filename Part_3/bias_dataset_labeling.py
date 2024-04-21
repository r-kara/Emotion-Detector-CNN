import os

'''
This script takes a dataset that has been split according to a bias (ex: Gender) and labels the data according to the bias, train/test and emotion.
'''

# Define folder paths
#male_train_folder = '/Users/kara/Desktop/Gender/Male/train'
#male_test_folder = '/Users/kara/Desktop/Gender/Male/test'
#female_train_folder = '/Users/kara/Desktop/Gender/Female/train'
#female_test_folder = '/Users/kara/Desktop/Gender/Female/test'
current_dir = os.path.dirname(os.path.abspath(__file__))
male_train_folder = os.path.join(current_dir, 'Gender/Male/train')
male_test_folder = os.path.join(current_dir, 'Gender/Male/test')
female_train_folder = os.path.join(current_dir, 'Gender/Female/train')
female_test_folder = os.path.join(current_dir, 'Gender/Female/test')

# Function to rename files
def rename_files(folder_path, class_name, gender):
    counter = 1
    for filename in sorted(os.listdir(folder_path)):
        base, ext = os.path.splitext(filename)
        new_name = f'{gender}_{class_name}_{counter}{ext}'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
        counter += 1

# Rename files for Male dataset
for class_folder in ('engaged', 'happy', 'neutral', 'surprised'):
    # Train dataset
    train_class_folder = os.path.join(male_train_folder, class_folder)
    rename_files(train_class_folder, class_folder, 'male_train')

    # Test dataset
    test_class_folder = os.path.join(male_test_folder, class_folder)
    rename_files(test_class_folder, class_folder, 'male_test')

# Rename files for Female dataset
for class_folder in ('engaged', 'happy', 'neutral', 'surprised'):
    # Train dataset
    train_class_folder = os.path.join(female_train_folder, class_folder)
    rename_files(train_class_folder, class_folder, 'female_train')

    # Test dataset
    test_class_folder = os.path.join(female_test_folder, class_folder)
    rename_files(test_class_folder, class_folder, 'female_test')