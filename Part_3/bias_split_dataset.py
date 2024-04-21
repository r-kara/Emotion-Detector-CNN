import os
import shutil
from sklearn.model_selection import train_test_split

'''
This script takes a bias subfolder (ex: Male) and splits the dataset into a training set and a testing set, each containing the 4 emotions.
'''

# Define the path to your dataset directory
dataset_path = '/Users/kara/Desktop/Gender_Dataset/Male'

# Define the path to the new dataset directory with split folders
new_dataset_path = '/Users/kara/Desktop/Gender/Male'

# Create directories for training, validation, and testing sets
train_dir = os.path.join(new_dataset_path, "train")
test_dir = os.path.join(new_dataset_path, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define emotions
emotions = ["engaged", "happy", "neutral", "surprised"]

# Define split ratios
train_ratio = 0.7
test_ratio = 0.3

# Split each emotion folder into training, validation, and testing sets
for emotion in emotions:
    # Create subdirectories for each emotion in training, validation, and testing sets
    os.makedirs(os.path.join(train_dir, emotion), exist_ok=True)
    os.makedirs(os.path.join(test_dir, emotion), exist_ok=True)

    # Get all image filenames for the current emotion
    image_filenames = os.listdir(os.path.join(dataset_path, emotion))

    # Split image filenames into training, validation, and testing sets
    train_size = int(train_ratio * len(image_filenames))
    test_size = len(image_filenames) - train_size

    train_filenames, test_filenames = train_test_split(image_filenames, train_size=train_ratio, random_state=42)

    # Copy images to corresponding subdirectories
    for filename in train_filenames:
        src = os.path.join(dataset_path, emotion, filename)
        dst = os.path.join(train_dir, emotion, filename)
        shutil.copy(src, dst)

    for filename in test_filenames:
        src = os.path.join(dataset_path, emotion, filename)
        dst = os.path.join(test_dir, emotion, filename)
        shutil.copy(src, dst)