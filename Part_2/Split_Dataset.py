import os
import shutil
from sklearn.model_selection import train_test_split

# Define the path to your dataset directory
dataset_path = '/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset'

# Define the path to the new dataset directory with split folders
new_dataset_path = '/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/NewDataset'

# Create directories for training, validation, and testing sets
train_dir = os.path.join(new_dataset_path, "training")
val_dir = os.path.join(new_dataset_path, "validation")
test_dir = os.path.join(new_dataset_path, "testing")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define emotions
emotions = ["engaged", "happy", "neutral", "surprised"]

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Split each emotion folder into training, validation, and testing sets
for emotion in emotions:
    # Create subdirectories for each emotion in training, validation, and testing sets
    os.makedirs(os.path.join(train_dir, emotion), exist_ok=True)
    os.makedirs(os.path.join(val_dir, emotion), exist_ok=True)
    os.makedirs(os.path.join(test_dir, emotion), exist_ok=True)

    # Get all image filenames for the current emotion
    image_filenames = os.listdir(os.path.join(dataset_path, emotion))

    # Split image filenames into training, validation, and testing sets
    train_size = int(train_ratio * len(image_filenames))
    val_size = int(val_ratio * len(image_filenames))
    test_size = len(image_filenames) - train_size - val_size

    train_filenames, remaining_filenames = train_test_split(image_filenames, train_size=train_size, random_state=42)
    val_filenames, test_filenames = train_test_split(remaining_filenames, train_size=val_size, random_state=42)

    # Copy images to corresponding subdirectories
    for filename in train_filenames:
        src = os.path.join(dataset_path, emotion, filename)
        dst = os.path.join(train_dir, emotion, filename)
        shutil.copy(src, dst)

    for filename in val_filenames:
        src = os.path.join(dataset_path, emotion, filename)
        dst = os.path.join(val_dir, emotion, filename)
        shutil.copy(src, dst)

    for filename in test_filenames:
        src = os.path.join(dataset_path, emotion, filename)
        dst = os.path.join(test_dir, emotion, filename)
        shutil.copy(src, dst)
