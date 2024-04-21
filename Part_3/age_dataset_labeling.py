import os

'''
This script labels the Age dataset according to age groups, train/test splits, and emotional expressions.
'''

# Define folder path
age_folder = 'Age'


# Function to rename files
def rename_files(folder_path, age_group, train_test, emotion):
    counter = 1
    for filename in sorted(os.listdir(folder_path)):
        base, ext = os.path.splitext(filename)
        new_name = f'{age_group}_{train_test}_{emotion}_{counter}{ext}'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
        counter += 1


# Iterate through age groups
for age_group in ('Adult', 'Senior', 'Young'):
    age_group_folder = os.path.join(age_folder, age_group)

    # Check train and test folders inside each age group
    for train_test in ('train', 'test'):
        train_test_folder = os.path.join(age_group_folder, train_test)

        # Iterate through emotional expressions
        for emotion in ('engaged', 'happy', 'neutral', 'surprised'):
            emotion_folder = os.path.join(train_test_folder, emotion)
            rename_files(emotion_folder, age_group, train_test, emotion)
