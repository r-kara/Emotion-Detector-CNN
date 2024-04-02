# This code is to determine the paths for our datasets and clean each folder by using the clean_folder() method

from Data_Cleaning import clean_folder

# Paths for our datasets: training set and testing set for all emotions: happy, neutral, surprise and engaged
train_happy_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/train/happy"
train_neutral_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/train/neutral"
train_surprised_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/train/surprised"
train_engaged_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/train/engaged"
test_happy_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/test/happy"
test_neutral_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/test/neutral"
test_surprised_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/test/surprised"
test_engaged_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/test/engaged"

# Clean all images for all emotions: happy, neutral, surprise and engaged
clean_folder(train_happy_path)
clean_folder(train_neutral_path)
clean_folder(train_surprised_path)
clean_folder(train_engaged_path)
clean_folder(test_happy_path)
clean_folder(test_neutral_path)
clean_folder(test_surprised_path)
clean_folder(test_engaged_path)

