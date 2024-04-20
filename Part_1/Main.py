# This code is to determine the paths for our datasets and clean each folder by using the clean_folder() method

from Data_Cleaning import clean_folder

# Paths for our datasets: training set and testing set for all emotions: happy, neutral, surprise and engaged
train_happy_path = "/Users/doghm/OneDrive/Bureau/Young/happy/train"
train_neutral_path = "/Users/doghm/OneDrive/Bureau/Young/neutral/train"
train_surprised_path = "/Users/doghm/OneDrive/Bureau/Young/surprised/train"
train_engaged_path = "/Users/doghm/OneDrive/Bureau/Young/engaged/train"
test_happy_path = "/Users/doghm/OneDrive/Bureau/Young/happy/test"
test_neutral_path = "/Users/doghm/OneDrive/Bureau/Young/neutral/test"
test_surprised_path = "/Users/doghm/OneDrive/Bureau/Young/surprised/test"
test_engaged_path = "/Users/doghm/OneDrive/Bureau/Young/engaged/test"

# Clean all images for all emotions: happy, neutral, surprise and engaged
clean_folder(train_happy_path)
clean_folder(train_neutral_path)
clean_folder(train_surprised_path)
clean_folder(train_engaged_path)
clean_folder(test_happy_path)
clean_folder(test_neutral_path)
clean_folder(test_surprised_path)
clean_folder(test_engaged_path)

