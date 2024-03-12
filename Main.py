from Data_Cleaning import clean_folder

# Paths for our datasets: training set and testing set for all emotions: happy, neutral, surprised and focused
train_happy_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/train/happy"
train_neutral_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/train/neutral"
train_surprise_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/train/surprised"
test_happy_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/test/happy"
test_neutral_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/test/neutral"
test_surprise_path = "/Users/doghm/OneDrive/Documents/Concordia/Semester 6/COMP 472/Dataset/test/surprised"

# Clean all images for all emotions: happy, neutral, surprised and focused
clean_folder(train_happy_path)
clean_folder(train_neutral_path)
clean_folder(train_surprise_path)
clean_folder(test_happy_path)
clean_folder(test_neutral_path)
clean_folder(test_surprise_path)
