# This file is for cleaning our dataset images for the 4 emotions: happy, neutral, focused and engaged.
# The data cleaning steps that we used are:
#   1. Data resizing
#   2. Slight rotations to increase robustness
#   3. Brightness adjustments
#   4. Slight cropping
#   5. Grayscale colour

# Import the necessary libraries
import os # Used for navigating directories/paths
import cv2 # Used for reading, writing and manipulating images
import numpy as numpy # Used for data robustness operations such as rotations, brightness and cropping

# Function to perform data cleaning on a folder of the dataset
def clean_folder(folder_path):
    # List files contained in the folder
    files = os.listdir(folder_path)

    # Iterate through the files
    for file in files:
        # Construct the complete path to the file
        file_path = os.path.join(folder_path, file)

        # Check if the file is an image or not
        if file.endswith((".jpg", ".png", ".jpeg")):
            # Load the image using OpenCV (cv2)
            img = cv2.imread(file_path)

            # Data cleaning operations

            # Apply minor cropping to increase robustness of the model
            crop_percentage = 0.05
            crop_height = int(img.shape[0] * crop_percentage)
            crop_width = int(img.shape[1] * crop_percentage)
            img = img[crop_height: -crop_height, crop_width: -crop_width]

            # Resizing the images to 100 x 100 px with interpolation method to minimize the loss of image quality
            img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)

            # Apply slight rotation
            rotation_angle = numpy.random.randint(-10, 10)  # Random rotation angle between -10 and 10 degrees
            rows, cols, _ = img.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
            img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

            # Apply brightness adjustment
            brightness_factor = numpy.random.uniform(0.5, 1.5)  # Random brightness factor between 0.5 and 1.5
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

            # Convert color images to grayscale
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save the processed image back to the same file path
            cv2.imwrite(file_path, img)

            print(f"Processed image: {file_path}")
        else:
            print(f"Ignored non-image file: {file_path}")

