import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# the path to our data:
# INSERT DIRECTORY TO FILE
path_data = "C:/Users/marou/PycharmProjects/Emotion-Detector-CNN/Data_Visual/datasets/"

# initializing the array for each set
happy_set = []
neutral_set = []
engaged_set = []
surprised_set = []


# function to load data from the folders
def load_data(set_name, folder_name):  # function to load images into a list or array
    folder_path = os.path.join(path_data, folder_name)
    for obj in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, obj)):
            image = Image.open(os.path.join(folder_path, obj)).convert("RGB")
            image = np.array(image)
            set_name.append(image)


# Function to plot the bar graph
def bar_graph(class_labels, class_count):
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_count, color="#FF69B4")
    plt.xlabel("Emotion Class")
    plt.ylabel("No. of images in each class")
    plt.title("Class Distribution")
    plt.show()


# function to choose a sample from every class
def random_images(set_name, no_samples):
    if (len(set_name)) < no_samples:
        raise ValueError("Not enough number of images in the dataset")
    return random.sample(set_name, no_samples)


# function to display the samples
def plot_grid(dataset, data_class):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))  # here, initializing the sub figures and the axis for each
    fig.suptitle(data_class, fontsize=16) # adding a subtitle for each depending on the class
    for i in range(5):
        for j in range(5):  # iterating between two values to get the position of each image
            ax = axes[i, j]  # given i, j we have a position for the ax
            image = dataset[i*5 + j]   # choose an image for the position, this will iterate over all the 25 images
            ax.imshow(image)  # showing the image here
            ax.axis("off")  # removing the axes to show off the images better
    plt.show()  # show off all the images in one plot


# Here, creating a histogram to show the distribution of pixel intensity for each sample image of each class
def plot_histogram(dataset, data_label):
    fig, axes = plt.subplots(len(dataset), 1, figsize=(5, 5))
    fig.suptitle(f"Pixel Intensity for class {data_label}")
    for i in range(len(dataset)):
        axes[i].hist(dataset[i].ravel(), bins=256, color="pink", edgecolor="black")
    plt.show()


# Loading the datasets
load_data(happy_set, "happy")
load_data(neutral_set, "neutral")
load_data(engaged_set, "engaged")
load_data(surprised_set, "surprised")

# then, plot a bar graph to show the number of images in each class
# counting first:
happy_count = len(happy_set)
neutral_count = len(neutral_set)
engaged_count = len(engaged_set)
surprised_count = len(surprised_set)

# placing the class label and its dataset count in arrays, so we can plot them in a bar graph
class_counts = [happy_count, neutral_count, engaged_count, surprised_count]
class_label = ["happy", "neutral", "engaged", "surprised"]

# plotting bar graph:
bar_graph(class_label, class_counts)

# Here, choosing the sample images
happy_im = random_images(happy_set, 25)
neutral_im = random_images(neutral_set, 25)
engaged_im = random_images(engaged_set, 25)
surprised_im = random_images(surprised_set, 25)

# Creating a plot with 5x5 images (25 each)
plot_grid(happy_im, "happy")
plot_grid(neutral_im, "neutral")
plot_grid(engaged_im, "engaged")
plot_grid(surprised_im, "surprised")

# Finally, plotting histograms for every class and its sample images
plot_histogram(happy_im, "happy")
plot_histogram(neutral_im, "neutral")
plot_histogram(engaged_im, "engaged")
plot_histogram(surprised_im, "surprised")


