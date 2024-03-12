# Emotion-Detector-CNN

<table>
  <tr>
    <td>Name</td>
    <td>Student ID </td>
    <td>GitHub</td>
  </tr>  
  <tr>
    <td>Racha Kara</td>
    <td>40210865</td>
    <td>[r-kara] (https://github.com/r-kara)</td>
  </tr>
  <tr>
    <td>Fadoua Doghmane</td>
    <td>40198495</td>
    <td>[dogmen6] (https://github.com/dogmen6)</td>
  </tr>
  <tr>
    <td>Mariam Kachouh</td>
    <td>40203526</td>
    <td>[emkay21] (https://github.com/emkay21)</td>
  </tr>
</table>

## Contents

1. Python Code:

- Data_Cleaning.py: Contains methods for cleaning the dataset by applying grayscale, cropping, light adjustments, rotation, and resizing.
- Main.py: Main script to execute the code and methods.
- Data_Labeling.py: Contains a Python code to label our dataset images
- data_visualization.py: Script for visualizing the dataset.

2. Dataset Documentation:

- Archive: Folder containing samples of 25 images for each class (happy, focused, surprised, neutral).
- Dataset References: Document providing provenance information for all images and datasets.
- Link to Full Dataset: Contains two links, one directing to the dataset before cleaning and another after cleaning was performed.

3. Readme File:

- Enumerates the contents and describes the purpose of each file in the submission. It explains how to run the code.

4. Report:

- Title page.
- Dataset section explaining the origin of the images and why they were chosen.
- Data cleaning section explaining the process of cleaning the images.
- Labeling section explaining how labelling was performed using a Python file to label images.
- Dataset Visualization section explaining how dataset visualization was done using Python and Matplotlib.
- Reference section containing pertinent resources consulted for this project.

5. Originality Form:

- Contains the expectation of originality form for each team member.

## Purpose

This project aims to develop a convolutional neural network (CNN) model for detecting emotions from facial expressions. It involves cleaning the dataset, visualizing the data, training the model, and evaluating its performance.

## Execution Steps

### Data Cleaning
1. Download the dataset on your computer.
2. Open the Main.py file.
3. Update the paths to the paths chosen for each emotion class in the Main.py file.
4. Call the clean_folder method and put your path as a parameter.
5. Run the script to apply preprocessing steps to the dataset images.

### Data Labeling
1. Download the cleaned dataset on your computer.
2. Open the Data_Labeling.py file.
3. Update the path of your data set folders (test and train folders) at the top of the code.
4. Make sure the label of classes remains in this format: happy, surprised, engaged and neutral.
5. Run the script to relabel the files.

### Data Visualization
1. Download the dataset on your computer.
2. Open the Visualization.py file.
3. Update the path of your data set folder at the top of the code (it is indicated).
4. Make sure the label of classes remains in this format: happy, surprised, engaged and neutral.
5. Run the script to generate visualizations of the dataset.
   
### Notes
1. Ensure that you have Python installed on your system.
2. Install the required Python libraries listed in the code files.
3. Download the dataset and place it in the appropriate directory before running the scripts.
