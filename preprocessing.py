import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
import os

root_dataset_dir = "./dataset"

# remove underscores from original file names (do this only once)
'''
for folder in os.listdir(root_dataset_dir):
    split_name = folder.split("__", 1)
    os.rename(os.path.join(root_dataset_dir, folder), os.path.join(root_dataset_dir, split_name[0] + " " + split_name[1])) 
'''

# each class will be a tuple of the image dimensions, number of color channels, label
classes = []    
# training data list
training_data = []

for folder in os.listdir(root_dataset_dir):
    classes.append(folder)

NEW_SIZE = 200

for category in classes:
    folder = os.path.join(root_dataset_dir, category)
    class_num = classes.index(category)
    for img in os.listdir(folder):
        img_array = cv2.imread(os.path.join(folder, img))
        resized_array = cv2.resize(img_array, (NEW_SIZE, NEW_SIZE))
        new_array = cv2.cvtColor(resized_array, cv2.COLOR_BGR2RGB)
        training_data.append([new_array, class_num])

        

