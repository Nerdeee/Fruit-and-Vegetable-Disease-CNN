import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
import os
import random

root_dataset_dir = "./dataset"

# remove underscores from original file names (do this only once)
'''
for folder in os.listdir(root_dataset_dir):
    split_name = folder.split("__", 1)
    os.rename(os.path.join(root_dataset_dir, folder), os.path.join(root_dataset_dir, split_name[0] + " " + split_name[1])) 
'''

# balances the dataset
def balanceDatasets(root_dataset_dir):
    for folder in os.listdir(root_dataset_dir):
        class_dir = os.path.join(root_dataset_dir, folder)
        if os.path.isdir(class_dir):
            num_imgs = os.listdir(class_dir)
            num_img_to_delete = len(num_imgs) - 200
            
            if num_img_to_delete > 0:
                images_to_delete = random.sample(num_imgs, num_img_to_delete)
                
                for img_file in images_to_delete:
                    img_path = os.path.join(class_dir, img_file)
                    os.remove(img_path)
                    print(f"Deleted {img_path}")
                    
                print(f"Finished balancing {class_dir}")
            else:
                print(f"{class_dir} already balanced")
        else:
            print(f"{class_dir} is not a directory")
            

balanceDatasets(root_dataset_dir)



# each class will be a tuple of the image dimensions, number of color channels, label
classes = []    
# training data list
training_data = []

for folder in os.listdir(root_dataset_dir):
    classes.append(folder)

NEW_SIZE = 200
skippedImages = 0

for category in classes:
    folder = os.path.join(root_dataset_dir, category)
    class_num = classes.index(category)
    for img in os.listdir(folder):
        try:
            img_array = cv2.imread(os.path.join(folder, img))
            resized_array = cv2.resize(img_array, (NEW_SIZE, NEW_SIZE))
            new_array = cv2.cvtColor(resized_array, cv2.COLOR_BGR2RGB)
            training_data.append([new_array, class_num])
        except Exception as e:
            skippedImages += 1
            pass
print(f"Successfully added images to training data. {skippedImages} images skipped.")

random.shuffle(training_data)
print(training_data[1])

print('Training data length: ', len(training_data))