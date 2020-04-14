import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# set the path to the downloaded data:
data_path = 'RedLights2011_Medium'

# set a path for saving predictions:
preds_path = 'data/hw01_preds'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed



# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

with open(os.path.join(preds_path, 'preds.json')) as f:
    preds = json.load(f)

def box_plot(I, bounding_boxes, filename, vis_path, flag):
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(I)
    for box in bounding_boxes:
        # Create a Rectangle patch
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title(flag + ' example ' + filename)
    plt.savefig(os.path.join(vis_path, flag + filename))

index_easy = [9, 17]
index_hard = [5, 12, 28]

for i in index_easy:
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))
    bounding_boxes = preds[file_names[i]]
    # convert to numpy array:
    I = np.asarray(I)
    flag = 'easy'
    box_plot(I, bounding_boxes, file_names[i], preds_path, flag)

for i in index_hard:
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))
    bounding_boxes = preds[file_names[i]]
    # convert to numpy array:
    I = np.asarray(I)
    flag = 'hard'
    box_plot(I, bounding_boxes, file_names[i], preds_path, flag)