import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import visualize

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath(".\\")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib

class Config(Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 3
    DETECTION_MIN_CONFIDENCE = 0.1
    # IMAGE_MIN_DIM = 832
    # IMAGE_MAX_DIM = 832

config = Config()
config.display()
MRCNN_model_path = "mrcnn\\mask_rcnn_custom_0002.h5"
print(MRCNN_model_path)
model = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_model_path, config=Config())
#load model weights
model.load_weights(MRCNN_model_path, by_name=True)



class Config1(Config):
    NAME = "model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 31
    DETECTION_MIN_CONFIDENCE = 0.1
    # IMAGE_MIN_DIM = 832
    # IMAGE_MAX_DIM = 832

config = Config1()
config.display()

print(MRCNN_model_path)
model1 = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_model_path, config=Config1())
#load model weights
model.load_weights(MRCNN_model_path, by_name=True)

import cv2
import os


save_directory = './out'
input_directory = './input'

for filename in os.listdir(input_directory):
    f = os.path.join(input_directory, filename)
    if os.path.isfile(f):
        print(f)
        image = cv2.imread(f)
        results1 = model.detect([image], verbose=1)
             
        
        r1 = results1[0]
        thrld = 0.1
        dr = save_directory + '/' + filename
        al, dd = visualize.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
                                r1['scores'], thrld, dr)
                
