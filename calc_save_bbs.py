"""

This script will calculate the bounding boxes for the images at the give location using the Mask RCNN implementation and
store the resulting bounding boxes at the desired location. Be aware that the script does not calculate memory consumption
and thus the hardware may not be capable of handling all images at once.

"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

def calc_save_bbs(imagesPath, outputPath, saveImagesPath, customBatchSize):
    numberOfImages = len(os.listdir(imagesPath))

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to trained weights file
    # Download this file and place in the root of your 
    # project (See README file for details)
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, imagesPath)

    batchSize = numberOfImages
    # Only assign the batch size that the user set if it is actually less than the number of images at the specified location
    if customBatchSize and customBatchSize <= numberOfImages:
      batchSize = customBatchSize

    if numberOfImages % batchSize != 0:
        raise Exception("Invalid batch size. Number of images needs to be dividable by batch size.")

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = batchSize

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # Load a random image from the images folder
    images = []
    # Store the file names of the individual images to be able to retrieve them later if they are supposed
    # to be saved with the bounding boxes drawn into them
    imageFileNames = os.listdir(imagesPath)
    for image in [imagesPath + s for s in imageFileNames]:
        images.append(skimage.io.imread(image))

    #Run detection
    currentBatch = 0
    while currentBatch < numberOfImages:
        # Batch size stays the same everytime we call the model, we just need to keep track of what batch we are processing
        results = model.detect(images, verbose=1)
        currentBatch += batchSize

        # Visualize results
        if saveImagesPath:
            for index in range(numberOfImages):
                r = results[index]
                print(r['rois'])
                visualize.display_instances(images[index], r['rois'], r['masks'], False, r['class_ids'], 
                                        class_names, r['scores'])
                plt.savefig(saveImagesPath + "/" + imageFileNames[index] + "_out.png")
