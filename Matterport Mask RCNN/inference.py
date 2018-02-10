"""

This script will calculate the bounding boxes for the images at the give location using the Mask RCNN implementation and
store the resulting bounding boxes at the desired location. Be aware that the script does not calculate memory consumption
and thus the hardware may not be capable of handling all images at once.

"""

import matplotlib
matplotlib.use('Agg')

import os
import shutil
import sys
import random
import math
import json
import datetime
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import mrcnn.coco as coco
import mrcnn.utils as utils
import mrcnn.model as modellib
import visualize
import conversion
import misc

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = misc.COCO_CLASSES


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def inference(modelWeightsPath, imagesPath, limit, outputPath, storeImages=False):
    """
    This method loads the .h5 weights and performs inference on all the images at the specified path.
    modelWeightsPath: the path to the .h5 weights
    imagesPath: the path to the images to perform inference on
    outputPath: the path where the results are to be output to, the file will be named instances.json
    storeImages: if this is set to true, the images will be stored at the provided output path with
                 the detected bouding boxes drawn inside them
    """
    numberOfImages = len(os.listdir(imagesPath))

    if numberOfImages == 0:
        return []

    if limit == 0:
        limit = numberOfImages

    today = datetime.datetime.now()
    todayString = "{}.{:02}.{:02}".format(today.year , today.month, today.day)
    outputPath = os.path.join(outputPath, todayString)

    imagesOutputPath = os.path.join(outputPath, "images")
    if not os.path.exists(imagesOutputPath):
        os.makedirs(imagesOutputPath)

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=outputPath, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(modelWeightsPath, by_name=True)

    # Load a random image from the images folder
    images = []
    # Store the file names of the individual images to be able to retrieve them later if they are supposed
    # to be saved with the bounding boxes drawn into them
    imageFileNames = os.listdir(imagesPath)
    imageCount = 0
    for image in [os.path.join(imagesPath, s) for s in imageFileNames]:
        images.append(skimage.io.imread(image))
        imageCount += 1
        if imageCount >= limit:
            break

    accumulatedResults = []
    for i in range(0, min(len(images), limit)):
        result = model.detect([images[i]], verbose=1)
        result = result[0]
        # Add the filename of the image to be able to convert it to COCO format
        result["image_id"] = i
        accumulatedResults.append(result)
        # Visualize results
        image = images[i]
        if storeImages:
            visualize.display_instances(image, result['rois'], None, result['class_ids'], 
                                class_names, result['scores'])
            plt.savefig(os.path.join(imagesOutputPath, imageFileNames[i] + "_out.png"))

    annotationsOutputPath = os.path.join(outputPath, "annotations")

    if not os.path.exists(annotationsOutputPath):
        os.makedirs(annotationsOutputPath)

    with open(os.path.join(annotationsOutputPath, "instances.json"), "w") as jsonFile:
        json.dump(conversion.mrcnn_instance_detections_to_coco_format(accumulatedResults), jsonFile)

    return accumulatedResults

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Perform inference on a set of images with pre-defined weights.")
    parser.add_argument("--modelWeightsPath",
                        "-w", 
                        required=True, 
                        metavar="/path/to/weights",
                        help="The path to the pre-defined weights.")
    parser.add_argument("--imagesPath",
                        "-i",
                        required=True,
                        metavar="/path/to/images",
                        help="The path to the images.")
    parser.add_argument("--outputPath",
                        "-o",
                        required=True,
                        metavar="/path/to/output",
                        help="The path where the detections will be stored as a json.")
    parser.add_argument("--storeImages",
                        "-s",
                        required=False,
                        help="If set to true, the images will be saved to the output path" +
                        " with the bouding boxes drawn into them.")
    parser.add_argument("--limit",
                        "-l",
                        required=False,
                        help="If set, only the first [limit] images will be used for inference.")

    args = parser.parse_args()
    if args.limit:
        args.limit = int(args.limit)
        if args.limit < 0:
            args.limit = 0
    else:
        args.limit = 0

    inference(args.modelWeightsPath, args.imagesPath, args.limit, args.outputPath, args.storeImages)