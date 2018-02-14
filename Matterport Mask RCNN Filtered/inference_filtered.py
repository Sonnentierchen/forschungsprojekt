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
import cv2

import coco_filtered as coco
import mrcnn.utils as utils
import mrcnn.model as modellib
import visualize
import conversion
import misc
import extract_frames

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = misc.COCO_CLASSES


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def video_inference(modelWeightsPath, videoPath, limit, outputPath, storeImages=False):
    videoFolder = os.path.dirname(videoPath)
    videoName = os.path.basename(videoPath)
    videoFramesFolder = os.path.join(videoFolder, videoName.lower() + "_extracted_frames")
    if os.path.exists(videoFramesFolder):
        shutil.rmtree(videoFramesFolder)
    os.makedirs(videoFramesFolder)

    extract_frames.extract_frames(videoPath, videoFramesFolder, limit, "none")

    results, inferenceOutputPath = images_inference(modelWeightsPath, videoFramesFolder, limit, outputPath, storeImages)

    #shutil.rmtree(videoFramesFolder)

    # Get the folder of the images with the detections drawn into them and assemble video
    detectedFramesPath = os.path.join(inferenceOutputPath, "images")
    imageFileNames = os.listdir(detectedFramesPath)
    imageFileNames.sort()
    numberOfImages = len(imageFileNames)
    if numberOfImages == 0:
        return

    img = cv2.imread(os.path.join(detectedFramesPath, imageFileNames[0]))
    height , width , layers =  img.shape
    video = cv2.VideoWriter(os.path.join(inferenceOutputPath, 'video.avi'),cv2.VideoWriter_fourcc('M','J','P','G'),24,(width,height))

    for i in range(0, numberOfImages):
        currentImagePath = os.path.join(detectedFramesPath, imageFileNames[i])
        currentImage = cv2.imread(currentImagePath)
        video.write(currentImage)

    video.release()


def images_inference(modelWeightsPath, imagesPath, limit, outputPath, storeImages=False):
    """
    This method loads the .h5 weights and performs inference on all the images at the specified path.
    At the end, the classes irrelevant for VUFO will be filtered out.
    modelWeightsPath: the path to the .h5 weights
    imagesPath: the path to the images to perform inference on
    outputPath: the path where the results are to be output to, the file will be named instances.json
    storeImages: if this is set to true, the images will be stored at the provided output path with
                 the detected bouding boxes drawn inside them
    """
    # Filter for only images
    includedExtensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    imageFileNames = [fn for fn in os.listdir(imagesPath)
              if any(fn.endswith(ext) for ext in includedExtensions)]
    numberOfImages = len(imageFileNames)
    if numberOfImages == 0:
        return []

    if limit == 0:
        limit = numberOfImages

    today = datetime.datetime.now()
    todayString = "{}_{:02}_{:02}_{:02}_{:02}_{:02}".format(today.year , today.month, today.day, today.hour, today.minute, today.second)
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
    imageCount = 0
    for image in [os.path.join(imagesPath, s) for s in imageFileNames]:
        images.append(skimage.io.imread(image))
        imageCount += 1
        if imageCount >= limit:
            break

    # Build our index filter to block all COCO classes that we don't want
    class_filter = []
    for class_name in misc.VUFO_CLASSES:
        class_filter.append(misc.COCO_CLASSES.index(class_name))

    accumulatedResults = []
    for i in range(0, min(len(images), limit)):
        result = model.detect([images[i]], verbose=1)
        result = result[0]
        # Filter out non-VUFO classes
        result = misc.filter_result_for_classes(result, class_filter)

        # Add the filename of the image to be able to convert it to COCO format
        result["image_id"] = i
        accumulatedResults.append(result)
        # Visualize results
        image = images[i]
        if storeImages:
            visualize.display_instances(image, result['rois'], None, result['class_ids'], 
                                class_names, result['scores'])
            plt.savefig(os.path.join(imagesOutputPath, imageFileNames[i] + "_out.png"))
            plt.close()

    annotationsOutputPath = os.path.join(outputPath, "annotations")

    if not os.path.exists(annotationsOutputPath):
        os.makedirs(annotationsOutputPath)

    with open(os.path.join(annotationsOutputPath, "instances.json"), "w") as jsonFile:
        json.dump(conversion.mrcnn_instance_detections_to_coco_format(accumulatedResults), jsonFile)

    with open(os.path.join(outputPath, "log.txt"), "w") as logFile:
        logFile.write("Running [FILTERED] inference with the following parameters:\n\n")
        logFile.write("Weights at: " + modelWeightsPath + "\n")
        logFile.write("Run on {} of {} images at: {}\n".format(limit, numberOfImages, imagesPath))
        logFile.write("Results stored at {}\n".format(outputPath))

    return accumulatedResults, outputPath

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
                        required=False,
                        metavar="/path/to/images",
                        help="The path to the images that inference is to be run on. " +
                        "Either images or video have to be specified.")
    parser.add_argument("--videoPath",
                        "-v",
                        required=False,
                        metavar="/path/to/video/",
                        help="The path to the video that inference is to be run on. " +
                        "Either images or video have to be specified.")
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

    if not args.imagesPath and not args.videoPath:
        raise ValueError("Either the images or video path have to be specified.")

    if args.imagesPath:
        images_inference(args.modelWeightsPath, args.imagesPath, args.limit, args.outputPath, args.storeImages)
    else:
        video_inference(args.modelWeightsPath, args.videoPath, args.limit, args.outputPath, args.storeImages)