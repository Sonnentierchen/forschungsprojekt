import matplotlib
matplotlib.use('Agg')

import os
import sys
import datetime

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import mrcnn.coco as coco
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

import misc
import dataset

def evaluate(weightsPath, imagesPath, groundTruthAnnotationsPath, outputPath, outputModelPath, limit):

    if not os.path.exists(weightsPath):
        raise ValueError("The path to the pre-trained weights does not exist.")

    if not os.path.exists(imagesPath):
        raise ValueError("The path to the images does not exist.")

    if not os.path.exists(groundTruthAnnotationsPath):
        raise ValueError("The path to the groundtruth annotation files does not exist.")

    today = datetime.datetime.now()
    todayString = "{}.{:02}.{:02}".format(today.year, today.month, today.day)

    outputPath = os.path.join(outputPath, todayString)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    outputModelPath = os.path.join(outputModelPath, todayString)
    if not os.path.exists(outputModelPath):
        os.makedirs(outputModelPath)

    config = dataset.Config()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=outputModelPath, config=config)

    # Load weights trained on MS-COCO
    print("Loading weights...")
    model.load_weights(weightsPath, by_name=True)
    dataset = dataset.Dataset()
    # We do not need to provide whether we want val or train as dataset (we only use the via data) and
    # no year
    dataset_data = dataset.load_coco(imagesPath, groundTruthAnnotationsPath, return_coco=True)
    dataset.prepare()

    numberOfImages = str(limit) if limit > 0 else "all"
    print("Running COCO evaluation on " + numberOfImages + " images.")

    # COCO only prints to console but we want to store logs
    originalStdout = sys.stdout
    outputFilePath = os.path.join(outputPath, "log_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt")

    with open(outputFilePath, "w") as outputFile:
        sys.stdout = outputFile
        coco.evaluate_coco(model, dataset, dataset_data, "bbox", limit=limit)

    sys.stdout = originalStdout

    # Print logs to console
    with open(outputFilePath, "r") as outputFileContents:
        print(outputFileContents.read())

    print("COCO evaluation results in " + outputFilePath)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Use the COCO evaluation functions on the detection results by the " +
        "model with pre-computed weights.")
    parser.add_argument("--weightsPath",
                        "-w", 
                        required=True, 
                        metavar="/path/to/network/weights/",
                        help="The path to the pre-trained network weights in hf5 format.")
    parser.add_argument("--imagesPath",
                        "-i",
                        required=True,
                        metavar="/path/to/images/",
                        help="The path to the folder with the images that are to be evaluated.")
    parser.add_argument("--groundTruth",
                        "-g",
                        required=True,
                        metavar="/path/to/grountruth/",
                        help="The path to the ground truth file in COCO format.")
    parser.add_argument("--outputPath",
                        "-o",
                        required=True,
                        metavar="/path/to/output/",
                        help="The path to the folder where the results are to be stored. Results " +
                        "are the logs as well as the images with the drawn in bounding boxes.")
    parser.add_argument("--outputModelPath",
                        "-m",
                        required=False,
                        metavar="/path/to/model/output",
                        help="The path to the folder where the model stores outputs like its internal logs " +
                        "and the weights if it is being trained. If unsure set to --outputPath.")
    parser.add_argument("--limit",
                        "-l",
                        required=True,
                        default=100,
                        help="The number of images to be used for evaluation. If limit is 0, all images will be used.")

    args = parser.parse_args()

    limit = args.limit if args.limit else 30
    limit = int(limit)
    if limit < 0:
        limit = 0

    if not args.outputModelPath:
        args.outputModelPath = args.outputPath

    evaluate(args.weightsPath, args.imagesPath, args.groundTruth, args.outputPath, args.outputModelPath, limit)