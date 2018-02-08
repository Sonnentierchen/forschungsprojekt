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

class EvaluationConfig(Config):
    """Configuration for evaluation on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    GPU_COUNT = 1

class EvaluationDataset(coco.CocoDataset):
    def load_coco(self, imagesPath, groundTruthAnnotationsPath, year=None, subset=None, class_ids=None,
                  class_map=None, return_coco=True):
        # Override method to provide own loading of data
        # Thus we can omit year and do not need to concatenate paths as
        # dataset_dir is the actual dataset file already

        if not os.path.exists(groundTruthAnnotationsPath):
            raise ValueError("Annotations file does not exist.")

        coco = COCO(groundTruthAnnotationsPath)

        # Load all classes or a subset?
        if not class_ids:
        # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
        for id in class_ids:
            image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
            "coco", image_id=i,
            path=os.path.join(imagesPath, coco.imgs[i]['file_name']),
            width=coco.imgs[i]["width"],
            height=coco.imgs[i]["height"],
            annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=False)))
        if return_coco:
            return coco

def evaluate(weightsPath, imagesPath, groundTruthAnnotationsPath, outputPath, outputModelPath, limit):

    config = EvaluationConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=outputModelPath, config=config)

    # Load weights trained on MS-COCO
    print("Loading weights...")
    model.load_weights(weightsPath, by_name=True)
    dataset = EvaluationDataset()
    # We do not need to provide whether we want val or train as dataset (we only use the via data) and
    # no year
    dataset_data = dataset.load_coco(imagesPath, groundTruthAnnotationsPath, return_coco=True)
    dataset.prepare()

    numberOfImages = str(limit) if limit > 0 else "all"
    print("Running COCO evaluation on " + numberOfImages + " images.")

    # COCO only prints to console but we want to store logs
    originalStdout = sys.stdout
    outputFilePath = os.path.join(outputPath, "log_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt")
    outputFile = open(outputFilePath, "w")
    sys.stdout = outputFile

    coco.evaluate_coco(model, dataset, dataset_data, "bbox", limit=limit, output=outputModelPath, classNames=coco.COCO_CLASSES)

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