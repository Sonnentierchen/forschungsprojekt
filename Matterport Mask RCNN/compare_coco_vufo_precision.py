import matplotlib
matplotlib.use('Agg')

import os
import sys
import datetime
import extract_frames as extract

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# The custom coco helper class implemented by the guys who made the Mask RCNN
import coco
# The adaption of the coco helper class to the VUFO videos
import vufo

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

ROOT_DIR = os.getcwd()

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

class InferenceConfig(Config):
	"""Configuration for training on MS COCO.
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

def evaluateCoco(model, dataset, year, output, limit):
	if dataset is None or year is None:
		return

	cocoInputPath = os.path.join(dataset, year)

	if not os.path.exists(cocoInputPath):
		return

	print("MS COCO dataset settings:")
	print("MS COCO dataset of " + year + " at " + cocoInputPath)

	cocoOutputPath = None
	if output is not None:
		cocoOutputPath = os.path.join(output, year) 
		print("Saving coco result images to " + cocoOutputPath)

	if not os.path.exists(cocoOutputPath):
		os.makedirs(cocoOutputPath)

	print("\nProcessing COCO datatset:")
	dataset_val = coco.CocoDataset()
	cocoDataSet = dataset_val.load_coco(cocoInputPath, year, "val", return_coco=True)
	dataset_val.prepare()
	print("Running COCO evaluation on {} images.".format(limit))

	# Redirect output stream to file, unfortunately coco doesn't let us define a file
	# instead of printing to the console
	originalStdout = sys.stdout
	outputFilePath = cocoOutputPath + "/log_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".txt"
	outputFile = open(outputFilePath, "w")
	sys.stdout = outputFile

	coco.evaluate_coco(model, dataset_val, cocoDataSet, "bbox", limit=limit, output=cocoOutputPath, classNames=class_names)
	sys.stdout = originalStdout
	print("COCO evaluation results in " + outputFilePath)

def evaluateVufo(model, videoPath, videoConversionOutput, output, limit):
	if videoPath is None or not os.path.exists(videoPath):
		return

	if videoConversionOutput is None:
		videoConversionOutput = os.path.join(videoPath, "vufo")


	print("Video settings:")
	print("Video at: ", videoPath)

	if not os.path.exists(output):
		return

	# We want only 30 frames per Video
	framesOutputPath = os.path.join(videoConversionOutput, "images")
	frames = extract.extract_frames(videoPath, framesOutputPath, limit)
	# We might have extracted less frames than limit
	limit = len(frames)

	dataset_val = vufo.VufoDataset()
	# TODO: Adjust for vatic XML
	# We get the path to the video including the video filename, that's why
	# we need to strip the filename off
	vufoAnnotationData = os.path.join(os.path.dirname(videoPath), "annotations.json")
	cocoFormatedAnnotationData = os.path.join(videoConversionOutput, "annotations", "instances.json")
	dataset_val.transform(vufoAnnotationData, cocoFormatedAnnotationData)
	vufoDataSet = dataset_val.load_vufo(videoConversionOutput, return_vufo=True)

	# Redirect output stream to file, unfortunately coco doesn't let us define a file
	# instead of printing to the console
	originalStdout = sys.stdout
	outputFilePath = output + "/log_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".txt"
	outputFile = open(outputFilePath, "w")
	sys.stdout = outputFile

	coco.evaluate_coco(model, dataset_val, vufoDataSet, "bbox", limit=limit, output=output, classNames=class_names)
	sys.stdout = originalStdout
	print("\n")
	print("COCO evaluation results in " + outputFilePath)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Compare performance of" +
												 " Mask R-CNN on " +
												 "MS COCO and videos.")
	parser.add_argument("--videoPath",
						"-vp", 
						required=False, 
						metavar="/path/to/video/video.xyz",
						help="The path to the video.")
	parser.add_argument("--videoConversionOutputPath",
						"-vcop",
						required=False,
						metavar="/path/to/video/conversion/output",
						help="The path where the split video and the annotation data" +
							 " formatted to COCO style will be saved.")
	parser.add_argument("--vufoOutputPath",
						"-vop",
						required=False,
						metavar="/path/to/video/output/folder",
						help="The path where the frames of the vufo video with the bounding" +
							 " boxes drawn are saved at.")
	parser.add_argument("--year",
						"-y",
						required=False,
						metavar="2017",
						help="The year of the MS COCO dataset to be used. This defines" +
							 "what the annotation files are named like.")
	parser.add_argument("--cocoPath",
						"-cp",
						required=False,
						metavar="/path/to/mscoco/dataset",
						help="The path to the MS COCO dataset. /coco/ and /year/ will" +
							 " automatically be appended.")
	parser.add_argument("--cocoOutputPath",
						"-cop",
						required=False,
						metavar="/path/to/coco/output/folder",
						help="The path where the frames of the coco images with the bounding" +
							 " boxes drawn are saved at.")
	parser.add_argument("--limit",
						"-l",
						required=False,
						metavar="30",
						help="The maximum number of images to run inference on for COCO and" +
							 " for the videos.")

	args = parser.parse_args()

	config = InferenceConfig()
	config.display()

	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	print("Loading weights...")
	model.load_weights("mask_rcnn_coco.h5", by_name=True)

	limit = args.limit if args.limit else 30
	limit = int(limit)
	if limit < 0:
		limit = 0

	evaluateCoco(model, args.cocoPath, args.year, args.cocoOutputPath, limit)
	evaluateVufo(model, args.videoPath, args.videoConversionOutputPath, args.vufoOutputPath, limit)