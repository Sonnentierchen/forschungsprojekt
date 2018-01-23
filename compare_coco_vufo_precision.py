import os
import calc_bbs as cb
import extract_frames as extract

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# The custom coco helper class implemented by the guys who made the Mask RCNN
import coco

from config import Config
import utils
import model as modellib

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

	print("MS COCO dataset of " + year + " at " + dataset)

	storeOutput = False
	if output is not None:
		storeOutput = True
		print("Saving images to ", output)

	cocoImagesOutputPath = output + "coco/" + year + "/"

	if not os.path.exists(cocoImagesOutputPath):
		os.makedirs(cocoImagesOutputPath)

	dataset_val = coco.CocoDataset()
	cocoDataSet = dataset_val.load_coco(dataset, year, "val", return_coco=True)
	dataset_val.prepare()
	print("Running COCO evaluation on {} images.".format(limit))
	coco.evaluate_coco(model, dataset_val, cocoDataSet, "bbox", limit=int(limit), output=cocoImagesOutputPath, classNames=class_names)

def evaluateVideo(model, videoPath, output, limit):
	if videoPath is None:
		return

	print("Video: ", videoPath)

	# We want only 30 frames per Video
	frames = extract.extract_frames(videoPath, output, limit)

	# TODO: extract annotation data from Vatic
	annotations = []

	#for i in range(len(frames)):
		#annotations.append({
		#		"image_id": i,
        #        "category_id": "car",
        #        "bbox": [x, y, width, height],
        #        "score": score})
	evaluateVideoFrames(model, frames, annotations)

def evaluateVideoFrames(model, frames, annotations):
	print("test")

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Compare performance of" +
												 " Mask R-CNN on " +
												 "MS COCO and videos.")
	parser.add_argument("--video", 
						"-v", 
						required=False, 
						metavar="/path/to/video/video.xyz",
						help="The path to the video")
	parser.add_argument("--videoOutput",
						required=False,
						metavar="/path/to/video/output/folder",
						help="The path where the frames of the video with the bounding" +
							 " boxes drawn are saved at")
	parser.add_argument("--year",
						"-y",
						required=False,
						metavar="2017",
						help="The year of the MS COCO dataset to be used. This defines" +
							 "what the annotation files are named like.")
	parser.add_argument("--dataset",
						"-d",
						required=False,
						metavar="/path/to/mscoco/dataset",
						help="The path to the MS COCO dataset")
	parser.add_argument("--cocoOutput",
						required=False,
						metavar="/path/to/coco/output/folder",
						help="The path where the frames of the coco images with the bounding" +
							 " boxes drawn are saved at")
	parser.add_argument("--limit",
						"-l",
						required=False,
						metavar="30",
						help="The maximum number of images to run inference on for COCO and" +
							 " for the videos")

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

	evaluateCoco(model, args.dataset, args.year, args.cocoOutput, limit)
	evaluateVideo(model, args.video, args.videoOutput, limit)