import cv2
import os

def storeImage(image, frameId, frames, outputImagesPath):
	frames.append({'image' : image,
			'frameNumber' : frameId})
	if outputImagesPath:
		cv2.imwrite(outputImagesPath + "_frame_%d.jpg" % frameId, image)

def extract_frames(videoPath, outputImagesPath, maxNumberOfFrames=0, distribute="even"):
	if not distribute in ["even", "none"]:
		raise ValueError("Distribute needs to be either even or none.")

	videocap = cv2.VideoCapture(videoPath)
	if not videocap.isOpened():
		return []

	success, image = videocap.read()

	length = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))

	# The user might not know that he has provided a number higher than the max. number of frames
	if not maxNumberOfFrames < length:
		maxNumberOfFrames = length

	distribution = length if maxNumberOfFrames == 0 else maxNumberOfFrames
	# Number of frames we need to skip to get an even distribution of the required frames count
	skipFrames = int(round(length / distribution))

	success = True
	frames = []
	frameId = 0

	# For naming pattern Video.avi_framex.jpg
	outputImagesPath = os.path.join(outputImagesPath, os.path.basename(videoPath))

	while success:
		success, image = videocap.read()
		if distribute == "even" and frameId % skipFrames == 0:
			# This is a frame that we have to store, because the user wanted an even
			# distribution and we skipped enough frames to ensure the distribution
			storeImage(image, frameId, frames, outputImagesPath)
		elif distribute == "none" and frameId < maxNumberOfFrames:
			storeImage(image, frameId, frames, outputImagesPath)

		frameId += 1

	return frames

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Extract frames from video file.")
	parser.add_argument("--videoPath",
						"-vp", 
						required=True, 
						metavar="/path/to/video/video.xyz",
						help="The path to the video.")
	parser.add_argument("--outputPath",
						"-op",
						required=True,
						metavar="/path/to/video/conversion/output",
						help="The path where the split video and the annotation data" +
							 " formatted to COCO style will be saved.")
	parser.add_argument("--maxNumberOfFrames",
						"-n",
						required=True,
						default=100,
						help="The maximum number of frames to be extracted.")
	parser.add_argument("--distribute",
						"-d",
						required=True,
						default="even",
						metavar="[even|none]",
						help="Sets whether the frames are to be taken from the beginning" +
						"of the video or at even intervalls.")

	args = parser.parse_args()
	extract_frames(args.videoPath, args.outputPath, int(args.maxNumberOfFrames), args.distribute)