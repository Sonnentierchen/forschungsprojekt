import cv2

def extract_frames(videoPath, frequency):
	count = 0
	videocap = cv2.VideoCapture(videoPath)
	success, image = videocap.read()
	success = True
	frames = []
	while success:
		videocap.set(cv2.CAP_PROP_POS_MSEC, (count * frequency))
		success, image = videocap.read()
		frames.append(image)
		count += 1

	return frames