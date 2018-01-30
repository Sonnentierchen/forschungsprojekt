import os
import cv2
import time
import numpy as np
from pycocotools.coco import COCO
import mrcnn.utils as utils

import json

############################################################
#  VUFO Dataset
############################################################

class VufoDataset(utils.Dataset):
	
	def load_vufo(self, dataset_dir, class_ids=None, class_map=None, return_vufo=False):
		"""Loads the VUFO data. The data has to be in the same form (i.e. subfolders
		   and data structures) as the COCO dataset because the COCO Evaluation functions
		   are used. 
		dataset_dir: The root directory of the VUFO dataset.
		subset: What to load (train, val).
		class_ids: If provided, only loads images that have the given classes.
		return_vufo: If tue, returns the VUFO object.
		"""
		image_dir = os.path.join(dataset_dir, "images")

		vufo = COCO(os.path.join(dataset_dir, "annotations/instances.json"))

		# Load all classes or a subset?
		if not class_ids:
			# All classes
			class_ids = sorted(vufo.getCatIds())

		# All images or a subset?
		if class_ids:
			image_ids = []
			for id in class_ids:
				image_ids.extend(list(vufo.getImgIds(catIds=[id])))
				# Remove duplicates
				image_ids = list(set(image_ids))
		else:
			# All images
			image_ids = list(vufo.imgs.keys())

		# Add classes
		for i in class_ids:
			self.add_class("coco", i, vufo.loadCats(i)[0]["name"])

		# Add images
		for i in image_ids:
			self.add_image(
			"coco", image_id=i,
			path=os.path.join(image_dir, vufo.imgs[i]['file_name']),
			width=vufo.imgs[i]["width"],
			height=vufo.imgs[i]["height"],
			annotations=vufo.loadAnns(vufo.getAnnIds(imgIds=[i], 
													 catIds=class_ids, 
													 iscrowd=False)))

		if return_vufo:
			return vufo