import os

def transform(self, vaticDataInputPath, cocoDataOutputPath):
	"""
	Loads the VUFO data file at the given location, extracts the information
	and stores it in a COCO compatible file format at the specified output
	location
	"""

	# Irrelevant for us but request by coco eval
	cocoCaptionsData = {}
	cocoCaptionsData["annotations"] = []

	cocoData = {}
	# Description for the data
	cocoData["info"] = {"description" : "Frames of videos from VUFO",
						"url" : "",
						"version" : "1.0",
						"year" : "2018",
						"contributor" : "",
						"date_created" : ""}
	# Add a default license for all images
	cocoData["licenses"] = [{"url" : "",
							"id" : "1",
							"name" : "license"}]
	cocoData["images"] = []
	cocoData["annotations"] = []
	cocoData["categories"] = []

	basePath = os.path.dirname(cocoDataOutputPath)
	basePath = os.path.join(basePath, os.pardir)
	basePath = os.path.join(basePath, "images")

	imageIndex = 0
	categoryIndex = 0
	categoryIndexDict = {}
	annotationIndex = 0

	with open(vufoDataInputPath) as jsonFile:
		jsonData = json.load(jsonFile)
		for entry in jsonData:
			# Entry of an image
			entry = jsonData[entry]
			imageFile = entry["filename"]
			image = cv2.imread(os.path.join(basePath, imageFile))
			width, height, channels = image.shape
			cocoData["images"].append({"license" : "1",
							   "file_name" : imageFile,
							   "coco_url" : "",
							   "height" : height,
							   "width" : width,
							   "date_captured" : "",
							   "flickr_url" : "",
							   "id" : imageIndex})
			cocoCaptionsData["annotations"].append({"image_id" : imageIndex,
													"id" : imageIndex,
													"caption" : ""})

			regions = entry["regions"]
			for regionIndex in regions:
				region = regions[regionIndex]
				# Process region attributes
				regionAttributes = region["region_attributes"]
				# Check if already have this class and, if not, create it
				_categoryIndex = -1
				if regionAttributes["class"] in categoryIndexDict:
					_categoryIndex = categoryIndexDict[regionAttributes["class"]]
				else:
					_categoryIndex = categoryIndex
					categoryIndexDict[regionAttributes["class"]] = _categoryIndex
					cocoData["categories"].append({"supercategory" : regionAttributes["supertype"],
											   "id" : _categoryIndex,
											   "name" : regionAttributes["class"]})
					categoryIndex += 1

				# Process region values
				shapeAttributes = region["shape_attributes"]
				cocoData["annotations"].append({"segmentation" : [[]],
												"area" : "0",
												"iscrowd" : "0",
												"image_id" : str(imageIndex),
												"bbox" : [int(shapeAttributes["x"]),
														  int(shapeAttributes["y"]),
														  int(shapeAttributes["width"]),
														  int(shapeAttributes["height"])],
												"category_id" : _categoryIndex,
												"id" : annotationIndex})
				annotationIndex += 1

			imageIndex += 1

	cocoCaptionsData["licenses"] = cocoData["licenses"]
	cocoCaptionsData["info"] = cocoData["info"]
	cocoCaptionsData["images"] = cocoData["images"]

	# TODO: read in VUFO file
	# TODO: parse and write at output location

	with open(cocoDataOutputPath, "w") as cocoJsonFile:
		json.dump(cocoData, cocoJsonFile)
	
	return cocoData