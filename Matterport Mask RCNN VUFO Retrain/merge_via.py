import os

def merge_via_annotations(basePath, folderPrefix, filePrefix, fileSuffix, outputPath):
	assert os.path.exists(basePath)

	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	with open(os.path.join(outputPath, "annotations.json"), "w") as outputFile:
		result = []
		annotationFolders = [x[0] for x in os.walk(basePath) if x[0].startswith(folderPrefix)]
		for index in range(len(annotationFolders)):
			annotationFilepath = os.path.join(basePath, annotationFolders[index], filePrefix + str(index))
			with open(annotationFilepath) as annotationFile:
				data = json.load(annotationFile)
			        for entry in data:
			            result.append({entry : data[entry]})
		json.dump(result, outputFile)

import argparse

    parser = argparse.ArgumentParser(description="Merge an arbitraty number of VIA-format groundtruth annotations.")
    parser.add_argument("--annotationBasePath",
    					"-a",
    					nargs="+",
    					required=True,
    					metavar="/path/to/folders/with/annotations",
    					help="The paths to the annotation files.")
    parser.add_argument("--folderPrefix",
    					required=True,
    					help="The prefix of the folders that contain the respective annotations.")
    parser.add_argument("--filePrefix",
    					required=True,
    					help="The prefix of the annotation files.")
    parser.add_argument("--fileSuffix",
    					required=True,
    					help="The suffix of the annotation files.")
    args = parser.parse_args()

