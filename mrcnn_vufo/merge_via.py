import os
import json

def merge_via_annotations(basePath, folderPrefix, start, number, filePrefix, fileSuffix, outputPath):
    assert os.path.exists(basePath)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    with open(os.path.join(outputPath, "annotations.json"), "w") as outputFile:
        result = {}
        for index in range(start, number + 1):
            annotationFilepath = os.path.join(basePath, folderPrefix + str(index), filePrefix + str(index) + fileSuffix + ".json")
            with open(annotationFilepath) as annotationFile:
                data = json.load(annotationFile)
                for entry in data:
                    result[entry] = data[entry]
        json.dump(result, outputFile)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Merge an arbitraty number of VIA-format groundtruth annotations.")
    parser.add_argument("--annotationBasePath",
        "-a",
        required=True,
        metavar="/path/to/folder/with/annotation/folders",
        help="The paths to the annotation files.")
    parser.add_argument("--folderPrefix",
        required=True,
        help="The prefix of the folders that contain the respective annotations.")
    parser.add_argument("--start",
                        required=True,
                        help="The number to start indexing folders from.")
    parser.add_argument("--number",
                        required=True,
                        help="The number of annotation files to merge.")
    parser.add_argument("--filePrefix",
        required=True,
        help="The prefix of the annotation files.")
    parser.add_argument("--fileSuffix",
        required=True,
        help="The suffix of the annotation files.")
    parser.add_argument("--outputPath",
        required=True,
        help="The path where the result will be stored as annotations.json")
    args = parser.parse_args()

    merge_via_annotations(args.annotationBasePath, args.folderPrefix, int(args.start), int(args.number), args.filePrefix, args.fileSuffix, args.outputPath)