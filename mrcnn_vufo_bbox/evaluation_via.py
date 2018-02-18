import os
import conversion
import evaluation
import json

def evaluate_via(weightsPath, imagesPaths, viaGroundTruthAnnotationsPaths, outputPath, outputModelPath, limit):
    # Only converts data, does not store it
    assert len(imagesPaths) == len(viaGroundTruthAnnotationsPaths)

    if len(imagesPaths) == 0:
        return

    convertedGroundTruthPaths = []
    for index in range(len(imagesPaths)):
        viaGroundTruthPath = viaGroundTruthAnnotationsPaths[index]
        convertedViaData = conversion.via_data_to_coco_evaluation_format(imagesPaths[index], viaGroundTruthPath)
        convertedViaDataPath = os.path.splitext(viaGroundTruthPath)[0] + "_converted.json"
        with open(convertedViaDataPath, "w") as convertedJsonFile:
            json.dump(convertedViaData, convertedJsonFile)
        convertedGroundTruthPaths.append(convertedViaDataPath)

    evaluation.evaluate(weightsPath, imagesPaths, convertedGroundTruthPaths, outputPath, outputModelPath, limit)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Use the COCO evaluation functions on the detection results by the " +
        "model with pre-computed weights.")
    parser.add_argument("--weightsPath",
                        "-w", 
                        required=True, 
                        metavar="/path/to/network/weights/",
                        help="The path to the pre-trained network weights in hf5 format.")
    parser.add_argument("--imagesPaths",
                        "-i",
                        required=True,
                        nargs='+',
                        metavar="/paths/to/images/",
                        help="The path to the folders with the images that are to be evaluated.")
    parser.add_argument("--groundTruthPaths",
                        "-g",
                        required=True,
                        nargs='+',
                        metavar="/paths/to/viagrountruths/",
                        help="The paths to the ground truth files in VIA format.")
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

    evaluate_via(args.weightsPath, args.imagesPaths, args.groundTruthPaths, args.outputPath, args.outputModelPath, limit)