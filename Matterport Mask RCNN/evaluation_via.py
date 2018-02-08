import os
import conversion
import evaluation
import json

def evaluate_via(weightsPath, imagesPath, viaGroundTruthAnnotationsPath, outputPath, outputModelPath, limit):
    # Only converts data, does not store it
    converted_via_data = conversion.via_data_to_coco_evaluation_format(imagesPath, viaGroundTruthAnnotationsPath)
    dirname = os.path.dirname(viaGroundTruthAnnotationsPath)
    converted_via_data_path = os.path.splitext(viaGroundTruthAnnotationsPath)[0] + "_converted.json"
    with open(converted_via_data_path, "w") as converted_json_file:
        json.dump(converted_via_data, converted_json_file)

    evaluation.evaluate(weightsPath, imagesPath, converted_via_data_path, outputPath, outputModelPath, limit)

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
                        metavar="/path/to/viagrountruth/",
                        help="The path to the ground truth file in VIA format.")
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

    evaluate_via(args.weightsPath, args.imagesPath, args.groundTruth, args.outputPath, args.outputModelPath, limit)