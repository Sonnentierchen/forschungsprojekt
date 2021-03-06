import os
import dataset_retraining as dataset
from mrcnn import model

import datetime

def log(text, logFile):
    logFile.write(text)
    print(text)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("--weightsPath",
                        "-w",
                        required=True,
                        metavar="/path/to/pre-trained/weights",
                        help="The path to the weights to start training from.")
    parser.add_argument("--excludeWeightsOfLayers",
                        required=False,
                        nargs='+',
                        help="The layers that are to be excluded from weight loading. This way " +
                        "the weights will initialized randomly and can be re-trained from scratch.")
    parser.add_argument("--logsPath",
                        "-l",
                        required=True,
                        help="The path where the logs and final weights are to be stored.")
    parser.add_argument("--trainImagesPaths",
                        required=True,
                        nargs='+',
                        help="The paths to the images of the training data." + 
                        " All images will be loaded together.")
    parser.add_argument("--trainAnnotationsPaths",
                        required=True,
                        nargs='+',
                        help="The path to the COCO-style annotations of the training data. " +
                        " All annotations will be loaded together.")
    parser.add_argument("--valImagesPaths",
                        required=False,
                        nargs='+',
                        help="The paths to the images of the validation data." + 
                        " All images will be loaded together.")
    parser.add_argument("--valAnnotationsPaths",
                        required=False,
                        nargs='+',
                        help="The paths to the COCO-style annotations of the validation data." + 
                        " All annotations will be loaded together.")
    parser.add_argument("--runs",
                        "-r",
                        required=True,
                        type=int,
                        help="The number of training runs that are to be completed." +
                        " The number has to have a matching number of entries in" + 
                        " the list of learning rates, epochs and layers.")
    parser.add_argument("--learningRates",
                        required=True,
                        nargs='+',
                        type=float,
                        help="The list of learning rates. This specifies the learning" +
                        " rate for each training run.")
    parser.add_argument("--epochs",
                        required=True,
                        nargs='+',
                        type=int,
                        help="The list of epochs. This specifies the epochs per" + 
                        " training run.")
    parser.add_argument("--layers",
                        required=True,
                        nargs='+',
                        help="The list of layers. This specifies the layers that" + 
                        " are to be trained in the respective training run.")

    args = parser.parse_args()

    weights = args.weightsPath
    assert os.path.exists(weights)

    trainImagesPaths = args.trainImagesPaths
    trainAnnotationsPaths = args.trainAnnotationsPaths
    valImagesPaths = args.valImagesPaths
    valAnnotationsPaths = args.valAnnotationsPaths

    assert len(trainImagesPaths) == len(trainAnnotationsPaths)
    if not valImagesPaths is None:
        assert len(valImagesPaths) == len(valAnnotationsPaths)

    today = datetime.datetime.now()
    todayString = "{}.{:02}.{:02}".format(today.year, today.month, today.day)

    logs = args.logsPath
    logs = os.path.join(logs, todayString)
    if not os.path.exists(logs):
        os.makedirs(logs)

    runs = args.runs
    learningRates = args.learningRates
    epochs = args.epochs
    layers = args.layers

    assert runs == len(learningRates) == len(epochs) == len(layers)

    config = dataset.Config()
    model = model.MaskRCNN(mode="training", config=config, model_dir=logs)

    log_file = open(os.path.join(logs, "log.txt"), "w")

    log("Startin from weights at: " + weights + "\n", log_file)
    if args.excludeWeightsOfLayers:
        log("Excluding layers {} from weight loading to fully retrain them.\n".format(args.excludeWeightsOfLayers), log_file)
        model.load_weights(weights, by_name=True, exclude=args.excludeWeightsOfLayers)
    else:
        model.load_weights(weights, by_name=True)

    dataset_train = dataset.Dataset()
    for index in range(len(trainImagesPaths)):
        trainImagesPath = trainImagesPaths[index]
        trainAnnotationsPath = trainAnnotationsPaths[index]
        assert os.path.exists(trainImagesPath), "Images path {} does not exist.".format(trainImagesPath)
        assert os.path.exists(trainAnnotationsPath), "Annotations file {} does not exist.".format(trainAnnotationsPath)
        log("Adding images at {} with groundtruth at {}.\n".format(trainImagesPath, trainAnnotationsPath), log_file)
        dataset_train.load_coco(trainImagesPath, trainAnnotationsPath)

    dataset_train.prepare()

    dataset_val = dataset.Dataset()
    if valImagesPaths:
        for index in range(len(valImagesPaths)):
            valImagesPath = valImagesPaths[index]
            valAnnotationsPath = valAnnotationsPaths[index]
            assert os.path.exists(valImagesPath), "Images path {} does not exist.".format(valImagesPath)
            assert os.path.exists(valAnnotationsPath), "Annotations file {} does not exist.".format(valAnnotationsPath)
        log("Adding images at {} with groundtruth at {}.\n".format(valImagesPath, valAnnotationsPath), log_file)
        dataset_val.load_coco(valImagesPath, valAnnotationsPath)

    dataset_val.prepare()

    for run in range(runs):
        log("Performing RE-training run {} of {}.\n".format(run + 1, runs), log_file)
        currentLayers = layers[run]
        currentLearningRate = learningRates[run]
        currentEpochs = epochs[run]
        log("Training layers {} with learning rate {} for {} epochs.\n".format(
            currentLayers,
            currentLearningRate,
            currentEpochs), log_file)
        model.train(dataset_train, dataset_val,
                learning_rate=currentLearningRate,
                epochs=currentEpochs,
                layers=currentLayers)