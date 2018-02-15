import os
import dataset_retraining as dataset
from mrcnn import model

import datetime

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

    print("Loading weights ", weights)
    exclude = [] if not args.excludeWeightsOfLayers else args.excludeWeightsOfLayers
    model.load_weights(weights, by_name=True, exclude=exclude)

    dataset_train = dataset.Dataset()
    for index in range(len(trainImagesPaths)):
        trainImagesPath = trainImagesPaths[index]
        trainAnnotationsPath = trainAnnotationsPaths[index]
        assert os.path.exists(trainImagesPath)
        assert os.path.exists(trainAnnotationsPath)
        dataset_train.load_coco(trainImagesPath, trainAnnotationsPath)

    dataset_train.prepare()

    dataset_val = dataset.Dataset()
    if valImagesPaths:
        for index in range(len(valImagesPaths)):
            valImagesPath = valImagesPaths[index]
            valAnnotationsPath = valAnnotationsPaths[index]
            assert os.path.exists(valImagesPath)
            assert os.path.exists(valAnnotationsPath)
            dataset_val.load_coco(valImagesPath, valAnnotationsPath)

    dataset_val.prepare()

    for run in range(runs):
        print("Performing training run {} of {}.".format(run + 1, runs))
        currentLayers = layers[run]
        currentLearningRate = learningRates[run]
        currentEpochs = epochs[run]
        print("Training layers {} with learning rate {} for {} epochs".format(
            currentLayers,
            currentLearningRate,
            currentEpochs))
        model.train(dataset_train, dataset_val,
                learning_rate=currentLearningRate,
                epochs=currentEpochs,
                layers=currentLayers)