import numpy as np
import sys
from machineLearning.rf import (
    RandomForest, DecisionTree,
    Gini, Entropy, MAE, MSE,
    Mode, Mean, Confidence,
    UsersChoice, Variance, Random, MutualInformation, ANOVA, KendallTau,
    CART, ID3, C45,
    AdaBoosting, GradientBoosting,
    Majority, Confidence, Average, Median
)
from machineLearning.metric import ConfusionMatrix
from machineLearning.utility.timer import Time
from machineLearning.settings import ForrestSettings
from machineLearning.data import Data


def getImurity(impurity: str):
    if impurity == 'gini':
        return Gini() # Use Gini index as the impurity measure
    elif impurity == 'entropy':
        return Entropy() # Use Entropy index as the impurity measure
    elif impurity == 'mae':
        return MAE() # Use MAE index as the impurity measure
    elif impurity == 'mse':
        return MSE() # Use MSE index as the impurity measure


def getLeaf(leaf: str):
    if leaf == 'mode':
        return Mode() # Use mode as the leaf function
    elif leaf == 'mean':
        return Mean() # Use mean as the leaf function


def getSplit(split: str, percentile: int = None):
    if split == 'id3':
        return ID3(percentile) # Use ID3 algorithm for splitting
    elif split == 'c45':
        return C45(percentile) # Use C4.5 algorithm for splitting
    elif split == 'cart':
        return CART(percentile) # Use CART algorithm for splitting


def getVoting(voting: str, weights: list):
    if voting == 'majority':
        return Majority(weights)
    elif voting == 'confidence':
        return Confidence(weights)
    elif voting == 'average':
        return Average(weights)
    elif voting == 'median':
        return Median(weights)


def getFeatureSelection(selection: str, *args):
    if selection == 'choice':
        return UsersChoice(*args)
    elif selection == 'variance':
        return Variance(*args)
    elif selection == 'random':
        return Random(*args)
    elif selection == 'mutual':
        return MutualInformation(*args)
    elif selection == 'anova':
        return ANOVA(*args)
    elif selection == 'kendall':
        return KendallTau(*args)


def getBooster(booster: str):
    if booster == 'adaptive':
        return AdaBoosting()
    elif booster == 'gradient':
        return GradientBoosting()


if __name__ == "__main__":
    settings = ForrestSettings()
    try:
        configFile = sys.argv[1]
        settings.getConfig(configFile)
        settings.setConfig()
    except IndexError:
        pass
    print(settings)

    # Create a timer object to measure execution time
    timer = Time()

    print("Importing data...\n")
    timer.start()
    data = Data(trainAmount=settings['trainAmount'], evalAmount=settings['validAmount'], dataPath=settings['dataPath'], normalize=settings['normalize'])
    data.inputFeatures(*settings['features'])
    data.importData(*settings['dataFiles'])
    print(data)
    timer.record("Importing Data")

    # Set up random forest
    timer.start()
    print("setting up forrest")
    forrest = RandomForest(bootstrapping=settings['bootstraping'], retrainFirst=settings['retrainFirst'])
    if settings['booster'] is not None:
        forrest.setComponent(getBooster(settings['booster']))
    if settings['voting'] is not None:
        forrest.setComponent(getVoting(settings['voting'], settings['votingWeights']))
    for i in range(settings['numTrees']):
        tree = DecisionTree(settings['depth'][i], settings['minSamples'][i])
        tree.setComponent(getImurity(settings['impurity'][i]))
        tree.setComponent(getLeaf(settings['leaf'][i]))
        tree.setComponent(getSplit(settings['split'][i], settings['percentile'][i]))
        if settings['featSelection'][i] is not None:
            tree.setComponent(getFeatureSelection(settings['featSelection'][i], settings['featParameter'][i]))
        forrest.append(tree)
    timer.record("Forrest setup")

    # Train the random forest
    timer.start()
    print("begin training")
    forrest.train(data.trainSet.data,data.trainSet.labels.argmax(1))
    timer.record("Training")

    # Evaluate the random forest
    timer.start()
    print("making predictions\n")
    #prediction = forrest.eval(validData)
    prediction = tree.eval(data.evalSet.data)
    timer.record("Prediction")
    print(forrest)
    print()

    # Calculate and print confusion matrix
    confusion = ConfusionMatrix(2)
    confusion.update(prediction, data.evalSet.labels.argmax(1))
    confusion.percentages()
    confusion.calcScores()
    print(confusion)
    print()

    # Print total execution time
    print(timer)