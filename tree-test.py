import numpy as np
import sys, random
from matplotlib import pyplot as plt
from machineLearning.rf import (
    DecisionTree,
    Gini, Entropy, MAE, MSE,
    Mode, Mean,
    CART, ID3, C45,
    UsersChoice, Variance, Random, MutualInformation, ANOVA, KendallTau
)
from machineLearning.metric import ConfusionMatrix
from machineLearning.utility import Time
from machineLearning.settings.treeSettings import TreeSettings
from machineLearning.data import Data


def dataShift(dims):
    offSet = [0.25, 0.5, 0.25]
    diffLen = abs(len(offSet) - dims)
    offSet.extend([0] * diffLen)
    random.shuffle(offSet)
    return offSet[:dims]


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


if __name__ == "__main__":
    settings = TreeSettings()
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

    # Create and train a decision tree
    timer.start()
    print('Setting up tree')
    tree = DecisionTree(settings['depth'], settings['minSamples'])
    tree.setComponent(getImurity(settings['impurity']))
    tree.setComponent(getLeaf(settings['leaf']))
    tree.setComponent(getSplit(settings['split'],settings['percentile']))
    if settings['featSelection'] is not None:
        tree.setFeatureSelection(settings['featSelection'], settings['featParameter']) # Use random feature selection
    timer.record("Tree setup")

    # Train the tree using the training data
    timer.start()
    print('begin training...')
    #tree.train(trainData,trainLabels)
    tree.train(data.trainSet.data,data.trainSet.labels.argmax(1))
    timer.record("Training")

    # Evaluate the tree on the validation data
    timer.start()
    print('making predictions...')
    #prediction = tree.eval(validData)
    prediction = tree.eval(data.evalSet.data)
    timer.record("Prediction")

    # Print the trained decision tree
    print(tree)
    print()

    # Compute confusion matrix to evaluate the performance of the decision tree
    confusion = ConfusionMatrix(2)
    #confusion.update(prediction, validLabels)
    confusion.update(prediction, data.evalSet.labels.argmax(1))
    confusion.percentages()
    confusion.calcScores()

    # Compute confusion matrix to evaluate the performance of the decision tree
    print(confusion)
    print()
    print(timer)
