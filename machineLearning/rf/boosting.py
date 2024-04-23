import numpy as np
from abc import ABC, abstractmethod
from .decisionTree import DecisionTree
from ..data.dataLoader import DataSet


class Boosting(ABC):
    """
    Abstract base class for implementing boosting algorithms.
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abstractmethod
    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the decision tree on the data and targets.
        """
        pass

    @property
    def qualifiedName(self) -> tuple:
        """Returns the fully qualified name of the class."""
        return self.__class__.__module__, self.__class__.__name__


class AdaBoosting(Boosting):
    """
    Adaptive Boosting (AdaBoost) implementation.
    """
    def __init__(self):
        super().__init__()
        self.alpha = []

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the AdaBoost model using a DecisionTree.
        """
        if data.weights is None:
            data.weights = np.ones(len(data)) / len(data)
        tree.train(data)
        self.updateWeights(tree, data)

    def updateWeights(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Updates the data weights based on the prediction errors.
        """
        predictions = tree.eval(data)
        errorRate = np.sum(data.weights[data.targets != predictions]) / np.sum(data.weights)
        treeWeights = 0.5 * np.log((1 - errorRate) / errorRate)
        self.alpha.append(treeWeights)
        data.weights[data.targets == predictions] *= np.exp(-treeWeights)
        data.weights[data.targets != predictions] *= np.exp(treeWeights)
        data.weights /= np.sum(data.weights)


class GradientBoosting(Boosting):
    """
    Gradient Boosting implementation.
    """
    def __init__(self, learningRate=0.1) -> None:
        super().__init__()
        self.residuals = None  # stores the summation of all errors
        self.predictions = {}  # stores past predictions
        self.counter = 0  # counts time steps
        self.learningRate = learningRate  # Learning rate for boosting

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the GradientBoosting model using a DecisionTree.
        """
        if self.residuals is None:
            tree.train(data)  # Assuming data includes targets
            self.residuals = data.targets - tree.eval(data)
        else:
            updated_targets = data.targets - self.learningRate * self.residuals
            tree.train(data)  # Train on original data but with new targets
            self.residuals = updated_targets - tree.eval(data)

        self.predictions[self.counter] = tree.eval(data)
        self.counter += 1


class XGBoosting(Boosting):
    """
    Extreme Gradient Boosting (XGBoost) implementation.
    Not working yet
    """
    def __init__(self, learningRate=0.1, regularization=1.0):
        super().__init__()
        self.residuals = None  # Summation of all errors
        self.predictions = {}  # Storing past predictions
        self.counter = 0  # Counting time steps
        self.learningRate = learningRate  # Learning rate for boosting
        self.regularization = regularization  # Regularization term

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the XGBoost model using a DecisionTree.
        """
        # Calculate gradients and hessians for the first round or subsequent rounds
        if self.counter == 0:
            gradients = data.targets  # initial gradients are the targets themselves for regression
            hessians = np.ones_like(data.targets)  # assuming squared error loss
        else:
            # Update gradients and hessians based on the latest residuals and predictions
            gradients = data.targets - self.residuals
            hessians = np.ones_like(data.targets)  # constant hessian for squared error

        # Train the tree with gradients and hessians (assume your tree can handle this)
        tree.train(data=data.data, targets=gradients, weights=hessians)

        # Evaluate the tree's performance on data
        new_predictions = tree.eval(data.data)

        # Update residuals for next round's gradient calculation
        if self.counter == 0:
            self.residuals = self.learningRate * new_predictions
        else:
            self.residuals += self.learningRate * new_predictions

        # Store the new predictions and increment the counter
        self.predictions[self.counter] = new_predictions
        self.counter += 1
