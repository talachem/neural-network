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
        self.weights = None

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the AdaBoost model using a DecisionTree.
        """
        if self.weights is None:
            self.weights = np.ones(len(data.targets)) / len(data.targets)
        tree.train(data.data, data.targets, self.weights)
        self.updateWeights(tree, data.data, data.targets)

    def updateWeights(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Updates the data weights based on the prediction errors.
        """
        predictions = tree.eval(data.data)
        errorRate = np.sum(self.weights[data.targets != predictions]) / np.sum(self.weights)
        treeWeights = 0.5 * np.log((1 - errorRate) / errorRate)
        self.alpha.append(treeWeights)
        self.weights[data.targets == predictions] *= np.exp(-treeWeights)
        self.weights[data.targets != predictions] *= np.exp(treeWeights)
        self.weights /= np.sum(self.weights)


class GradientBoosting(Boosting):
    """
    Gradient Boosting implementation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.residuals = None  # stores the summation of all errors
        self.predictions = {}  # stores past predictions
        self.counter = 0  # counts time steps

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the GradientBoosting model using a DecisionTree.
        """
        if self.residuals is None:
            tree.train(data.data, data.targets)
        else:
            tree.train(data.data, self.residuals)

        self.predictions[self.counter] = data.targets - tree.eval(data.data)
        self.residuals = np.sum([self.predictions[key] for key in self.predictions.keys()], axis=0)
        self.counter += 1


class XGBoosting(Boosting):
    """
    Extreme Gradient Boosting (XGBoost) implementation.
    Not working yet
    """
    def __init__(self, learning_rate=0.1, regularization=1.0):
        super().__init__()
        self.residuals = None  # Summation of all errors
        self.predictions = {}  # Storing past predictions
        self.counter = 0  # Counting time steps
        self.learning_rate = learning_rate  # Learning rate for boosting
        self.regularization = regularization  # Regularization term

    def train(self, tree: DecisionTree, data: DataSet) -> None:
        """
        Trains the XGBoost model using a DecisionTree.
        """
        # Initialize residuals in the first round
        if self.residuals is None:
            self.residuals = np.zeros(data.targets.shape[0])

        # Calculate gradients and hessians
        # This is a simplification, usually you'd customize this for your loss function
        gradients = data.targets - self.residuals  # First-order gradient (derivative)
        hessians = np.ones(data.targets.shape[0])  # Second-order gradient (constant for squared error)

        # Train the tree to fit the residuals, but pass gradients and hessians for leaf optimization
        tree.train(data.data, data.gradients, hessians=hessians, regularization=self.regularization)

        # Update the residuals based on the new predictions
        new_predictions = tree.eval(data.data)
        self.residuals += self.learning_rate * new_predictions

        # Store the new predictions
        self.predictions[self.counter] = new_predictions
        self.counter += 1
