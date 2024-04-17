import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class ImpurityMeasure(ABC):
    """
    Abstract base class for impurity measures, such as Gini and Entropy.
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def __call__(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        """
        Calls the _loss method to compute the impurity measure value of the input target array.
        """
        return self._loss(targets, weights=weights, classWeights=classWeights)

    @abstractmethod
    def _loss(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        """
        Computes the impurity measure value of the input target array.
        """
        pass


class Gini(ImpurityMeasure):
    """
    A class for computing the Gini impurity measure.
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        """
        Computes the Gini impurity measure value of the input target array.
        """
        uniTargets, counts = np.unique(targets, return_counts=True) # Get unique target classes and their counts
        classWeights = np.ones(uniTargets.shape[0]) if classWeights is None else classWeights # If no classWeights provided, consider all classes equally important
        p = classWeights * counts / len(targets) # Compute the probability of each class in the target array
        return (p * (1 - p)).sum() # Compute the Gini impurity measure value


class Entropy(ImpurityMeasure):
    """
    A class for computing the Entropy impurity measure.
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        """
        Computes the Entropy impurity measure value of the input target array.
        """
        uniTargets, counts = np.unique(targets, return_counts=True) # Get unique target classes and their counts
        classWeights = np.ones(uniTargets.shape[0]) if classWeights is None else classWeights # If no classWeights provided, consider all classes equally important
        p = classWeights * counts / len(targets) # Compute the probability of each class in the target array
        return - (p * np.log2(p)).sum() # Compute the Entropy


class MAE(ImpurityMeasure):
    """
    Computes the mean absolute error (MAE) of the given targets.
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        if len(targets) == 0:
            return 0.0
        targetsMean = np.mean(targets)
        return np.sum(np.abs(targets - targetsMean)) / len(targets)


class MSE(ImpurityMeasure):
    """
    Computes the mean squared error (MSE) of the given targets.
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        if len(targets) == 0:
            return 0.0
        targetsMean = np.mean(targets)
        return np.sum((targets - targetsMean) ** 2) / len(targets)


class ODD(ImpurityMeasure):
    """
    Used for anomaly detection, it's stubb class
    """
    def __init__(self) -> None:
        super().__init__()

    def _loss(self, targets: NDArray, *, weights: NDArray| None = None, classWeights: NDArray| None = None) -> float:
        return np.random.random() # return some default value
