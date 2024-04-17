import numpy as np
import sys, random
from machineLearning.som import (
    SOM,
    Rectangular, Hexagonal,
    GuassianNeighborhood, BubbleNeighborhood, MexicanHatNeighborhood, LinearNeighborhood, CosineNeighborhood, CauchyNeighborhood, EpanechnikovNeighborhood
)
from machineLearning.data import Data
from matplotlib import pyplot as plt
from matplotlib import cm
from machineLearning.utility import Time
from machineLearning.settings import SOMSettings
from machineLearning.nn.scheduler import ExponentialLR, SteppedLR


def dataShift(dims):
    offSet = [1, 0.5, 1]
    diffLen = abs(len(offSet) - dims)
    offSet.extend([0] * diffLen)
    random.shuffle(offSet)
    return offSet[:dims]


# Helper function for plotting dummy data
def scatterPairwise(data, weights, size: float = 10, colors: list[str, str] = ['tab:blue', 'tab:orange']):
    """
    Create a scatter plot of pairwise dimensions of a multidimensional dataset on a grid.

    Parameters:
    data (ndarray): The multidimensional dataset to be plotted.
    size (float): The size of each scatter point in the plot (default 10).
    color (str): The color of each scatter point in the plot (default 'blue').

    Returns:
    None.
    """
    num_dims = data.shape[1]
    fig, axes = plt.subplots(num_dims, num_dims, figsize=(12, 12))

    for i in range(num_dims):
        for j in range(num_dims):
            if i == j:
                axes[i][j].axis('off')
            else:
                axes[i][j].scatter(data[:, i], data[:, j], s=size, c=colors[0], alpha=0.5,label='data')
                axes[i][j].scatter(weights[:, i], weights[:, j], s=1.5*size, c=colors[1], alpha=1,label='weights')
                axes[i][j].set_xlabel(f"Dim {i}")
                axes[i][j].set_ylabel(f"Dim {j}")
                axes[i,j].legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()


def map(values: np.ndarray, arange: list = [0,1]) -> np.ndarray:
    assert len(arange) == 2, 'arange must be of length 2'
    assert arange[0] < arange[1], 'arange must start at a lower value than it ends'

    c, d = arange[0], arange[1]
    a, b = np.min(values), np.max(values)

    return c + ((d - c) / (b - a)) * (values - a)

def plotMatrix(matrix, grid, shape='o', size=1):
    plt.figure(figsize=(12,12))  # specify the figure size in inches
    fig, ax = plt.subplots()

    # create an array of points for the grid
    points = grid

    # plot each point
    values = map(matrix.flatten(), [0,255]).astype('int')
    colors = cm.viridis(values)


    ax.scatter(points[:, 0], points[:, 1], color=colors, s=250*size, marker=shape)

    ax.set_title('Weight Matrix')

    # Increase the size of the plot (or 'zoom out')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, points[:,0].max()+1)
    ax.set_ylim(-1, points[:,1].max()+1)

    plt.show()


def pickNeighborhood(neighborhood: str, scale: float):
    if neighborhood == 'gaussian':
        return GuassianNeighborhood(scale)
    elif neighborhood == 'mexicanhat':
        return MexicanHatNeighborhood(scale)
    elif neighborhood == 'bubble':
        return BubbleNeighborhood(scale)
    elif neighborhood == 'linear':
        return LinearNeighborhood(scale)
    elif neighborhood == 'cosine':
        return CosineNeighborhood(scale)
    elif neighborhood == 'cauchy':
        return CauchyNeighborhood(scale)
    elif neighborhood == 'epanechnikov':
        return EpanechnikovNeighborhood(scale)
    else:
        raise ValueError(f"{neighborhood} is not an option")


def pickTopology(topology: str, gridSize: tuple, numFeatures: int):
    if topology == 'rectangular':
        return Rectangular(gridSize, numFeatures)
    elif topology == 'hexagonal':
        return Hexagonal(gridSize, numFeatures)
    else:
        raise ValueError(f"{topology} is not an option")


def plotPies(grid, title, countSet: int = 0):
    fig, ax = plt.subplots(*grid.gridSize, figsize=(grid.gridSize[1], grid.gridSize[0]))

    # Adjust the subplot parameters to reduce the space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Set aspect ratio of all subplots to be equal so that the pie charts look like circles, not ellipses
    for a in ax.ravel():
        a.set_aspect('equal')

    for index, count in enumerate(grid.counts[countSet]):
        xx, yy = np.unravel_index(index, grid.gridSize)
        if np.sum(count) > 0:  # check if there are counts for this neuron
            ax[xx,yy].pie(count)
        else:  # if no counts, you can leave it blank or put something else here
            ax[xx,yy].axis('off')

    plt.suptitle(title)
    plt.show()


def plotBars(grid, countSets = [0, 1]):
    fig, ax = plt.subplots(*grid.gridSize, figsize=(grid.gridSize[1], grid.gridSize[0]))

    # Adjust the subplot parameters to reduce the space between subplots
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Set aspect ratio of all subplots to be equal so that the pie charts look like circles, not ellipses
    for a in ax.ravel():
        a.set_aspect('equal')

    backgroundColor = grid._umatrix.reshape(*grid.gridSize)
    minValue = backgroundColor.min()
    backgroundColor = backgroundColor - minValue
    maxValue = backgroundColor.max()
    backgroundColor = backgroundColor/maxValue
    counts0 = grid.counts[countSets[0]] / np.sum(grid.counts[countSets[0]], axis=1).reshape(grid.numNeurons,1)
    counts1 = grid.counts[countSets[1]] / np.sum(grid.counts[countSets[1]], axis=1).reshape(grid.numNeurons,1)
    counts = counts0 - counts1

    for index, count in enumerate(counts):
        xx, yy = np.unravel_index(index, grid.gridSize)
        ax[xx,yy].set_ylim(-1,1)
        ax[xx,yy].set_facecolor(f'{backgroundColor[xx,yy]}')
        ax[xx,yy].bar(np.arange(len(count)), count)

    plt.suptitle("Count Deltas")
    plt.show()

if __name__ == "__main__":
    settings = SOMSettings()
    try:
        configFile = sys.argv[1]
        settings.getConfig(configFile)
        settings.setConfig()
    except IndexError:
        pass
    print(settings)

    # Initialize a timer to measure the runtime of different parts of the code
    timer = Time()

    print("Importing data...\n")
    timer.start()
    data = Data(trainAmount=settings['trainAmount'], evalAmount=settings['validAmount'], batchSize=settings['batchSize'], dataPath=settings['dataPath'], normalize=settings['normalize'])
    data.inputFeatures(*settings['features'])
    data.importData(*settings['dataFiles'])
    print(data)
    timer.record("Importing Data")

    # Create and initialize the Self-Organizing Map (SOM)
    timer.start()
    grid = SOM(settings['learningRate'], settings['gridSteps'], settings['decreaseEvery'])

    grid.setComponent(pickTopology(settings['topology'], settings['gridSize'], data.trainSet.shape[-1]))
    grid.setComponent(pickNeighborhood(settings['neighborhood'], settings['scale']))
    if settings['scheduler'] == 'exponential':
        grid.setComponent(ExponentialLR(grid, settings['decayrate']))
    elif settings['scheduler'] == 'stepped':
        grid.setComponent(SteppedLR(grid, settings['decayrate'], settings['stepSize']))
    timer.record("SOM setup")


    # Train the SOM using the test data
    timer.start()
    print('beginn training...')
    grid.train(data.train, settings['epochs'])
    timer.record("Training")

    timer.start()
    print('beginn evaluation...')
    grid.eval(data.train)
    grid.eval(data.eval)
    timer.record("Evaluation")

    # Visualize the SOM and U-Matrix
    if grid.topology.numFeatures == 2:
        # If the data has only two dimensions, create a simple scatter plot
        plt.scatter(data.trainSet.data[:,0],data.trainSet.data[:,1],label='data')
        plt.scatter(grid.weights[:,0],grid.weights[:,1],label='weights')
        plt.legend()
        plt.show()
    else:
        # If the data has more than two dimensions, create a scatter plot of pairwise dimensions
        scatterPairwise(data.trainSet.data, grid.weights)

    if settings['topology'] == 'hexagonal':
        plotMatrix(grid.weightMatrix, grid.topology.gridIndices,'H',1.5)
        plotMatrix(grid.uMatrix, grid.topology.gridIndices,'H',1.5)
    else:
        plotMatrix(grid.weightMatrix, grid.topology.gridIndices,'s')
        plotMatrix(grid.uMatrix, grid.topology.gridIndices,'s')

    #plotPies(grid, "Train Seit", 0)
    #plotPies(grid, "Eval Set", 1)
    #plotBars(grid)

    # Print the total runtime of the code
    print()
    print(timer)