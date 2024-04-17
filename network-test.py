import numpy as np
import sys
from matplotlib import pyplot as plt
from machineLearning.nn.layer import (
    Linear, Dropout, Flatten, Convolution2D, Unsqueeze,
    Tanh, SoftMax, Sigmoid, SoftPlus, Relu, Elu, LeakyRelu, SoftSign, Identity,
    BatchNorm1D, BatchNorm2D
)
from machineLearning.nn.optim import SGD, SGDMomentum, NesterovMomentum, AdaGrad, AdaDelta, RMSprop, Adam
from machineLearning.nn.scheduler import ExponentialLR, SteppedLR, CyclicalLR
from machineLearning.nn.module import Sequential, Parallel
from machineLearning.nn.loss import CrossEntropyLoss, MSELoss, NLLLoss, MAELoss, FocalLoss
from machineLearning.data import Data
from machineLearning.utility import Time, Progressbar
from machineLearning.metric import ConfusionMatrix, Observables
from machineLearning.settings import NetworkSettings


def actiPicker(acti: str):
    if acti == 'tanh':
        return Tanh()
    elif acti == 'elu':
        return Elu()
    elif acti == 'relu':
        return Relu()
    elif acti == 'lrelu':
        return LeakyRelu()
    elif acti == 'sigmoid':
        return Sigmoid()
    elif acti == 'id':
        return Identity()
    elif acti == 'softplus':
        return SoftPlus()
    elif acti == 'softmax':
        return SoftMax()
    elif acti == 'softsign':
        return SoftSign()


if __name__ == "__main__":
    settings = NetworkSettings()
    try:
        configFile = sys.argv[1]
        settings.getConfig(configFile)
        settings.setConfig()
    except IndexError:
        pass
    print(settings)

    # Initialize a timer to measure the runtime of different parts of the code
    timer = Time()

    # Loading some test data
    print("Importing data...\n")
    timer.start()
    data = Data(trainAmount=settings['trainAmount'], evalAmount=settings['validAmount'], batchSize=settings['batchSize'], kFold=settings['kFold'], dataPath=settings['dataPath'], normalize=settings['normalize'])
    #data.generateTestData(settings['numCategories'])
    data.inputFeatures(*settings['features'])
    data.importData(*settings['dataFiles'])
    print(data)
    timer.record("Importing Data")

    # Create and initialize the Network
    print("Setting up network")
    timer.start()
    # Configuring number of Neurons

    # Setting up Convolution
    network = Sequential()
    if len(settings['convolutions']) > 0:
        neurons = 0
        convolutions = Parallel()
        for key in settings['convolutions']:
            conv = Sequential()
            kernels = settings['kernelSize'][key]
            channels = settings['channels'][key]
            activations = settings['activationConv'][key]
            norms = settings['convNorm'][key]
            xSize, ySize = 9, 9
            for i, (inChan, outChan, kern, acti, norm) in enumerate(zip(channels, channels[1:], kernels, activations, norms)):
                if i == 0:
                    conv.append(Unsqueeze((inChan, xSize, ySize)))
                xSize = int((xSize - kern[0])/1 + 1)
                ySize = int((ySize - kern[0])/1 + 1)
                conv.append(Convolution2D(inChan, outChan, kern))
                conv.append(actiPicker(acti))
                if norm is True:
                    conv.append(BatchNorm2D((outChan, xSize, ySize)))
            neurons += (xSize * ySize * outChan)
            convolutions.append(conv)
        network.append(convolutions)
    else:
        neurons = settings['numNeurons'][0]

    # Adding Linear Layer
    network.append(Flatten())
    # this code is a bit ugly and I am not sure it will work under all conditions
    for i, (inFeat, outFeat, drop, acti, norm) in enumerate(zip(settings['numNeurons'], settings['numNeurons'][1:], settings['dropoutRate'], settings['activationLin'], settings['linearNorm'])):
        if i == 0:
            network.append(Linear(neurons,outFeat))
            network.append(Dropout(outFeat,drop))
            network.append(actiPicker(acti))
            if norm is True:
                network.append(BatchNorm1D(outFeat))
        elif i == settings['numLayers'] - 1:
            network.append(Linear(inFeat,settings['numCategories']))
            network.append(SoftMax())
        else:
            network.append(Linear(inFeat,outFeat))
            network.append(Dropout(outFeat,drop))
            network.append(actiPicker(acti))
            if norm is True:
                network.append(BatchNorm1D(outFeat))

    print(network)
    timer.record('Network setup')

    # Setting up loss func
    print("Setting up loss/optimizer")
    timer.start()
    if settings['lossFunction'] == 'entropy':
        lossFunc = CrossEntropyLoss()
    elif settings['lossFunction'] == 'nllloss':
        lossFunc = NLLLoss()
    elif settings['lossFunction'] == 'focalloss':
        lossFunc = FocalLoss()
    elif settings['lossFunction'] == 'mseloss':
        lossFunc = MSELoss()
    elif settings['lossFunction'] == 'maeloss':
        lossFunc = MAELoss()

    # Setting up opimizer
    if settings['optim'] == 'sgd':
        optim = SGD(network, settings['learningRate'])
    elif settings['optim'] == 'momentum':
        optim = SGDMomentum(network, settings['learningRate'], settings['momentum'])
    elif settings['optim'] == 'nesterov':
        optim = NesterovMomentum(network, settings['learningRate'], settings['momentum'])
    elif settings['optim'] == 'adagrad':
        optim = AdaGrad(network, settings['learningRate'])
    elif settings['optim'] == 'adadelta':
        optim = AdaDelta(network, settings['learningRate'])
    elif settings['optim'] == 'rmsprop':
        optim = RMSprop(network, settings['learningRate'])
    elif settings['optim'] == 'adam':
        optim = Adam(network, settings['learningRate'])

    # LR scheduler
    scheduler = None
    if settings['scheduler'] == 'exponential':
        scheduler = ExponentialLR(optim, settings['decayrate'])
    elif settings['scheduler'] == 'stepped':
        scheduler = SteppedLR(optim, settings['decayrate'], settings['stepSize'])
    elif settings['scheduler'] == 'else':
        scheduler = CyclicalLR(optim, 1/5, 15, 5)
    timer.record('Network setup')
    print(optim)
    print(lossFunc)

    # setting up training observables
    metrics = Observables(settings['epochs'])
    metrics.addOberservable('losses', 'descending')
    metrics.addOberservable('validation', 'descending')
    metrics.addOberservable('accuracy', 'ascending')
    metrics.addOberservable('learningRate')

    # Beginn training
    print("Beginn training...")
    timer.start()
    epochs = settings['epochs']
    for i in range(settings['epochs']):
        data.trainMode()
        network.train()
        length = len(data.train)
        bar = Progressbar(f'epoch {str(i+1).zfill(len(str(epochs)))}/{epochs}', length, 55)
        losses = []
        for item in data.train:
            inputs = item['data']
            labels = item['labels']
            prediction = network(inputs)
            losses.append(lossFunc(prediction, labels))
            gradient = lossFunc.backward()
            optim.step(gradient)
            bar.step()
        data.evalMode()
        network.eval()
        accuracies = []
        valLosses = []
        for item in data.train:
            inputs = item['data']
            labels = item['labels']
            prediction = network(inputs)
            valLosses.append(lossFunc(prediction, labels))
            accuracies.append(np.sum(prediction.argmax(1) == labels.argmax(1)) / len(prediction))
            bar.step()
        if scheduler is not None:
            scheduler.step()
        metrics.update('losses', np.mean(losses))
        metrics.update('validation', np.mean(valLosses))
        metrics.update('accuracy', np.mean(accuracies))
        metrics.update('learningRate', optim.learningRate)
        metrics.print()
        metrics.step()
        data.fold()
        #bar.finish()
    timer.record('Training')

    # plotting training metrics
    fig, ax = plt.subplots()
    ax3 = ax.twinx()
    lns1 = ax.plot(metrics.losses.values, label='Train Loss')
    lns2 = ax.plot(metrics.validation.values, label='Eval Loss')
    lns3 = ax.plot(metrics.accuracy.values, label='Accuracy')
    lns4 = ax3.plot(metrics.learningRate.values, label='learning rate', color='tab:gray', ls='--')

    lns = lns1+lns2+lns3+lns4
    labs = [lab.get_label() for lab in lns]
    ax.legend(lns, labs)

    ax.grid(ls=':')
    plt.show()

    # evaluating on test data
    print("\nMaking predictions...")
    timer.start()
    confusion = ConfusionMatrix(settings['numCategories'])
    network.eval()
    length = len(data.eval)
    bar = Progressbar('evaluation', length)
    for item in data.eval:
        inputs = item['data']
        labels = item['labels']
        prediction = network(inputs)
        confusion.update(prediction, labels)
        bar.step()

    # Calculate and print confusion matrix
    confusion.percentages()
    confusion.calcScores()
    timer.record("Prediction")
    print()
    print(confusion)
    print()

    # Print total execution time
    print(timer)