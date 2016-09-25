# This script contains all the types of decision making functions for
#AI poker players.

from __future__ import division
import numpy as np
import math
import os

def simpleAIBet(bigBlind, handStrength, bets, position):
    # Simple betting scheme based on handStrength only
    maxBet = np.amax(bets)
    callValue = maxBet - bets[position]
    if(handStrength > 0.5):
        newBet = callValue + bigBlind
    elif(handStrength > 0.3):
        newBet = callValue
    else:
        newBet = 0
    return newBet

def openWeigthsTSVFile(filePath, layerSizes):
    numberLayers = len(layerSizes)
    maxLayerSize = np.amax(layerSizes)
    weights = np.zeros((numberLayers - 1, maxLayerSize, maxLayerSize))
    # Put all weights from the file into a 1D list
    weightsList = []
    with open(filePath,"r") as weightsfile:
        for line in weightsfile:
            for number in line.split("\t"):
                if(number != "\r\n"):
                    weightsList.append(number)
    # Transfer weights from 1D list to 3D list
    k = 0
    for layerCount in range(0, (numberLayers - 1)):
        for i in range(0, layerSizes[layerCount]):
            for j in range(0, layerSizes[layerCount + 1]):
                weights[layerCount][i][j] = float(weightsList[k])
                k += 1
    return weights

def getGeneticWeights(decisionRefNumber, layerSizes):
    currentPath = os.getcwd()
    subFolder = ("decisionMakers/decisionMaker" + str(decisionRefNumber)
    + "/geneticNNWeights.txt")
    fullpath = os.path.join(currentPath, subFolder)
    weights = openWeigthsTSVFile(fullpath, layerSizes)
    return weights

def oneLayerFeedForward(currentLayer, nextLayerSize, weights, activation):
    # Pass values through a single layer of the neural network and
    #return the outputs to the next layer.
    # weights should be given with bias multipliers in the first column.
    nextLayer = [0] * nextLayerSize
    # Compute sum of current Layer multiplied by weights.
    nextLayer = np.dot(currentLayer, weights)
    # Apply the activation function.
    if(activation == "relu"):
        for i in range(0, nextLayerSize):
            if(nextLayer[i] < 0):
                nextLayer[i] = 0
    elif(activation == "sigmoid"):
        for i in range(0, nextLayerSize):
            nextLayer[i] = 1 / (1 + (2.718282 ** (-1 * nextLayer[i])))
    elif(activation == "none"):
        # Do nothing.
        temp = 0
    else:
        print "Not recognized: Activation function '%s'" % activation
    return nextLayer

def geneticNNFeedForward(layerSizes, inputLayer, allWeights, activations):
    # Pass input values through all layers.
    # inputLayer includes bias input.
    # layerSizes include bias inputs.
    numberLayers = len(layerSizes)
    nextLayer = np.array([])
    # Copy inputLayer into currentLayer.
    currentLayer = np.array(inputLayer)
    # Repeatedly pass the current layer through each layer of the NN.
    for layerIndex in range(0,numberLayers - 1):
        # Put bias input of 1 at the start of the current layer.
        currentLayer[0] = 1
        # Set up weights for this feed forward layer.
        weightsTemp = allWeights[layerIndex][:][:]
        weights = np.copy(weightsTemp)
        weights.resize(layerSizes[layerIndex],layerSizes[layerIndex + 1])
        activation = activations[layerIndex]
        nextLayerSize = layerSizes[layerIndex + 1]
        nextLayer = oneLayerFeedForward(
            currentLayer, nextLayerSize, weights, activation)
        # Copy nextLayer to the new currentLayer.
        currentLayer = np.copy(nextLayer)
    # Return the output layer.
    return currentLayer

def normalizeUniformVariable(variable, Xmean, Xrange):
    # Normalize a unifrom distributed variable according to its mean and
    #range.
    normalizedVariable = 2 * ((variable - Xmean) / Xrange)
    return normalizedVariable

def normalizeNormalVariable(variable, mean, stDev):
    # Normalize a normally distributed variable accoring to its mean
    #and standard deviation.
    normalizedVariable = ((variable - mean) / stDev)
    return normalizedVariable

def prepareGeneticInputs(
    pot, handStrength, callValue, bigBlind, existingBet, roundNumber,
    playersActive, initialNumberPlayers, folds, raises, initialChipsAverage):
    # Transform/normalizing is irregular.
    inputLayer = [0] * 12
    inputLayer[0] = 1 # Bias input.
    inputLayer[1] = normalizeUniformVariable(
        math.log(pot / bigBlind), 3.190, 5.583)
    inputLayer[2] = normalizeUniformVariable(handStrength, 0.475, 0.45)
    if(callValue == 0):
        inputLayer[3] = -3.0
    else:
        inputLayer[3] = normalizeUniformVariable(
            math.log(callValue / bigBlind), 2.284, 4.549)
    if(existingBet == 0):
        inputLayer[4] = -1.5
    else:
        inputLayer[4] = normalizeUniformVariable(
            math.log(1 + (existingBet / bigBlind)), 2.479, 4.842)
    inputLayer[5] = normalizeUniformVariable(roundNumber, 2.5, 3.0)
    inputLayer[6] = normalizeNormalVariable(playersActive, 4.229, 1.640)
    inputLayer[7] = normalizeUniformVariable(initialNumberPlayers, 5.0, 6.0)
    # Sort raises to set as input variables.
    orderedRaises = [0] * initialNumberPlayers
    for i in range(0, initialNumberPlayers):
        if(not folds[i]):
            orderedRaises[i] = raises[i]
        else:
            orderedRaises[i] = 0
    orderedRaises.sort(reverse = True)
    inputLayer[8] = normalizeUniformVariable(
        math.log(1 + (orderedRaises[0] / bigBlind)), 6.54 , 5.92)
    inputLayer[9] = 0 # Equal to zero for historical reasons.
    inputLayer[10] = 0 # Equal to zero for historical reasons.
    inputLayer[11] = normalizeUniformVariable(
        math.log(initialChipsAverage / bigBlind), 3.851 , 2.905)
    return inputLayer

def prepareFirstNNInputs(betConditions):
    # Take betting conditions and return inputs.
    # betConditions contains new bet as first entry and the game state.
    # Turn bet conditions into game information
    newBet = betConditions[0]
    initialNumberPlayers = betConditions[1]
    position = betConditions[2]
    cardStrength = betConditions[3]
    bets = np.zeros(initialNumberPlayers)
    calls = np.zeros(initialNumberPlayers)
    raises = np.zeros(initialNumberPlayers)
    chips = np.zeros(initialNumberPlayers)
    folds = np.zeros(initialNumberPlayers)
    for i in range(0, initialNumberPlayers):
        chips[j] = betConditions[11 + j]
        bets[j] = betConditions[11 + (2 * j)]
        raises[j] = betConditions[11 + (3 * j)]
        calls[j] = betConditions[11 + (4 * j)]
        folds[j] = betConditions[11 + (5 * j)]
    # Calculate predictive parameters
    maxBet = np.amax(bets)
    callValue = maxBet - bets[position]
    pot = sum(bets)
    betsMade = bets[position]
    initialChipsAverage = sum(bets) + sum(chips)
    inputLayer = np.asarray([])
    if(newBet > 0):
        inputLayer.append(normalizeUniformVariable(
            math.log(newBet), 0, 1))
        np.append(inputLayer, 1)
    else:
        # Append -1 to to indicate newBet is 0.
        np.append(inputLayer, 0)
        np.append(inputLayer, -1)
    np.append(inputLayer, normalizeUniformVariable(
        handStrength, 0.475, 0.45))
    np.append(inputLayer, normalizeUniformVariable(roundNumber, 2.5, 3.0))
    np.append(inputLayer, normalizeNormalVariable(
        playersActive, 4.229, 1.640))
    np.append(inputLayer, normalizeUniformVariable(
        initialNumberPlayers, 5.0, 6.0))
    if(callValue > 0):
        np.append(inputLayer, normalizeUniformVariable(
            math.log(callValue), 2.284, 4.549))
        np.append(inputLayer, inputLayer, 1)
    else:
        # Append -1 to to indicate callValue is 0.
        np.append(inputLayer, 0)
        np.append(inputLayer, -1)    
    # Input is newBet a raise?
    if(newBet > callValue):
        np.append(inputLayer, 1)
    else:
        np.append(inputLayer, -1)    
    # Input is newBet a check?
    if(newBet == 0):
        inp.append(inputLayer, 1)
    else:
        np.append(inputLayer, -1)
    np.append(inputLayer, normalizeUniformVariable(
        math.log(initialChipsAverage), 3.851 , 2.905))
    np.append(inputLayer, normalizeUniformVariable(
        math.log(pot), 0 , 1))
    if(betsMade > 0):
        np.append(inputLayer, normalizeUniformVariable(
            math.log(betsMade), 0, 1))
        np.append(inputLayer, 1)
    else:
        # Append -1 to to indicate betsMade is 0.
        np.append(inputLayer, 0)
        np.append(inputLayer, -1)

def geneticNNDecision(
    decisionMakerReference, position, handStrength, roundNumber, bigBlind,
    folds, chips, bets, raises):
    activations = ["sigmoid","sigmoid","sigmoid"]
    pot = sum(bets)
    maxBet = np.amax(bets)
    initialNumberPlayers = len(bets)
    existingBet = bets[position]
    callValue = maxBet - bets[position]
    initialChipsAverage = (sum(chips) + pot) / initialNumberPlayers
    playersActive = initialNumberPlayers - sum(folds)
    # Layer sizes do not vary for different geneticNNDecision players.
    # Layer sizes exclude bias input.
    layerSizes = [12, 9, 3, 1]
    # Read weights from file.
    weights = getGeneticWeights(decisionMakerReference, layerSizes)
    # Final layer is not used in NN.
    numberLayers = len(layerSizes) - 1
    # Put game states into input. Transform and normalize the inputs.
    inputLayer = prepareGeneticInputs(
        pot, handStrength, callValue, bigBlind, existingBet, roundNumber,
        playersActive, initialNumberPlayers, folds, raises, initialChipsAverage)
    # Put inputs through NN.
    # Final layer weights are not used for NN.
    layerSizesTemp = layerSizes[0:(len(layerSizes) - 1)]
    outputs = geneticNNFeedForward(
        layerSizesTemp, inputLayer, weights, activations)
    # Use outputs to decide on bet.
    if(outputs[0] < 0.5):
        # Check/fold.
        newBet = 0
    elif(outputs[1] < 0.5):
        # Call.
        newBet = callValue
    else:
        # Raise.
        newBet = callValue + (outputs[2] * initialChipsAverage)
    return newBet
