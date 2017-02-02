# This script contains all the types of decision making functions for
#AI poker players.

from __future__ import division
import numpy as np
import math
import os
import time
import csv
from keras.models import model_from_json
from keras.models import Sequential

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
    subFolder = ("decisionMakers/decisionMaker" + str(int(decisionRefNumber))
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
    # Turn bet conditions into game information.
    # 14 distrinct inputs are created.
    newBet = betConditions[0]
    initialNumberPlayers = int(betConditions[1])
    position = int(betConditions[2])
    roundNumber = betConditions[3]
    cardStrength = betConditions[4]
    chips = np.zeros(initialNumberPlayers)
    bets = np.zeros(initialNumberPlayers)
    raises = np.zeros(initialNumberPlayers)
    calls = np.zeros(initialNumberPlayers)
    folds = np.zeros(initialNumberPlayers)
    for j in range(0, initialNumberPlayers):
        chips[j] = betConditions[12 + j]
        bets[j] = betConditions[12 + j + initialNumberPlayers]
        raises[j] = betConditions[12 + j + (initialNumberPlayers * 2)]
        calls[j] = betConditions[12 + j + (initialNumberPlayers * 3)]
        folds[j] = betConditions[12 + j + (initialNumberPlayers * 4)]
    playersActive = initialNumberPlayers - sum(folds)

    # Calculate predictive parameters
    maxBet = np.amax(bets)
    callValue = maxBet - bets[position]
    pot = sum(bets)
    betsMade = bets[position]
    initialChipsAverage = sum(bets) + sum(chips)

    # Create normalized input layer.
    inputLayer = np.asarray([])
    if(newBet > 0):
        inputLayer = np.append(inputLayer, normalizeNormalVariable(
            math.log(newBet), 3.41, 1.25))
        inputLayer = np.append(inputLayer, 1)
    else:
        inputLayer = np.append(inputLayer, 0)
        # Append -1 to to indicate newBet is 0.
        inputLayer = np.append(inputLayer, -1)
    inputLayer = np.append(inputLayer, normalizeNormalVariable(
        cardStrength, 0.495, 0.19))
    inputLayer = np.append(inputLayer, normalizeNormalVariable(
        roundNumber, 2.09, 1.16))
    inputLayer = np.append(inputLayer, normalizeNormalVariable(
        playersActive, 4.27, 1.94))
    inputLayer = np.append(inputLayer, normalizeNormalVariable(
        initialNumberPlayers, 4.59, 1.98))
    if(callValue > 0):
        inputLayer = np.append(inputLayer, normalizeNormalVariable(
            math.log(callValue), 0.81, 0.94))
        inputLayer = np.append(inputLayer, 1)
    else:
        inputLayer = np.append(inputLayer, 0)
        # Append -1 to to indicate callValue is 0.
        inputLayer = np.append(inputLayer, -1)
    # Input is newBet a raise?
    if(newBet > callValue):
        inputLayer = np.append(inputLayer, 1)
    else:
        inputLayer = np.append(inputLayer, -1)    
    # Input is newBet a check?
    if(newBet == 0):
        inputLayer = np.append(inputLayer, 1)
    else:
        inputLayer = np.append(inputLayer, -1)
    inputLayer = np.append(inputLayer, normalizeNormalVariable(
        math.log(initialChipsAverage), 5.49, 0.72))
    inputLayer = np.append(inputLayer, normalizeNormalVariable(
        math.log(pot), 4.08, 1.80))
    if(betsMade > 0):
        inputLayer = np.append(inputLayer, normalizeNormalVariable(
            math.log(betsMade), 0.15, 1.98))
        inputLayer = np.append(inputLayer, 1)
    else:
        # Append -1 to to indicate betsMade is 0.
        inputLayer = np.append(inputLayer, 0)
        inputLayer = np.append(inputLayer, -1)
    return inputLayer

def getNNPrediction(model, inputLayer):
    # Pass inputLayer through the model and return output.
    inputLength = np.shape(inputLayer)[0]
    newInputLayer = np.reshape(inputLayer,(1,inputLength))
    prediction = model.predict(newInputLayer)
    return prediction

def getProfitMoments(categoryDistribution, categoryValues):
    # Calculate mean and variance from profit distribution.
    # Sum over profit categories to get mean and variance.
    weightedProfits = np.multiply(categoryValues, categoryDistribution)
    meanProfit = np.sum(weightedProfits)
    weightedSquaredProfits = np.multiply(categoryValues, weightedProfits)
    meanSquaredProfit = np.sum(weightedSquaredProfits)
    varianceProfit = meanSquaredProfit - (meanProfit ** 2)
    profitMoments = [meanProfit, varianceProfit]
    return profitMoments

def getCategoryMeans(refNumber, filename):
    # Open file containing limits of each category.
    currentPath = os.getcwd()
    decisionSubFolder = ("decisionMakers/decisionMaker" + str(refNumber)
    + "/" + str(filename))
    fullFilePath = os.path.join(currentPath, decisionSubFolder)
    #with open(fullFilePath, 'rb') as csvfile:
    #categoryLimits = csv.reader(open(fullFilePath))
    
    csvData = csv.reader(open(fullFilePath))
    limitsList = [float(line[0]) for line in csvData]
    categoryLimits = np.asarray(limitsList)

    # Find the mean within each category.
    numberCategories = len(categoryLimits) - 1
    categoryMeans = [0.0] * numberCategories
    for i in range(0, numberCategories):
        categoryMeans[i] = ((categoryLimits[i] + categoryLimits[i + 1]) / 2)
    return categoryMeans
 
def getBetStats(
    inputLayer, winDefeatModel, profitModel, lossModel, refNumber, foldLoss):
    # Calculate the mean and variance of the potential gain.
    # Predict chance of winning.
    winChance = getNNPrediction(winDefeatModel, inputLayer)[0][0]
    
    profitPrediction = getNNPrediction(profitModel, inputLayer)[0]
    # Normalize the outputs to sum to 1.
    profitDistribution = np.divide(profitPrediction,np.sum(profitPrediction))
    # Open player files to get values of profit categories.
    profitCategoryValues = getCategoryMeans(
        refNumber, "profitCategoryValues.csv")
    profitMoments = getProfitMoments(profitDistribution, profitCategoryValues)
   
    lossPrediction = getNNPrediction(lossModel, inputLayer)[0]
    # Normalize the outputs to sum to 1.
    lossDistribution = np.divide(lossPrediction,np.sum(lossPrediction))
    # Open player files to get values of loss categories.
    lossCategoryValues = getCategoryMeans(
        refNumber, "lossCategoryValues.csv")
    lossMoments = getProfitMoments(lossDistribution, lossCategoryValues)
 
    # Calculate bet stats.
    expectedGain = ((winChance * profitMoments[0]) +
        ((1 - winChance) * lossMoments[0]) - foldLoss)
    gainVariance = ((winChance * profitMoments[1])
        + ((1 - winChance) * lossMoments[1]))
    betStats = [expectedGain, gainVariance]
    return betStats

def optimizeBet(
    decisionRefNumber, betConditions, callValue, chipCount, existingBet,
    bigBlind, decisionModels, searchResolution = 20):
    # Assign all NN models.
    winDefeatModel = decisionModels[0]
    profitModel = decisionModels[1]
    lossModel = decisionModels[2]
    
    # Find the optimum bet to make given the bet conditions.
    if(searchResolution < 2):
        errorMessage = ("Error in AIDecisions module in optimizeBet function." +
        "searchResolution must be an integer greater than 1")
        rawInput(errorMessage)
    optimumBet = 0
    foldLoss = existingBet
    # Check if any raise can be made.
    if(chipCount <= callValue):
        # Compare calling with folding.
    	callBet = (chipCount / bigBlind)
	betConditions[0] = 0
        inputLayer = prepareFirstNNInputs(betConditions)
    	callStats = getBetStats(
            inputLayer, winDefeatModel, profitModel, lossModel,
            decisionRefNumber, foldLoss)
    	callProfit = callStats[0]
    	if(callProfit > 0):
            optimumBet = chipCount
        else:
            optimumBet = 0
    else:
	betConditions[0] = (chipCount / bigBlind)
	# Compare possible bets between callvalue and chips.
    	logCallValue = math.log(1 + callValue / bigBlind)
    	logChipCount = math.log(1 + chipCount / bigBlind)
    	logBetRange = (logChipCount - logCallValue)
    	logBetInterval = logBetRange / (searchResolution - 1)
    	optimumBetScore = 0 # Folding always has score of 0.
    	for i in range(0, searchResolution):
	    logBet = logCallValue + (logBetInterval * i)
	    bet = (2.7183 ** logBet) - 1
	    betConditions[0] = bet
	    inputLayer = prepareFirstNNInputs(betConditions)
	    betStats = getBetStats(
                inputLayer, winDefeatModel, profitModel, lossModel,
                decisionRefNumber, foldLoss)
	    meanGain = betStats[0]
	    gainVariance = betStats[1]
	    betScore = meanGain / ((gainVariance + 0.001) ** 0.5)
	    if(betScore > optimumBetScore):
                optimumBet = int(((2.7183 ** logBet) - 1) * bigBlind)
                optimumBetScore = betScore
        if(optimumBet > chipCount):
            optimumBet = chipCount
    return optimumBet

def firstNNMethodDecision(
    decisionRefNumber, position, handStrength, roundNumber, bigBlind,
    chips, bets, raises, calls, folds, decisionModels, searchResolution = 10):
    # Use predictions for win/defeat and profit/loss to select a bet.
    # Put game state with normalized bets into list.
    bigBlind = float(bigBlind)
    initialNumberPlayers = len(bets)
    betConditions = [0.0] # First value in list is reserved for new bet.
    betConditions.append(initialNumberPlayers)
    betConditions.append(position)
    betConditions.append(roundNumber)
    betConditions.append(handStrength)
    betConditions.extend(np.zeros(7)) # Hole card and Community card fillers.
    betConditions.extend(np.divide(chips, bigBlind))
    betConditions.extend(np.divide(bets, bigBlind))
    betConditions.extend(np.divide(raises, bigBlind))
    betConditions.extend(np.divide(calls, bigBlind))
    betConditions.extend(folds)

    # Search for the best bet
    maxBet = np.amax(bets)
    existingBet = bets[position]
    chipCount = chips[position]
    callValue = maxBet - existingBet
    newBet = optimizeBet(
        decisionRefNumber, betConditions, callValue, chipCount, existingBet,
        bigBlind, decisionModels)
    print("New bet by NN decision maker is")
    print(newBet)
    return newBet

def geneticNNDecision(
    decisionMakerReference, position, handStrength, roundNumber, bigBlind,
    chips, bets, raises, folds):
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
