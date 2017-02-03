# This script plays various decision makers against each other to gather game
#data and test perfomance.

from __future__ import division
import numpy as np
import random
import os
import PokerGames
import math
import copy
import csv
from scipy.stats import norm
from keras.models import model_from_json
from sklearn.externals import joblib

random.seed()

def selectRandomPlayers(
    numberPlayers, refNumbers, essentialPlayerRefs = False,
    distribution = "uniform"):
    # A random selection of players is made by choosing players in the
    #refNumbers list with a distribution either "uniform" or "triangle".
    # All players in the list essentialPlayerRefs are automatically selected.
    playerRefs = []
    existingPlayers = []
    if((essentialPlayerRefs is not False) and (essentialPlayerRefs != [])):
        # Include players in essentialPlayerRefs.
        for i in range(len(essentialPlayerRefs)):
            playerRefs.append(essentialPlayerRefs[i])
            existingPlayers.append(essentialPlayerRefs[i])
    # Sample new players from the range of ref numbers.
    while(len(playerRefs) < numberPlayers):
        if(distribution == "uniform"):
            newRefIndex = np.random.randint(0, len(refNumbers))
            newRef = refNumbers[newRefIndex]
        elif(distribution == "triangle"):
            # Use triangle distribution to make later refs more likely.
            lower = 0
            mode = len(refNumbers) + 1
            upper = len(refNumbers) + 1
            newRefFloat = np.random.triangular(lower, mode, upper)
            newRefIndex = int(newRefFloat)
            if(newRef == len(refNumbers) + 1):
                newRefIndex = len(refNumbers)
            newRef = refNumbers[newRefIndex]
        # Check if ref has already been selected
        if(newRef not in existingPlayers):
            playerRefs.append(newRef)
            existingPlayers.append(newRef)
    # Shuffle playerRefs so that dealerPosition = 0 doesn't create bias
    random.shuffle(playerRefs)
    return playerRefs

def getFileNames(decisionRefs):
    # Populate a 2D list with all the file names required for a
    #deicision maker.
    # maxNumberFiles is the most number of files which an individual
    #decision maker has.
    maxNumberFiles = 3
    numberPlayers = len(decisionRefs)
    decisionFiles = np.array([''], dtype=object)
    decisionFiles.resize((numberPlayers, maxNumberFiles))
    # Find current working directory.
    currentPath = os.getcwd()
    for i in range(numberPlayers):
        decisionRefNumber = decisionRefs[i]
        # Find filepath to decisionFiles.txt.
        subFolder = ("decisionMakers/decisionMaker"
            + str(int(decisionRefNumber)) + "/decisionFiles.txt")
        filePath = os.path.join(currentPath, subFolder)
        # Open and read files
        with open(filePath, "r") as allNamesFile:
            # Separate all lines from the file.
            content = allNamesFile.readlines()
            # Remove the linebreaks at the end of each line.
            content = [line.strip("\r\n") for line in content]
        # Add new files to player's list.
        if(len(content) > maxNumberFiles):
            decisionFiles.resize((numberPlayers, maxNumberFiles))
        for j in range(len(content)):
            decisionFiles[i][j] = content[j]
    return decisionFiles

def randomChips(bigBlind, minChips, maxChips, initialNumberPlayers):
    initialChips = [0] * initialNumberPlayers
    for i in range(initialNumberPlayers):
        randNumber = np.random.random()
        logMin = math.log(minChips)
        logMax = math.log(maxChips)
        logRandomNumber = logMin + (randNumber * (logMax - logMin))
        chipValue = bigBlind * (2.71828 ** logRandomNumber)
        initialChips[i] = int(chipValue)
    return initialChips

def playRandomHand(
    decisionRefs, playerModels, bigBlind=100, minChips=10, maxChips=200,
    recordBets=False):
    # Players start with a random amount of chips to simulate a random
    #point in a tournament.
    initialNumberPlayers = len(decisionRefs)
    # Set initial chips for the players.
    startChips = randomChips(
        bigBlind, minChips, maxChips, initialNumberPlayers)
    initialChips = np.copy(startChips)
    # Set players' names as their ref numbers.
    playerNames = []
    for i in range(initialNumberPlayers):
        playerNames.append(str(decisionRefs[i]))
    # Prepare values for playHand function.
    AIPlayers = [True] * initialNumberPlayers
    fileNames = getFileNames(decisionRefs)
    # Play one hand.
    finalChips = PokerGames.playHand(playerNames, initialChips, bigBlind,
                                     playerModels, AIPlayers=AIPlayers,
                                     decisionRefs=decisionRefs,
                                     fileNames=fileNames,
                                     recordBets=recordBets)
    # Convert the chips list into a numpy array.
    np.asarray(finalChips)
    profits = np.subtract(finalChips, initialChips)
    return profits

def getMean(dataList):
    # Return the mean of a list of numerical data.
    sumData = sum(dataList)
    count = len(dataList)
    if(count == 0):
        return False
    else:
        mean = sumData/count
    return mean

def getVariance(dataList, mean):
    # Return the variance of a list of numerical data.
    sumData = sum(dataList)
    count = len(dataList)
    # Get sum of squares of entries.
    sumSq = 0
    if mean is not False:
        for i in range(count):
            sumSq += (dataList[i] ** 2)
        if(count <= 1):
            return False
        else:
            variance = ((sumSq/count)-(mean**2)) * (count / (count-1))
    else:
        return False
    return variance

def getAbsoluteThirdCentralMoment(dataList, mean):
    # Return the third absolute moment about the mean of the data.
    count = len(dataList)
    sumMoment = 0
    if mean is not False:
        for i in range(count):
            value = dataList[i]
            sumMoment += (abs(value - mean) ** 3)
        if(count <= 2):
            return False
        else:
            thirdMoment = (sumMoment / count)
    else:
        return False
    return thirdMoment

def getConfidenceInterval(sampleMean, variance, samples, alpha=0.05):
    # Find a two tailed confidence interval of the mean using the
    #central limit theorem.
    # Find Z score for alpha
    ZScore = norm.ppf(1-(alpha/2))
    if variance is not False:
        lowerLimit = sampleMean - ZScore * ((variance/samples)**0.5)
        upperLimit = sampleMean + ZScore * ((variance/samples)**0.5)
        interval = [lowerLimit, upperLimit]
    else:
        return False
    return interval

def conservativeConfidenceInterval(
    sampleMean, variance, thirdMoment, samples, alpha=0.05):
    # Use the Berry-Esseen theorem to find a conservative confidence
    #interval for the mean.
    if thirdMoment is False:
        return False
    else:
        stDev = variance**0.5
        if(stDev == 0.0):
            return False
        berryEsseenBoundNumerator = 0.33554*(thirdMoment + 0.415*(stDev**3.0))
        berryEsseenBoundDenominator = (stDev**3.0)*(samples ** 0.5)
        berryEsseenBound = berryEsseenBoundNumerator/berryEsseenBoundDenominator
        # Find the quantile of the conservative interval.
        # If Berry-Esseen bound is too large then interval is infinite.
        quantile = 1 - (alpha/2) + berryEsseenBound
        if(quantile >= 1):
            interval = False
        else:
            ZScore = norm.ppf(quantile)
            lowerLimit = sampleMean - (ZScore * ((variance/samples)**0.5))
            upperLimit = sampleMean + (ZScore * ((variance/samples)**0.5))
            interval = [lowerLimit, upperLimit]
    return interval

def conservativePValue(sampleMean, variance, thirdMoment, samples):
    # Use the Berry-Esseen theorem to find a conservative p value for
    #the hypothesis that the mean is above/below 0.
    if thirdMoment is False:
        return False
    else:
        # Find the Berry-Esseen bound.
        stDev = variance**0.5
        if(stDev == 0.0):
            return False
        berryEsseenBoundNumerator = 0.33554 * (thirdMoment + 0.415*(stDev**3.0))
        berryEsseenBoundDenominator = (stDev**3.0)*(samples ** 0.5)
        berryEsseenBound = berryEsseenBoundNumerator/berryEsseenBoundDenominator
        # Find the sample mean Z score above 0.
        ZScore = (samples**0.5)*abs(sampleMean)/stDev
        # Find quantile for the ZScore and add the bound to be conservative.
        quantile = norm.cdf(ZScore)
        pValue = 1 - quantile + berryEsseenBound
        # Adjust p value to 1 if bound is too large.
        if(pValue >= 1):
            pValue = 1
    return pValue

def updateRefStats(refStats, playerRefs, profitRecords, confidenceAlpha=0.05):
    # For each player's profit record recalculate summary statistics.
    refNumbers = refStats[:,0]
    tournamentSize = len(refNumbers)
    numberPlayers = len(playerRefs)
    # For all players who played most recent game increase samples by 1.
    for player in range(numberPlayers):
        refIndex = np.where(refNumbers == playerRefs[player])[0][0]
        refStats[refIndex][1] += 1
    # Calculate summary statistics for every tournament player.
    for player in range(tournamentSize):
        profitHistory = profitRecords[player]
        mean = getMean(profitHistory)
        refStats[player][2] = mean
        variance = getVariance(profitHistory, mean)
        refStats[player][3] = variance
        thirdMoment = getAbsoluteThirdCentralMoment(profitHistory, mean)
        refStats[player][4] = thirdMoment
        samples = len(profitHistory)
        confInt = getConfidenceInterval(mean, variance, samples,
                                        alpha=confidenceAlpha)
        if confInt is not False:
            refStats[player][5] = confInt[0]
            refStats[player][6] = confInt[1]
        else:
            refStats[player][5] = False
            refStats[player][6] = False
        conservativeInt = conservativeConfidenceInterval(mean, variance,
                                                         thirdMoment, samples,
                                                         alpha=confidenceAlpha)
        if conservativeInt is not False:
            refStats[player][7] = conservativeInt[0]
            refStats[player][8] = conservativeInt[1]
        else:
            refStats[player][7] = False
            refStats[player][8] = False
        pValue = conservativePValue(mean, variance, thirdMoment, samples)
        refStats[player][9] = pValue

def findMinSampleRef(refStats):
    # Find the ref number corresponding to the fewest samples.
    sampleSizes = refStats[:,1]
    minSamples = min(sampleSizes)
    minSamplesIndex = np.where(sampleSizes == minSamples)[0][0]
    minSamplesRef = refStats[minSamplesIndex][0]
    minSamplesInfo = [minSamplesRef, minSamples]
    return minSamplesInfo

def finishedSampling(refStats, keyPlayers, requiredSampleSize):
    # If there are no key players then all players must meet the
    #required sample size.
    #print("\n \n \n Ref Stats")
    #print(refStats)
    if((keyPlayers is False) or (keyPlayers == [])):
        minSamplesInfo = findMinSampleRef(refStats)
        minSamples = minSamplesInfo[1]
    else:
        # Check every key player's samples.
        refNumbers = refStats[:,0]
        refSamples = refStats[:,1]
        # Set minSamples to a high value before finding min.
        minSamples = requiredSampleSize
        # Find the lowest samples of all the key players.
        for player in keyPlayers:
            keyPlayerIndex = np.where(refNumbers==player)[0][0]
            playerSamples = refStats[keyPlayerIndex][1]
            if(playerSamples < minSamples):
                minSamples = playerSamples
    # Print the minimum samples to display progress.
    print "minSamples: " + str(int(minSamples))
    if(minSamples < requiredSampleSize):
        return False
    else:
        return True

def saveRefStats(fileName, refStats):
    # Title columns of results and save to a file
    columnTitles = np.array(
        ["Decision Ref", "Sample size", "Mean profit",
         "Variance profit", "Third absolute moment", "Confidence lower",
         "Confidence upper", "Conservative confidence lower",
         "Conservative confidence upper", "Mean p value"])
    results = np.vstack([columnTitles, refStats])
    with open(fileName, "wb") as resultsFile:
        writer = csv.writer(resultsFile)
        writer.writerows(results)

def loadKerasModel(jsonFilePath, hFiveFilePath):
    # Load Json and create winDefeat model.
    jsonFile = open(jsonFilePath, 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    kerasModel = model_from_json(loadedModelJson)    
    # Load weights into winDefeat model.
    kerasModel.load_weights(hFiveFilePath)
    return kerasModel

def loadFirstNNModels(decisionRefNumber):
    # Load all NN models used by decision maker.
    # Get file path of each model.
    currentPath = os.getcwd()
    decisionMakerFolder = ("decisionMakers/decisionMaker"
                           + str(decisionRefNumber))
    decisionMakerPath = os.path.join(currentPath, decisionMakerFolder)
    winDefeatJsonFile = "winDefeatPrediction.json"
    winDefeatJsonPath = os.path.join(decisionMakerPath, winDefeatJsonFile)
    winDefeatWeightsFile = "winDefeatPrediction.h5"
    winDefeatWeightsPath = os.path.join(
        decisionMakerPath, winDefeatWeightsFile)

    profitJsonFile = "profitPrediction.json"
    profitJsonPath = os.path.join(decisionMakerPath, profitJsonFile)
    profitWeightsFile = "profitPrediction.h5"
    profitWeightsPath = os.path.join(decisionMakerPath, profitWeightsFile)

    lossJsonFile = "lossPrediction.json"
    lossJsonPath = os.path.join(decisionMakerPath, lossJsonFile)
    lossWeightsFile = "lossPrediction.h5"
    lossWeightsPath = os.path.join(decisionMakerPath, lossWeightsFile)

    # Load Json and create winDefeat model.
    winDefeatModel = loadKerasModel(winDefeatJsonPath, winDefeatWeightsPath)
    profitModel = loadKerasModel(profitJsonPath, profitWeightsPath)
    lossModel = loadKerasModel(lossJsonPath, lossWeightsPath)

    decisionModels = [winDefeatModel, profitModel, lossModel]
    return decisionModels

def loadLogisticModels(
    decisionRefNumber, winDefeatFile="winDefLogistic.pkl",
    profitFile="profitLogistic.pkl", lossFile="lossLogistic.pkl"):
    # Load all logistic models used by decision maker.
    # Get file path of each model.
    currentPath = os.getcwd()
    decisionMakerFolder = ("decisionMakers/decisionMaker"
                           + str(decisionRefNumber))
    decisionMakerPath = os.path.join(currentPath, decisionMakerFolder)
    winDefeatPath = os.path.join(decisionMakerPath, winDefeatFile)
    profitPath = os.path.join(decisionMakerPath, profitFile)
    lossPath = os.path.join(decisionMakerPath, lossFile)

    winDefeatModel = joblib.load(winDefeatPath)
    profitModel = joblib.load(profitPath)
    lossModel = joblib.load(lossPath)

    decisionModels = [winDefeatModel, profitModel, lossModel]
    return decisionModels

def loadPlayerModels(refNumbers):
    # Load each of the models used to decide bets.
    # playerModels contains two columns, one with ref Numbers, the other with
    #lists of decision models for that ref Number.
    playerModels = [0,0]
    playerModels[0] = refNumbers
    playerModels[1] = [[0]] * len(refNumbers)
    for index in range(len(refNumbers)):
        decisionMethod = PokerGames.getDecisionType(refNumbers[index])
        if(decisionMethod == "firstNNMethod"):
            decisionModels = loadFirstNNModels(refNumbers[index])
            playerModels[1][index] = decisionModels
        elif(decisionMethod == "logistic"):
            decisionModels = loadLogisticModels(refNumbers[index])
            playerModels[1][index] = decisionModels
        else:
            playerModels[1][index] = []
    return playerModels

def getPlayerModels(playerRefs, tournamentModels):
    # Return models in tournamentModels which correspond to playerRefs.
    playerModels = [0,0]
    # playerRefs are those playing the hand, refNumbers are those in the
    #tournament.
    playerModels[0] = playerRefs
    playerModels[1] = [[0]] * len(playerRefs)
    refNumbers = np.asarray(tournamentModels[:][0])

    for i in range(len(playerRefs)):
        # Find where in the list this player reference is.
        refIndex = np.where(refNumbers == playerRefs[i])[0][0]
        # Copy models for this player.
        playerModels[1][i] = tournamentModels[1][refIndex]
    return playerModels

def choosePlayers(
    refStats, maxPlayers, keyPlayers = False, playerDistribution = "uniform"):
    refNumbers = refStats[:,0]
    essentialPlayers = []
    # Player who has the fewest games played must be in game.
    minSamplesInfo = findMinSampleRef(refStats)
    minSamplesRef = minSamplesInfo[0]
    essentialPlayers.append(int(minSamplesRef))
    minSamples = minSamplesInfo[1]
    # One keyPlayer must be in the game.
    if((keyPlayers is not False) and (keyPlayers != [])):
        # Pick a random key player.
        keyIndex = np.random.randint(0, len(keyPlayers))
        if(keyPlayers[keyIndex] != minSamplesRef):
            # If random key player is not the one chosen for having
            #minimum samples then append this player.
            essentialPlayers.append(int(keyPlayers[keyIndex]))
    # Pick a random number of players to play a game.
    playerLimit = min(maxPlayers, len(refNumbers))
    numberPlayers = np.random.randint(2, playerLimit + 1)
    # Select random players.
    playerRefs = selectRandomPlayers(
        numberPlayers, refNumbers, essentialPlayers, playerDistribution)
    return playerRefs

def recordNewProfits(profitRecords, playerRefs, refNumbers, scaledProfits):
    # Record scaledprofits in the profitRecords list.
    # playerRefs are those who correspond to the new profits.
    # refNumbers are all the players in the tournament.
    for i in range(len(playerRefs)):
        # Find index of player's profit.
        playerIndex = refNumbers.index(playerRefs[i])
        profitRecords[playerIndex].append(scaledProfits[i])

def countPlayersActive(chips):
    # Count the number of players with more than 0 chips.
    playersActive = 0
    for i in range(len(chips)):
        if(chips[i] > 0):
            playersActive+=1
    return playersActive

def updateTableInfo(
    chips, refNumbers, tournamentPositions, originalRefNumbers):
    # Remove any players with 0 chips left and update lists accordingly.
    previousPlayersActive = len(chips)
    activePlayers = countPlayersActive(chips)
    position = activePlayers+1
    newChips = []
    newRefNumbers = []
    for player in range(previousPlayersActive):
        if chips[player] == 0:
            # Find this player's index in the original list of players
            #to record their position in the tournament.
            playerIndex = originalRefNumbers.index(refNumbers[player])
            # Multiple players get he same position if they lose in the
            #same hand.
            tournamentPositions[playerIndex] = position
        else:
            # Only append those players who have not lost.
            newChips.append(chips[player])
            newRefnumbers.append(refnumbers[player])
    # Set previous lists equal to new lists to avoid returning them.
    refNumbers = newRefNumbers
    chips = newChips
    dealerPosition = (dealerPosition+1) % activePlayers
    # If there is one player left record them as the winner.
    if activePlayers == 1:
        winnerIndex = originalRefNumbers.index(refNumbers[0])
        tournamentPositions[playerIndex] = 1
    return dealerPosition

def tableTournament(refNumbers, startChips=100, bigBlind=100):
    # Play a tournament until 1 player wins all chips.
    # Return a list with each player's position in the tournament.
    numberPlayers = len(refNumbers)
    if(numberPlayers > 10):
        print("Warning, larger number of players at one table")
        print(numberplayers, " players")
    tournamentPositions = [0]*numberPlayers
    chips = [startChips]*numberPlayers
    dealerPosition = 0
    originalRefNumbers = copy.copy(refNumbers)
    playerTournamentModels = loadPlayerModels(refNumbers)
    while(numberPlayers > 1):
        # Play one hand then update table conditions.
        playerNames = []
        playerModels = getPlayerModels(refNumbers, playerTournamentModels)
        for i in range(numberPlayers):
            playerNames.append(str(refNumbers[i]))
        chips = playHand(playerNames, chips, bigBlind, playerModels,
                         dealerPosition, decisionRefs, fileNames)
        # If anyone has 0 chips left then update table.
        if 0.0 in chips:
            dealerPosition = updateTableInfo(chips, refNumbers,
                                             tournamentPositions,
                                             originalRefNumbers)
    return tournamentPositions

def monteCarloGames(
    refNumbers, playerDistribution = "uniform", keyPlayers = False,
    bigBlind=100, minChips=10, maxChips=200, sampleSize=1000, maxPlayers=8,
    recordBets=False,
    resultsFile = "decisionMakers/decisionMakersComparison.csv"):
    # Play hands with random chip counts to test player performance in
    #random scenarios.
    # refNumbers is a list of all refNumberswhich can play.
    tournamentSize = len(refNumbers)
    # keyPlayers is list of specific refs being tested. At least one
    #must play in each game.
    if((keyPlayers is not False) and (keyPlayers != [])):
        for i in range(len(keyPlayers)):
            if(keyPlayers[i] not in refNumbers):
                # Put key player at the start of the refs list.
                refNumbers.insert(keyPlayer[i])
    # Prepare stats array for each player.
    # refStats contains refNumbers, samples, sumProfit, sumSqProfit,
    #meanProfit, varProfit, ZScore.
    refNumArray = np.copy(refNumbers)
    refStats = np.zeros((len(refNumbers), 10))
    refStats[:,0] = refNumbers

    # Load decision making models.
    playerTournamentModels = loadPlayerModels(refNumbers)

    # Loop until all relevant players reach the sample size requirement.
    sampleSizeReached = False

    # Create a list of lists to contain profits from games.
    profitRecords = []
    for i in range(tournamentSize):
        profitRecords.append([])

    while(not sampleSizeReached):
        # Choose players to play in this game.
        playerRefs = choosePlayers(refStats, maxPlayers,
                                   keyPlayers = keyPlayers,
                                   playerDistribution = playerDistribution)
        # Get models for those playing this game.
        playerModels = getPlayerModels(playerRefs, playerTournamentModels)
        # Play one hand.
        profits = playRandomHand(playerRefs, playerModels, bigBlind=bigBlind,
                                 minChips=minChips, maxChips=maxChips,
                                 recordBets=recordBets)
        # Scale profits by big blind.
        profitsArray = np.array(profits)
        scaledProfits = np.divide(profitsArray, bigBlind)
        # Add latest game profits to records.
        recordNewProfits(profitRecords, playerRefs, refNumbers, scaledProfits)
        # Update refStats.
        updateRefStats(refStats, playerRefs, profitRecords)
        # Test if sampleSize is reached.
        sampleSizeReached = finishedSampling(refStats, keyPlayers, sampleSize)
        
    # Display the stats for each player.
    print refStats
    saveRefStats(resultsFile, refStats)

# Test performance of decision makers 1-10.
players = [1,2,3,4,5,6,7,8,9,10,12,13]
print players
keyPlayers = [13]
recordBets = 13
monteCarloGames(players, sampleSize = 10000, keyPlayers=keyPlayers,
                recordBets=recordBets)
