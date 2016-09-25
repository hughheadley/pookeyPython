from __future__ import division
import numpy as np
import random
import os
import PokerGames
import math
import csv

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
        for i in range(0, len(essentialPlayerRefs)):
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
    maxNumberFiles = 1
    numberPlayers = len(decisionRefs)
    decisionFiles = np.array([''], dtype=object)
    decisionFiles.resize((numberPlayers, maxNumberFiles))    
    # Find current working directory.
    currentPath = os.getcwd()
    for i in range(0, numberPlayers):
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
        for j in range(0, len(content)):
            decisionFiles[i][j] = content[j]
    return decisionFiles

def randomChips(bigBlind, minChips, maxChips, initialNumberPlayers):
    initialChips = [0] * initialNumberPlayers
    for i in range(0, initialNumberPlayers):
        randNumber = np.random.random()
        logMin = math.log(minChips)
        logMax = math.log(maxChips)
        logRandomNumber = logMin + (randNumber * (logMax - logMin))
        chipValue = bigBlind * (2.71828 ** logRandomNumber)
        initialChips[i] = int(chipValue)
    return initialChips

def playRandomHand(
    decisionRefs, bigBlind = 100, minChips = 10, maxChips = 200):
    # Players start with a random amount of chips to simulate a random
    #point in a tournament.
    initialNumberPlayers = len(decisionRefs)
    # Set initial chips for the players.
    startChips = randomChips(
        bigBlind, minChips, maxChips, initialNumberPlayers)
    initialChips = np.copy(startChips)
    # Set players' names as their ref numbers.
    playerNames = []
    for i in range(0, initialNumberPlayers):
        playerNames.append(str(decisionRefs[i]))
    # Prepare values for playHand function.
    AIPlayers = [True] * initialNumberPlayers
    manualDealing = False
    trainingMode = True
    dealerPosition = 0
    fileNames = getFileNames(decisionRefs)
    # Play one hand.
    finalChips = PokerGames.playhand(
    playerNames, initialChips, bigBlind, dealerPosition,
    manualDealing, trainingMode, AIPlayers, decisionRefs, fileNames,
    recordBets = True)
    # Convert the chips list into a numpy array.
    np.asarray(finalChips)
    profits = np.subtract(finalChips, initialChips)
    return profits

def updateRefStats(refStats, playerRefs, scaledProfits):
    refNumbers = refStats[:,0]
    for i in range(0, len(playerRefs)):
        # Find where in the list this decision maker is.
        refIndex = np.where(refNumbers == playerRefs[i])[0][0]
        # Update the stats for that index.
        # Increase sample size.
        refStats[refIndex][1] += 1
        # Add profit.
        refStats[refIndex][2] += scaledProfits[i]
        # Add profit squared.
        refStats[refIndex][3] += (scaledProfits[i] ** 2)
        # Compute new mean profit.
        refStats[refIndex][4] = refStats[refIndex][2] / refStats[refIndex][1]
        # Compute new variance of profit.
        refStats[refIndex][5] = ((refStats[refIndex][3] / refStats[refIndex][1])
            - (refStats[refIndex][4] ** 2))
        # Compute new Z score. If variance or sample size is zero
        #then Z=0.
        if((refStats[refIndex][1] > 0) and (refStats[refIndex][5] > 0)):
            refStats[refIndex][6] = (refStats[refIndex][4] /
                ((refStats[refIndex][5] / refStats[refIndex][1]) ** 0.5))

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
        for i in range(0, len(keyPlayers)):
            keyPlayerIndex = refNumbers.index(keyPlayers[i])
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
        ["Decision Ref", "Sample size", "Profit sum", "Sum profit squared",
        "Mean profit", "Variance profit", "Z score"])
    results = np.vstack([columnTitles, refStats])
    with open(fileName, "wb") as resultsFile:
        writer = csv.writer(resultsFile)
        writer.writerows(results)

def monteCarloGames(
    refNumbers, playerDistribution = "uniform", keyPlayers = False,
    bigBlind = 100, minChips = 10, maxChips = 200, sampleSize = 1000,
    maxPlayers = 8, resultsFile = "decisionTest.csv"):
    # Play hands with random chip counts to test player performance in
    #random scenarios.
    # refNumbers is a list of all refNumbers which can play.
    # keyPlayers is list of specific refs being tested. At least one
    #must play in each game.
    if((keyPlayers is not False) and (keyPlayers != [])):
        for i in range(0, ln(keyPlayers)):
            if(keyPlayers[i] not in refNumbers):
                # Put key player at the start of the refs list.
                refNumbers.insert(keyPlayer[i])
    # Prepare stats array for each player.
    # refStats contains refNumbers, samples, sumProfit, sumSqProfit,
    #meanProfit, varProfit, ZScore.
    refNumArray = np.copy(refNumbers)
    refStats = np.zeros((len(refNumbers), 7))
    refStats[:,0] = refNumbers
    # Loop until all relevant players reach the sample size requirement.
    sampleSizeReached = False
    while(not sampleSizeReached):
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
                essentialPlayers.append(int(keyPlayers[keyIndex]))
        # Pick a random number of players to play a game.
        playerLimit = min(maxPlayers, len(refNumbers))
        numberPlayers = np.random.randint(2, playerLimit + 1)
        # Select random players.
        playerRefs = selectRandomPlayers(
            numberPlayers, refNumbers, essentialPlayers, playerDistribution)
        # Play one hand.
        profits = playRandomHand(
            playerRefs, bigBlind = bigBlind, minChips = minChips,
            maxChips = maxChips)
        # Scale profits by big blind.
        profitsArray = np.array(profits)
        scaledProfits = np.divide(profits, bigBlind)
        # Update refStats.
        updateRefStats(refStats, playerRefs, scaledProfits)
        # Test if sampleSize is reached.
        sampleSizeReached = finishedSampling(refStats, keyPlayers, sampleSize)
    # Display the stats for each player.
    print refStats
    saveRefStats(resultsFile, refStats)

# Test performance of decision makers 1-10
players = range(1,11)
print players
monteCarloGames(players, sampleSize = 1000)
