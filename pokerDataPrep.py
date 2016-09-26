from __future__ import division
import numpy as np
import csv
import math
import AIDecisions

def splitProfitLoss(
    allDataFile = "betRecords.csv", profitFile="profitData.csv",
    lossFile="lossData.csv", zeroProfit="profit"):
    # Split all data into two files with profit data and loss data.
    # Open data into a numpy array.
    csvData = csv.reader(open(allDataFile))
    betData = [line for line in csvData]
    allData = np.asarray(betData)
    print()
    profitData = np.empty(allData.shape[1])
    lossData = np.empty(allData.shape[1])
    for line in allData:
        # Split into profit and loss.
        stringLine = np.asarray(line)
        tempLine = np.asarray([float(l) for l in stringLine])
        if(tempLine[0] > 0):
            profitData = np.vstack([profitData, tempLine])
        elif(tempLine[0] < 0):
            lossData = np.vstack([lossData, tempLine])
        elif(zeroProfit == "profit"):
            profitData = np.vstack([profitData, tempLine])
        elif(zeroProfit == "loss"):
            lossData = np.vstack([lossData, tempLine])
        else:
            print(
                "Error in pokerDataPrep module in splitProfitLoss function"
                + " problem determining profit or loss")
    np.savetxt(profitFile, profitData, fmt='%3f', delimiter=',')
    np.savetxt(lossFile, lossData, fmt='%3f', delimiter=',')

def saveWinDefeat(
    inputsCount, allDataFile = "betRecords.csv",
    winDefeatFile="winDefeatData.csv", zeroProfit="profit"):
    # Convert data about profit and game state into Win/Defeat training data.
    csvData = csv.reader(open(allDataFile))
    betData = [line for line in csvData]
    allData = np.asarray(betData)
    # winDefeatData contains win/Defeat bool and input layer.
    winDefeatData = np.empty(inputsCount + 1)
    for line in allData:
        # Convert to win/Defeat bool and input layer.
        # Change strings from file to floats.
        stringLine = np.asarray(line)
        tempLine = np.asarray([])
        for l in stringLine:
            if(l != ''):
                tempLine = np.append(tempLine, float(l))
        # Get betting conditions (New bet and game state).
        betConditions = tempLine[1:]
        inputLayer = AIDecisions.prepareFirstNNInputs(betConditions)
        if(tempLine[0] > 0):
            # Insert a 1 to indicate win.
            inputLayer = np.insert(inputLayer, 0, 1)
            winDefeatData = np.vstack([winDefeatData, inputLayer])
        elif(tempLine[0] < 0):
            # Insert a 0 to indicate defeat.
            inputLayer = np.insert(inputLayer, 0, 0)
            winDefeatData = np.vstack([winDefeatData, inputLayer])
        elif(zeroProfit == "profit"):
            # Insert a 1 to indicate win.
            inputLayer = np.insert(inputLayer, 0, 1)
            winDefeatData = np.vstack([winDefeatData, inputLayer])
        elif(zeroProfit == "loss"):
            # Insert a 0 to indicate defeat.
            inputLayer = np.insert(inputLayer, 0, 0)
            winDefeatData = np.vstack([winDefeatData, inputLayer])
        else:
            print(
                "Error in pokerDataPrep module in saveWinDefeat function"
                + " problem determining profit or loss")
    # Save created data to file.
    np.savetxt(winDefeatFile, winDefeatData, fmt='%3f', delimiter=',')
