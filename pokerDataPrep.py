from __future__ import division
import numpy as np
import csv
import math
import AIDecisions
import os

def splitProfitLoss(
    inputsCount, allDataFile="trainingDump/betRecords.csv",
    profitFile="trainingDump/profitData.csv",
    lossFile="trainingDump/lossData.csv",
    zeroProfit="profit"):
    # Split all data into two files with profit data and loss data.
    # Open data into a numpy array.
    csvData = csv.reader(open(allDataFile))
    betData = [line for line in csvData]
    allData = np.asarray(betData)
    profitData = np.empty(inputsCount + 1)
    lossData = np.empty(inputsCount + 1)
    for line in allData:
        # Split into profit and loss.
        stringLine = np.asarray(line)
        tempLine = np.asarray([])
        for l in stringLine:
            if(l != ''):
                tempLine = np.append(tempLine, float(l))
        # Get betting conditions (New bet and game state).
        betConditions = tempLine[1:]
        inputLayer = AIDecisions.prepareFirstNNInputs(betConditions)
        if(tempLine[0] > 0):
            inputLayer = np.insert(inputLayer, 0, tempLine[0])
            profitData = np.vstack([profitData, inputLayer])
        elif(tempLine[0] < 0):
            inputLayer = np.insert(inputLayer, 0, tempLine[0])
            lossData = np.vstack([lossData, inputLayer])
        elif(zeroProfit == "profit"):
            inputLayer = np.insert(inputLayer, 0, tempLine[0])
            profitData = np.vstack([profitData, inputLayer])
        elif(zeroProfit == "loss"):
            inputLayer = np.insert(inputLayer, 0, tempLine[0])
            lossData = np.vstack([lossData, inputLayer])
        else:
            print(
                "Error in pokerDataPrep module in splitProfitLoss function"
                + " problem determining profit or loss")
    np.savetxt(profitFile, profitData, fmt='%3f', delimiter=',')
    np.savetxt(lossFile, lossData, fmt='%3f', delimiter=',')

def saveWinDefeat(
    inputsCount, allDataFile = "trainingDump/betRecords.csv",
    winDefeatFile="trainingDump/winDefeatData.csv", zeroProfit="profit"):
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

def discretizeProfits(
    inputsCount, numberCategories, decisionRefNumber, profitFile="trainingDump/profitData.csv",
    discreteProfitFile="trainingDump/discreteProfitData.csv"):
    # Read profit game data into an array.
    csvData = csv.reader(open(profitFile))
    betData = [line for line in csvData]
    stringProfitData = np.asarray(betData)
    profitData = stringProfitData.astype(float)
    # Put first column (profits) into an array.
    profits = profitData[:,0]
    # Divide the profits into categories according to data quantiles.
    quantiles = []
    for i in range(0, numberCategories):
        quantiles.append((100 * i) / numberCategories)
    # Get the profit values corresponding to quantiles.
    profitCatLimits = np.percentile(profits, quantiles)
    # Check if category limits duplicated because of many 0 values.
    duplicateCategories = (profitCatLimits[0] == profitCatLimits[1])
    if(duplicateCategories):
        print(
            "Two profit categories have the same limit."
            + "Try using fewer categories")
    # Lower limit of first category must always be 0.
    profitCatLimits[0] = 0
    # Add upper limit to categories using 1 above the max profit.
    profitCatLimits = np.append(profitCatLimits, np.amax(profits) + 1)
    # Create boolean values indicating which category a profit value is in.
    rowCount = profitData.shape[0]
    discretizedProfits = np.zeros([rowCount, numberCategories])
    for j in range(0, rowCount):
        rowProfit = profitData[j][0]
        for i in range(0, numberCategories):
            inCategory = ((profitCatLimits[i] <= rowProfit)
            and (rowProfit < profitCatLimits[i + 1]))
            if(inCategory):
                discretizedProfits[j][i] = 1
    # Join boolean profit values with remaining Data.
    discretizedData = np.hstack([discretizedProfits, profitData[:,1:]])
    # Save discretized data to file.
    with open(discreteProfitFile, 'wb') as f:
        np.savetxt(f, discretizedData, delimiter=',')
    # Get the average profit for each category.
    # Multiply boolean values for category by the profit in each row.
    categorySumProfits = np.zeros(numberCategories)
    categoryAverageProfits = np.zeros(numberCategories)
    for i in range(0, numberCategories):
        categoryBools = discretizedData[:, i]
        categorySumProfits[i] = np.dot(categoryBools, profits)
        categoryAverageProfits[i] = (categorySumProfits[i] / sum(categoryBools))
    # Write the profit category limits and the average profits in each category
    #to a file in the given decision maker's folder.
    categoryInfo = np.zeros([2, numberCategories + 1])
    for i in range(0, numberCategories + 1):
        categoryInfo[0][i] = profitCatLimits[i]
    for i in range(0, numberCategories):
        categoryInfo[1][i] = categoryAverageProfits[i]
    currentPath = os.getcwd()
    subFolder = ("decisionMakers/decisionMaker" + str(decisionRefNumber)
    + "/profitCategoryInfo.txt")
    fullpath = os.path.join(currentPath, subFolder)
    with open(fullpath, 'wb') as f:
        np.savetxt(f, categoryInfo, delimiter=',')
        
def discretizeLosses(
    inputsCount, numberCategories, decisionRefNumber, lossFile="trainingDump/lossData.csv",
    discreteLossFile="trainingDump/discreteLossData.csv"):
    # Read loss game data into an array.
    csvData = csv.reader(open(lossFile))
    betData = [line for line in csvData]
    stringLossData = np.asarray(betData)
    lossData = stringLossData.astype(float)
    # Put first column (losses) into an array.
    losses = lossData[:,0]
    # Divide the losses into categories according to data quantiles.
    quantiles = []
    for i in range(1, numberCategories + 1):
        quantiles.append((100 * i) / numberCategories)
    # Get the loss values corresponding to quantiles.
    lossCatLimits = np.percentile(losses, quantiles)
    # Check if category limits duplicated because of many 0 values.
    duplicateCategories = (lossCatLimits[numberCategories - 1]
    == lossCatLimits[numberCategories - 2])
    if(duplicateCategories):
        print(
            "Two loss categories have the same limit."
            + "Try using fewer categories")
    # Lower limit of final category must always be 0.
    lossCatLimits[numberCategories - 1] = 0
    # Add lower limit to categories using 1 below the max loss.
    lossCatLimits = np.insert(lossCatLimits, 0, np.amin(losses) - 1)
    # Create boolean values indicating which category a loss value is in.
    rowCount = lossData.shape[0]
    discretizedLosses = np.zeros([rowCount, numberCategories])
    for j in range(0, rowCount):
        rowLoss = lossData[j][0]
        for i in range(0, numberCategories):
            inCategory = ((lossCatLimits[i] < rowLoss)
            and (rowLoss <= lossCatLimits[i + 1]))
            if(inCategory):
                discretizedLosses[j][i] = 1
    # Join boolean loss values with remaining data.
    discretizedData = np.hstack([discretizedLosses, lossData[:,1:]])
    # Save discretized data to file.
    with open(discreteLossFile, 'wb') as f:
        np.savetxt(f, discretizedData, delimiter=',')
    # Get the average loss for each category.
    # Multiply boolean values for category by the loss in each row.
    categorySumLosses = np.zeros(numberCategories)
    categoryAverageLosses = np.zeros(numberCategories)
    for i in range(0, numberCategories):
        categoryBools = discretizedData[:, i]
        categorySumLosses[i] = np.dot(categoryBools, losses)
        categoryAverageLosses[i] = (categorySumLosses[i] / sum(categoryBools))
    # Write the loss category limits and the average losses in each category
    #to a file in the given decision maker's folder.
    categoryInfo = np.zeros([2, numberCategories + 1])
    for i in range(0, numberCategories + 1):
        categoryInfo[0][i] = lossCatLimits[i]
    for i in range(0, numberCategories):
        categoryInfo[1][i] = categoryAverageLosses[i]
    currentPath = os.getcwd()
    subFolder = ("decisionMakers/decisionMaker" + str(decisionRefNumber)
    + "/lossCategoryInfo.txt")
    fullpath = os.path.join(currentPath, subFolder)
    with open(fullpath, 'wb') as f:
        np.savetxt(f, categoryInfo, delimiter=',')

def prepareNNTrainingData(
    inputsCount, profitDiscretizationCategories, lossDiscretizationCategories,
    decisionRefNumber, zeroProfit="profit",
    allDataFile="trainingDump/betRecords.csv",
    winDefeatFile="trainingDump/winDefeatData.csv",
    profitFile="trainingDump/profitData.csv",
    lossFile="trainingDump/lossData.csv",
    discreteProfitFile="trainingDump/discreteProfitData.csv",
    discreteLossFile="trainingDump/discreteLossData.csv"):
    # From records of betting decisions and results produce data for the
    #win/defeat outcome, and for the profits and losses discretized.
    saveWinDefeat(
        inputsCount, allDataFile=allDataFile, winDefeatFile=winDefeatFile,
        zeroProfit=zeroProfit)        
    splitProfitLoss(
        inputsCount, allDataFile=allDataFile, profitFile=profitFile,
        lossFile=lossFile, zeroProfit=zeroProfit)
    discretizeProfits(
        inputsCount, profitDiscretizationCategories, decisionRefNumber,
        profitFile=profitFile, discreteProfitFile=discreteProfitFile)
    discretizeLosses(
        inputsCount, lossDiscretizationCategories, decisionRefNumber,
        lossFile=lossFile, discreteLossFile=discreteLossFile)
