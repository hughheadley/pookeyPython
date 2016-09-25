from __future__ import division
import numpy as np
import csv
import math

def splitProfitLoss(
    allDataFile = "betRecords.csv", profitFile="profitData.csv",
    lossFile="lossData.csv", zeroProfit="profit"):
    # Split all data into two files with profit data and loss data.
    # Open data into a numpy array.
    csvData = csv.reader(open(allDataFile))
    betData = [line for line in csvData]
    allData = np.asarray(betData)
    print()
    profitData = np.empty(allData.shape[1]) #np.asarray([])#
    lossData = np.empty(allData.shape[1])  #np.empty(1) #
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
            print("Error in pokerDataPrep module in splitProfitLoss function" +
                " problem determining profit or loss")
        print(profitData)
    # Save split data into two files.
    #np.savetxt(profitFile, profitData, delimiter = ",")
    #with open(profitFile,'wb') as f:
    np.savetxt(profitFile,profitData,fmt='%3f', delimiter=',')
    np.savetxt(lossFile,lossData,fmt='%3f', delimiter=',')

