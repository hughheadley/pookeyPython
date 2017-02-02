from __future__ import division, print_function
import numpy as np
import random
import os
import sklearn.linear_model
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, mean_absolute_error

def load_data(csvDataSet):
    # Load training dataset from csv and return list of lists with data.
    dataSet = np.loadtxt(csvDataSet, delimiter=",")
    return dataSet

def evaluate_auc_score(model, Xtest, Ytest):
    # Return the average ROC AUC score over all categories.
    numberCategories = len(Ytest[0])
    # Predict probability of each category.
    YDistribution = model.predict_proba(Xtest)
    
    aucSum = 0
    for category in range(numberCategories):
        YTrue = Ytest[:,category]
        YHat = YDistribution[:,category]
        auc = roc_auc_score(YTrue, YHat)
        aucSum += auc

    aucAverage = aucSum/numberCategories
    return aucAverage

def label_data(dataSet):
    # Take binary categories and convert them to class labels for a single
    #variable.
    dataSet = np.asarray(dataSet)
    dataLength = dataSet.shape[0]
    numberCategories = len(dataSet[0])
    classLabels = []
    for row in dataSet:
        for col in range(numberCategories):
            # If this column is the correct output the save this label.
            if(row[col]==1.0):
                # Use column index as label.
                label = str(col)
                classLabels.append(label)
    # Convert to numpy array for compatibility with sklearn.
    classLabels = np.asarray(classLabels)
    return classLabels

def prediction_performance(model, Xtest, Ytest, numberCategories):
    # Calculate metric for logistic regression performance.
    if(numberCategories == 1):
        # Get metrics for binary classification.
        YDistribution = model.predict_proba(Xtest)[:,1]
        YClassification = model.predict(Xtest)
        auc = roc_auc_score(Ytest, YDistribution)
        print("AUC", auc)
        MAE = mean_absolute_error(Ytest, YDistribution)
        print("MAE", MAE)
        accuracy = 1 - mean_absolute_error(YClassification, Ytest)
        print("Accuracy", accuracy)
        metrics = [accuracy, auc, MAE]
    else:
        # Get metric for multiple class classification.
        YPredictions = model.predict(Xtest)
        YDistribution = model.predict_proba(Xtest)
        YTestLabels = label_data(Ytest)
        accuracy = model.score(Xtest, YTestLabels)
        print("Accuracy", accuracy)
        avAUC = evaluate_auc_score(model, Xtest, Ytest)
        print("Av AUC", avAUC)
        #auc = roc_auc_score(Ytest, YPredictions)
        MAE = mean_absolute_error(Ytest, YDistribution)
        print("MAE", MAE)
        metrics = [accuracy, avAUC, MAE]
    return metrics

def make_data_quadratic(dataSet, constantIncluded=False):
    # Take data for variables X and return data for variables XX and X.
    dataLength = dataSet.shape[0]
    rowLength = dataSet.shape[1]
    # Add a constant variable if not already included.
    if constantIncluded==False:
        # Make table of ones with one column more than dataSet.
        b = np.ones((dataLength, rowLength+1))
        # Overwrite ones table with dataSet
        b[:,1:] = dataSet
        dataSet = b
        
    # For each row of data take the outer product of all variables.
    newData = []
    for rowIndex in range(dataLength):
        row = dataSet[rowIndex]
        newRow = []
        # Compute all unique multiples of variables.
        for i in range(len(row)):
            for j in range(i, len(row)):
                newRow.append(row[i]*row[j])
        newData.append(newRow)
    return newData

def fit_logistic(
    dataSet, numberCategories, trainingProportion=0.7, quadratic=True,
    constantIncluded=False):
    # Order the data randomly.
    random.shuffle(dataSet)
    # Split into input (X) and output (Y) variables and into training
    #and test data.
    dataSet = np.asarray(dataSet)
    dataLength = dataSet.shape[0]
    print("dataLength", dataLength)
    trainingDataLength = int(dataLength * trainingProportion)
    testDataLength = dataLength - trainingDataLength
    Xtrain = dataSet[:trainingDataLength,numberCategories:]
    Ytrain = dataSet[:trainingDataLength,0:numberCategories]
    Xtest = dataSet[trainingDataLength:,numberCategories:]
    Ytest = dataSet[trainingDataLength:,0:numberCategories]

    # If using quadratic variables then modify data.
    if(quadratic == True):
        Xtrain = make_data_quadratic(Xtrain,
                                     constantIncluded=constantIncluded)
        Xtest = make_data_quadratic(Xtest,
                                    constantIncluded=constantIncluded)

    # If not training for multiple categories flatten Ytrain.
    if(numberCategories==1):
        Ytrain = np.ndarray.flatten(Ytrain)
    else:
        # Convert binary outputs to labels.
        Ytrain = label_data(Ytrain)

    # Train and test model.
    logistic = sklearn.linear_model.LogisticRegression(penalty="l2")
    model = logistic.fit(Xtrain, Ytrain)
    prediction_performance(model, Xtest, Ytest, numberCategories)

    return model

def pickel_model(model, fileName):
    # Save model to a .pkl file
    joblib.dump(model, fileName)

def produce_logistic(
    dataFile, numberCategories, modelFile, resultsFile=None,
    constantIncluded=False, makeQuadratic=True):
    # Using data from dataFile csv train a logistic regression model.
    # Save model to modelFile in .pkl file format.
    # Open the data.
    dataSet = load_data(dataFile)
    # Fit the model.
    model = fit_logistic(dataSet, numberCategories, quadratic=makeQuadratic,
                        constantIncluded=constantIncluded)
    # Save the model.
    pickel_model(model, modelFile)

def train_all_predictionModels(
    decisionRefNumber, winDefeatDataFile, profitDataFile, lossDataFile,
    profitCategoryCount, lossCategoryCount, winDefeatModelFile,
    profitModelFile, lossModelFile):
    # Train logistic regression models for winDefeat, profit, and loss
    #prediction. Save models to specified files.
    # Get full path for files    
    currentPath = os.getcwd()
    subFolder = "decisionMakers/decisionMaker" + str(int(decisionRefNumber))
    fullpath = os.path.join(currentPath, subFolder)
    winDefeatPath = os.path.join(fullpath, winDefeatModelFile)
    profitPath = os.path.join(fullpath, profitModelFile)
    lossPath = os.path.join(fullpath, lossModelFile)
    
    windefeatCategories=1
    produce_logistic(winDefeatDataFile, windefeatCategories, winDefeatPath)
    produce_logistic(profitDataFile, profitCategoryCount, profitPath)
    produce_logistic(lossDataFile, lossCategoryCount, lossPath)

profitCategoryCount=5
lossCategoryCount=5
decisionRef = 12
train_all_predictionModels(decisionRef, "trainingDump/winDefeatData.csv", "trainingDump/discreteProfitData.csv",
                           "trainingDump/discreteLossData.csv", profitCategoryCount,
                           lossCategoryCount, "winDefLogistic.pkl",
                           "profitLogistic.pkl", "lossLogistic.pkl")
