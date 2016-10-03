from sklearn.metrics import roc_auc_score, mean_absolute_error
import sys
import numpy as np
import random
import time
from keras.models import model_from_json
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, rmsprop, SGD

def cartesian(arrays, out=None):
    # Generate a cartesian product of input arrays.
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def getLayerSizes(numberHiddenLayers, inputLayerSize, units1, units2, units3):    
    # Set the layer sizes.
    if(numberHiddenLayers == 1):                                
        layerSizes = [inputLayerSize, units1, 1]
    elif(numberHiddenLayers == 2):                                
        layerSizes = [inputLayerSize, units1, units2, 1]
    elif(numberHiddenLayers == 3):                                
        layerSizes = [inputLayerSize, units1, units2, units3, 1]
    return layerSizes

def buildModel(layerSizes, dropouts, optimizer, learningRate):
    numberHiddenLayers = len(layerSizes) - 2
    inputLayerSize = layerSizes[0]
    units1 = layerSizes[1]
    dropout1 = dropouts[0]
    dropout2 = dropouts[1]
    dropout3 = dropouts[2]
    
    model = Sequential()
    # Add connections from input layer to first hidden layer.
    model.add(Dense(output_dim=units1, input_dim = inputLayerSize)) 
    model.add(Activation('relu'))
    model.add(Dropout(dropout1))

    # Add connections from first hidden layer to second hidden layer.
    if numberHiddenLayers >= 2:
        units2 = layerSizes[2]
        model.add(Dense(output_dim=units2, init = "glorot_normal")) 
        model.add(Activation('relu'))
        model.add(Dropout(dropout2))

    # Add connections from second hidden layer to second third layer.
    if numberHiddenLayers >= 3:
        units3 = layerSizes[3]
        model.add(Dense(output_dim=units3, init = "glorot_normal")) 
        model.add(Activation('relu'))
        model.add(Dropout(dropout3))

    # Add final output layer with sigmoid activation.
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Set the optimizer.
    if(optimizer == 'sgd'):
        sgd = SGD(lr=learningRate)                                    
        model.compile(loss='binary_crossentropy', optimizer= sgd, metrics=['accuracy'])
    elif(optimizer == 'adadelta'):
        adadelta = Adadelta(lr=learningRate)
        model.compile(loss='binary_crossentropy', optimizer= adadelta, metrics=['accuracy'])
    elif(optimizer == 'adam'):
        adam = Adam(lr=learningRate)
        model.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'])
    elif(optimizer == 'rmsprop'):
        Rmsprop = rmsprop(lr=learningRate)
        model.compile(loss='binary_crossentropy', optimizer= Rmsprop, metrics=['accuracy'])
    return model

def testModel(
    model, layerSizes, Xtrain, Ytrain, Xtest, Ytest, learningRate, epochs,
    batchSize, optimizer, resultsFile = "selfOptLog.txt", printResults = False,
    elapsedTime = False):
    numberHiddenLayers = len(layerSizes) - 2
    inputLayerSize = layerSizes[0]
    units1 = layerSizes[1]
    dropout1 = dropouts[0]
    dropout2 = dropouts[1]
    dropout3 = dropouts[2]
    # Test MAE of model on training data (to check for overfitting).
    trainingPredY = model.predict_proba(Xtrain, verbose = 0)
    MAETrain = mean_absolute_error(Ytrain, trainingPredY)
    scores = model.evaluate(Xtrain, Ytrain, verbose=0)
    trainingAccuracy = scores[1] * 100

    # Test MAE, AUC, and Accuracy on test data.
    testPredY = model.predict_proba(Xtest, verbose = 0)
    auc = roc_auc_score(Ytest, testPredY)
    MAE = mean_absolute_error(Ytest, testPredY)
    bestGuess = [] # The estimated 0 or 1 output.
    for predictionCount in range (0, len(testPredY)):
        if(testPredY[predictionCount] < 0.5):
            bestGuess.append(0)
        else:
            bestGuess.append(1)
    testAccuracy = 1 - mean_absolute_error(bestGuess, Ytest)
                                    
    # Evaluate the model and write results to a file.
    if(printResults):
        print("Training MAE: %.2f%%" % (MAETrain * 100))
        print("acc: %.2f%%" % (testAccuracy*100))
        print("AUC: %.2f%%" % (auc*100))
        print("MAE: %.2f%%" % (MAE*100))
        print("%s , %s , %s, %s, %s , %s , %s , %s , %s , %s, %s \n"
            % (units1, units2, units3, learningRate, epochs, batchSize,
            patience, optimizer, dropout1, dropout2, dropout3))
        print("\n")        
    # Write model results to a file.
    if(elapsedTime is not False):
        with open(resultsFile, "a") as text_file:
            text_file.write(
                "%s, %s , %s , %s, %s , %s , %s , %s , %s , %s, %s , %s , %s , %s , %s , %s, %s \n"
                % (elapsedTime, MAETrain, trainingAccuracy, testAccuracy, MAE, units1,
                units2, units3, learningRate, epochs, batchSize, patience,
                optimizer, dropout1, dropout2, dropout3))
    else:
        with open(resultsFile, "a") as text_file:
            text_file.write(
                "%s , %s , %s , %s , %s , %s , %s , %s, %s , %s , %s , %s , %s , %s, %s \n"
                % (MAETrain, trainingAccuracy, testAccuracy, auc, MAE, units1, units2, units3,
                learningRate, epochs, batchSize, optimizer, dropout1, dropout2,
                dropout3))

# Load training dataset.
dataset = np.loadtxt("trainingDump/winDefeatData.csv", delimiter=",")
# Order the data randomly.
random.shuffle(dataset)
# Split into input (X) and output (Y) variables and into training
#and test data.
trainingProportion = 0.7 # 70% of the data is used for training.
dataLength = dataset.shape[0]
trainingDataLength = int(dataLength * trainingProportion)
testDataLength = dataLength - trainingDataLength

Xtrain = dataset[:trainingDataLength,1:]
Ytrain = dataset[:trainingDataLength,0]
Xtest = dataset[trainingDataLength:,1:]
Ytest = dataset[trainingDataLength:,0]
inputLayerSize = Xtrain.shape[1]

# Specify all hyperparameters which are not varying.
numberHiddenLayers = 2
numberLayers = numberHiddenLayers + 2

# Specify options for other hyperparameters.
learningRateOptions = [0.0045]
units1Options = [20]
units2Options = [20]
units3Options = [0]
batchSizeOptions = [15]
epochsOptions = [4000]
optimizerOptions = ['adam']#['rmsprop','sgd','adadelta']
dropout1Options = [0.5]
dropout2Options = [0.5]
dropout3Options = [0]
patienceOptions = [10]

# Get list of arrays for options. Optimizer options are excluded.
hyperParameterOptions = []
hyperParameterOptions.append(learningRateOptions)
hyperParameterOptions.append(units1Options)
hyperParameterOptions.append(units2Options)
hyperParameterOptions.append(units3Options)
hyperParameterOptions.append(batchSizeOptions)
hyperParameterOptions.append(epochsOptions)
hyperParameterOptions.append(dropout1Options)
hyperParameterOptions.append(dropout2Options)
hyperParameterOptions.append(dropout3Options)
hyperParameterOptions.append(patienceOptions)
hyperParameterCombinations = cartesian(hyperParameterOptions)

# Test performance of all hyperparameter options.
for i in range(0, len(optimizerOptions)):
    optimizer = optimizerOptions[i]
    for j in range(0, hyperParameterCombinations.shape[0]):
        startTime = time.time() * 1000
        learningRate = hyperParameterCombinations[j][0]
        units1 = int(hyperParameterCombinations[j][1])
        units2 = int(hyperParameterCombinations[j][2])
        units3 = int(hyperParameterCombinations[j][3])
        batchSize = int(hyperParameterCombinations[j][4])
        epochs = int(hyperParameterCombinations[j][5])
        dropout1 = hyperParameterCombinations[j][6]
        dropout2 = hyperParameterCombinations[j][7]
        dropout3 = hyperParameterCombinations[j][8]
        patience = int(hyperParameterCombinations[j][9])

        # Build the model.
        layerSizes = getLayerSizes(
            numberHiddenLayers, inputLayerSize, units1, units2, units3)
        dropouts = [dropout1, dropout2, dropout3]
        model = buildModel(layerSizes, dropouts, optimizer, learningRate)
        
        # Fit the model with early stopping.
        early_stopping = EarlyStopping(monitor='loss', patience=patience)
        model.fit(Xtrain, Ytrain,
          nb_epoch = epochs, batch_size = batchSize, verbose = 2,                  
          validation_data=(Xtest, Ytest), callbacks=[early_stopping])
        
        # Test model and save results.
        endTime = time.time() * 1000
        elapsedTime = endTime - startTime
        testModel(
            model, layerSizes, Xtrain, Ytrain, Xtest, Ytest, learningRate,
            epochs, batchSize, optimizer, elapsedTime = elapsedTime,
            printResults = True)

# Save final model as JSON.
    model_json = model.to_json()
    with open("decisionMakers/decisionMaker11/winDefeatPrediction.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5.
    model.save_weights("decisionMakers/decisionMaker11/winDefeatPrediction.h5")
    print("Saved model to disk")
