#This script will run poker games to collect data

import numpy as np

def playhand(playerNames, initialChips, aiPlayers, bigBlind):
    #playhand takes the players' names and initial chips and plays one hand of poker
    #set initial values for the game
    initialNumberPlayers = len(initialChips)
    activePlayerCount = initialNumberPlayers
    pot = 0
    maxBet = bigBlind
    
    bets = np.zeros(initialNumberPlayers)
    calls = np.zeros(initialNumberPlayers) #the amount called over the entire game
    raises = np.zeros(initialNumberPlayers) #the amount raised over the entire game

    #folds and active are lists of bool values
    folds = [False] * initialNumberPlayers #produces a list with all values False
    active = [True] * initialNumberPlayers #produces a list with all values False
    print folds

    #initially cards are 0 before dealt. Card values are numbers 1-52
    playerCards = np.zeros((2 * initialNumberPlayers,2))
    communityCards = np.zeros(5)
    print playerCards
   
    
playerNames = ["Hugh", "Robin", "Pookey"]
initialChips = [1000,2000,2500]
aiPlayers = [False, False, False]
bigBlind = 100

playhand(playerNames = playerNames, initialChips = initialChips, aiPlayers = aiPlayers, bigBlind = 100)
