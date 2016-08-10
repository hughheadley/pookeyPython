#This script will run poker games to collect data

import numpy as np

def setBlinds(dealerPosition, initialNumberPlayers, bets, chips, calls):
    smallBlindPosition = (dealerPosition + 1) % initialNumberPlayers
    bigBlindPosition = (dealerPosition + 2) % initialNumberPlayers
    if(chips[smallBlindPosition] >= (bigBlind / 2)):
        #if player can afford the full small blind then bet half the big blind amount
        smallBlindBet = bigBlind / 2
    else:
        #if the player cannot afford the full small blind then go all in
        smallBlindBet = chips[smallBlindPosition]
    if(chips[bigBlindPosition] >= bigBlind):
        #if player can afford the full big blind then bet the big blind amount
        bigBlindBet = bigBlind
    else:
        #if the player cannot afford the full big blind then go all in
        bigBlindBet = chips[bigBlindPosition]

    #update values for blinds set
    bets[smallBlindPosition] = smallBlindBet
    chips[smallBlindPosition] -= smallBlindBet
    calls[smallBlindPosition] = smallBlindBet
    bets[bigBlindPosition] = bigBlindBet
    chips[bigBlindPosition] -= bigBlindBet
    calls[bigBlindPosition] = bigBlindBet
    pot = bigBlindBet + smallBlindBet

    #create list with blind information to return
    blindInformation = []
    blindInformation.append(smallBlindPosition)
    blindInformation.append(bigBlindPosition)
    blindInformation.append(pot)

    return blindInformation
    

def playhand(playerNames, initialChips, aiPlayers, bigBlind, dealerPosition):
    #playhand takes the players' names and initial chips and plays one hand of poker
    #set initial values for the game
    initialNumberPlayers = len(initialChips)
    activePlayerCount = initialNumberPlayers

    chips = np.zeros(initialNumberPlayers)
    bets = np.zeros(initialNumberPlayers)
    calls = np.zeros(initialNumberPlayers) #the amount called over the entire game
    raises = np.zeros(initialNumberPlayers) #the amount raised over the entire game
    for i in range(0,initialNumberPlayers):
        chips[i] = initialChips[i]

    #folds and active are lists of bool values
    folds = [False] * initialNumberPlayers #produces a list with all values False
    active = [True] * initialNumberPlayers #produces a list with all values True

    #initially cards are 0 before dealt. Card values are numbers 1-52
    playerCards = np.zeros((2 * initialNumberPlayers,2))
    communityCards = np.zeros(5)

    #set the blinds
    blindInfo = setBlinds(dealerPosition, initialNumberPlayers, bets, chips, calls)

#play one example game
playerNames = ["Hugh", "Robin", "Pookey"]
initialChips = [1000,200,50]
aiPlayers = [False, False, False]
bigBlind = 100
dealerPosition = 0

playhand(playerNames=playerNames, initialChips=initialChips, aiPlayers=aiPlayers, bigBlind=100, dealerPosition=dealerPosition)
