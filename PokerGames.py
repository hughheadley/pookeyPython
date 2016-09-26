# This script will host poker games to collect data.

from __future__ import division
import numpy as np
import random
import datetime
import AIDecisions
from deuces import Card, Evaluator
import os
import math
import csv

random.seed()
# Create the poker hand evaluator.
evaluator = Evaluator()

def setBlinds(dealerPosition, bets, chips, calls, bigBlind):
    initialNumberPlayers = len(bets)
    smallBlindPosition = (dealerPosition + 1) % initialNumberPlayers
    bigBlindPosition = (dealerPosition + 2) % initialNumberPlayers
    # Check if player can afford the full small blind.
    if(chips[smallBlindPosition] >= (bigBlind / 2)):
        smallBlindBet = bigBlind / 2
    else:
        smallBlindBet = chips[smallBlindPosition] # Go all in.
    # Check if player can afford the full big blind.
    if(chips[bigBlindPosition] >= bigBlind):
        bigBlindBet = bigBlind
    else:
        bigBlindBet = chips[bigBlindPosition] # Go all in.
    # Update values for blinds set.
    bets[smallBlindPosition] = smallBlindBet
    chips[smallBlindPosition] -= smallBlindBet
    calls[smallBlindPosition] = smallBlindBet
    bets[bigBlindPosition] = bigBlindBet
    chips[bigBlindPosition] -= bigBlindBet
    calls[bigBlindPosition] = bigBlindBet

def dealCard(existingCards):
    # Deal one new card which is not in the existingCards.
    uniqueCard = False
    while(not uniqueCard):
        newCard = random.randint(1,52)
        uniqueCard = True
        # Check if card is unique.
        for i in range(0,len(existingCards)):
            if(existingCards[i] == 0):
                # Card is unique, end the testing for uniqueness.
                i = 52
            elif(existingCards[i] == newCard):
                uniqueCard = False
                i = 52
    return newCard

def checkCardValidity(cardText):
    # Check if a string is a valid card number.
    # If valid return card number, if invalid return 0.
    if(cardText.isdigit()):
        cardNumber = int(float(cardText))
        if((cardNumber > 0) and (cardNumber < 15)):
            return cardNumber
        else:
            return 0
    elif(cardText in ["A", "a", "ace", "Ace", "ACE"]):
        return 14
    elif(cardText in ["K", "k", "king", "King", "KING"]):
        return 13
    elif(cardText in ["Q", "q", "queen", "Queend", "QUEEN"]):
        return 12
    elif(cardText in ["J", "j", "jack", "Jack", "JACK"]):
        return 11
    else:
        return 0

def checkSuitValidity(suitText):
    # Check if a string is a valid suit.
    # If valid return suit number as an int, if invalid return 0.
    if(suitText.isdigit()):
        suitNumber = int(float(suitText))
        if((suitNumber > 0) and (suitNumber < 5)):
            return suitNumber
        else:
            return 0
    elif(suitText in ("c", "C", "club", "clubs", "Club", "Clubs")):
        return 1
    elif(suitText in ("d", "D", "diamond", "diamonds", "Diamond", "Diamonds")):
        return 2
    elif(suitText in ("h", "H", "heart", "hearts", "Heart", "Hearts")):
        return 3
    elif(suitText in ("s", "S", "spade", "spades", "Spade", "Spades")):
        return 4
    else:
        return 0

def getCardNumber(inputPrompt):
    # Prompt user to enter a card and return the entered card number.
    cardNumber = 0
    cardText = raw_input(inputPrompt)
    while(True):
        cardNumber = checkCardValidity(cardText)
        if(cardNumber):
            break
        else:
            cardText = raw_input("Enter a valid card\n")
    return cardNumber

def getSuitNumber(inputPrompt):
    # Prompt user to enter a suit and return the entered suit number.
    suitNumber = 0
    suitText = raw_input(inputPrompt)
    while(True):
        suitNumber = checkSuitValidity(suitText)
        if(suitNumber):
            break
        else:
            suitText = raw_input("Enter a valid suit number\n")
    return suitNumber

def cardAndSuitToValue(cardNumber, suitNumber):
    # Turn the card and suit number into a value between 1 and 52.
    # Note that a cardValue of 1 is a two of clubs.
    cardValue = cardNumber - 1 + (13 * (suitNumber - 1))
    return cardValue

def cardIndexToNumber(cardIndex):
    # Convert card index to a card Number and return as a string.
    cardNumber = (cardIndex - 1) % 13 + 2
    if(cardNumber < 11):
        cardNumberString = str(int(cardNumber))
    elif(cardNumber == 11):
        cardNumberString = "J"
    elif(cardNumber == 12):
        cardNumberString = "Q"
    elif(cardNumber == 13):
        cardNumberString = "K"
    elif(cardNumber == 14):
        cardNumberString = "A"
    return cardNumberString

def cardIndexToSuit(cardIndex):
    # Convert card index to its suit symbol and return as a string.
    suitNumber = int((cardIndex - 1) / 13) + 1
    if(suitNumber == 1):
        return u"\u2663"
    elif(suitNumber == 2):
        return u"\u2666"
    elif(suitNumber == 3):
        return u"\u2665"
    elif(suitNumber == 4):
        return u"\u2660"
    
def printCard(cardIndex):
    # Convert the card index to a suit and number.
    suit = cardIndexToSuit(cardIndex)
    cardNumber = cardIndexToNumber(cardIndex)
    print str(cardNumber) + suit

def showHoleCards(position, playerNames, AIPlayers, playerCards):
    # Print the hole cards for one player to read.
    if(not AIPlayers[position]):
        print playerNames[position] + " your cards are:"
        printCard(playerCards[position][0])
        printCard(playerCards[position][1])
        temp = raw_input("Enter anything to continue\n")
        # Print many line breaks to hide cards from the next player.
        for i in range(0,20):
            print "\n"

def getDeucesCardNumber(cardIndex):
    # Takes a card index 1-52 and finds the card number string compatible
    #with the deuces library.
    # Find the card's rank 2-Ace.
    cardNumber = (cardIndex - 1) % 13 + 2
    if(cardNumber < 10):
        cardNumberString = str(int(cardNumber))
    elif(cardNumber == 10):
        cardNumberString = "T"
    elif(cardNumber == 11):
        cardNumberString = "J"
    elif(cardNumber == 12):
        cardNumberString = "Q"
    elif(cardNumber == 13):
        cardNumberString = "K"
    elif(cardNumber == 14):
        cardNumberString = "A"
    return cardNumberString

def getDeucesSuit(cardIndex):
    # Takes a card index 1-52 and finds the suit string compatible with
    #the deuces library.
    suitNumber = int((cardIndex - 1) / 13) + 1
    if(suitNumber == 1):
        cardSuitString = "c"
    elif(suitNumber == 2):
        cardSuitString = "d"
    elif(suitNumber == 3):
        cardSuitString = "h"
    elif(suitNumber == 4):
        cardSuitString = "s"
    return cardSuitString

def convertToDeuces(cardIndex):
    cardNumberString = getDeucesCardNumber(cardIndex)
    cardSuitString = getDeucesSuit(cardIndex)
    # Put card number and suit together.
    cardString = cardNumberString + cardSuitString
    return cardString

def setUpDeucesCards(cardsList):
    # Take a list of cards numbered 1-52 and put them into the form
    #used by deuces evaluator.
    # Convert card numbers to a deuces form string.
    cardStrings = []
    for i in range (0,len(cardsList)):
        cardStrings.append(convertToDeuces(cardsList[i]))
    # Put cards into a deuces form hand.
    deucesCards = []
    for i in range (0,len(cardsList)):
        deucesCards.append(Card.new(cardStrings[i]))
    return deucesCards

def getHandStrength(holeCards, communityCards, roundNumber, sampleSize=400):
    # Calculate the chance of holeCards beating one opponent with random
    #cards.
    # deuces library is used for calculating hand ranks.
    # sampleSize=400 gives accuracy to within 5% for all hands.
    existingCards = np.zeros(9)
    board = [0] * 5
    # Find how many community cards are already played.
    if(roundNumber == 1):
        communityCardCount = 0
    elif(roundNumber == 2):
        communityCardCount = 3
    elif(roundNumber == 3):
        communityCardCount = 4
    elif(roundNumber == 4):
        communityCardCount = 5
    # Convert player hole cards and community cards to the form
    #compatiple with the deuces library.
    myHand = setUpDeucesCards(holeCards)
    # Add my cards and community cards to existing card set.
    existingCards[0] = holeCards[0]
    existingCards[1] = holeCards[1]
    for i in range(0, communityCardCount):
        existingCards[i + 2] = communityCards[i]
    # Test win/loss outcome repeatedly.
    winCount = 0
    opponentCards = [0] * 2
    for i in range(0, sampleSize):
        # Use copy of communityCards for testing
        communityCardsCopy = []
        for j in range(0, communityCardCount):
            communityCardsCopy.append(communityCards[j])
        # Generate opponent's cards and remaining community cards.
        for j in range(communityCardCount, 5):
            communityCardsCopy.append(dealCard(existingCards))
            existingCards[j + 2] = communityCardsCopy[j]
        for j in range(0,2):
            opponentCards[j] = dealCard(existingCards)
            existingCards[j + 7] = opponentCards[j]
        # Put opponent hand and community cards in deuces form.
        opponentHand = setUpDeucesCards(opponentCards)
        board = setUpDeucesCards(communityCardsCopy)
        # Evaluate hand ranks.
        myRank = evaluator.evaluate(board, myHand)
        opponentRank = evaluator.evaluate(board, opponentHand)
        # Compare hand ranks. Lower rank indicates better hand.
        if(myRank < opponentRank):
            winCount += 1
        elif(myRank == opponentRank):
            winCount += 0.5
        # Reset existing cards.
        for j in range(communityCardCount + 2, 9):
            existingCards[j] = 0 
    handStrength = (winCount / sampleSize)
    return handStrength

def manualDealRoundOne(
    dealerPosition, initialNumberPlayers, playerNames, playerCards):
    # Request user input to deal the player's hole cards, nothing to
    #return.
    dealPositionEnd = dealerPosition + initialNumberPlayers + 1
    for i in range (dealerPosition + 1, dealPositionEnd):
        position = i % initialNumberPlayers
        # Get first hole card.
        inputPrompt = "\nEnter " + playerNames[position] + "'s first card\n"
        cardNumber = getCardNumber(inputPrompt)
        inputPrompt = "Enter " + playerNames[position] + "'s first suit\n"
        suitNumber = getSuitNumber(inputPrompt)
        # Convert the card number and suit to a value between 1 and 52.
        cardValue = cardAndSuitToValue(cardNumber, suitNumber)
        playerCards[position][0] = cardValue
        # Get second hole card.
        inputPrompt = "Enter " + playerNames[position] + "'s second card\n"
        cardNumber = getCardNumber(inputPrompt)
        inputPrompt = "Enter " + playerNames[position] + "'s second suit\n"
        suitNumber = getSuitNumber(inputPrompt)
        # Convert the card number and suit to a value between 1 and 52.
        cardValue = cardAndSuitToValue(cardNumber, suitNumber)
        playerCards[position][1] = cardValue

def manualDealRoundTwo(communityCards):
    # Request user input to deal the flop cards. Nothing to return.
    inputPrompt = "\nEnter the first flop card\n"
    flopCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit\n"
    flopSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(flopCard, flopSuit)
    communityCards[0] = cardValue
    inputPrompt = "Enter the second flop card\n"
    flopCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit\n"
    flopSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(flopCard, flopSuit)
    communityCards[1] = cardValue
    inputPrompt = "Enter the third flop card\n"
    flopCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit\n"
    flopSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(flopCard, flopSuit)
    communityCards[2] = cardValue

def manualDealRoundThree(communityCards):
    # Request user input to deal the turn card. Nothing to return.
    inputPrompt = "\nEnter the turn card\n"
    turnCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit\n"
    turnSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(turnCard, turnSuit)
    communityCards[3] = cardValue

def manualDealRoundFour(communityCards):
    # Request user input to deal the river card. Nothing to return.
    inputPrompt = "\nEnter the river card\n"
    riverCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit\n"
    riverSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(riverCard, riverSuit)
    communityCards[4] = cardValue

def autoDealRoundOne(
    dealerPosition, manualDealing,
    trainingMode, AIPlayers, playerNames, folds, playerCards,
    communityCards, existingCards):
    initialNumberPlayers = len(folds)
    # Fill card arrays with hole cards.
    dealingStart = dealerPosition + 1
    dealingEnd = dealerPosition + initialNumberPlayers + 1
    for i in range(dealingStart, dealingEnd):
        position = i % initialNumberPlayers
        if(not folds[position]):
            for j in range(0,2):
                newCard = dealCard(existingCards)
                playerCards[position][j] = newCard
                existingCards[(2 * (i - dealingStart)) + j] = newCard
        # Tell players what their cards are.
        showHoleCards(position, playerNames, AIPlayers, playerCards)
            
def autoDealRoundTwo(
    initialNumberPlayers, trainingMode, playerCards, communityCards,
    existingCards):
    if(not trainingMode):
        print "\nThe flop cards are:"
    # Generate and store a flop cards.
    for i in range(0,3):
        newCard = dealCard(existingCards)
        communityCards[i] = newCard
        existingCards[(initialNumberPlayers * 2) + i] = newCard
        # Announce the new card.
        if(not trainingMode):
            printCard(newCard)
    if(not trainingMode):
        print "\n"
        temp = raw_input("Enter anything to continue\n")
        print "\n"

def autoDealRoundThree(
    initialNumberPlayers, trainingMode, playerCards, communityCards,
    existingCards):
    # Generate and store the turn card.
    if(not trainingMode):
        print "\nThe turn card is:"
    newCard = dealCard(existingCards)
    communityCards[3] = newCard
    existingCards[(initialNumberPlayers * 2) + 3] = newCard
    # Announce the new card.
    if(not trainingMode):
        printCard(newCard)
        print "\n"
        temp = raw_input("Enter anything to continue\n")
        print "\n"
    
def autoDealRoundFour(
    initialNumberPlayers, trainingMode, playerCards, communityCards,
    existingCards):
    # Generate and store the river card.
    if(not trainingMode):
        print "\nThe river card is:"
    newCard = dealCard(existingCards)
    communityCards[4] = newCard
    existingCards[(initialNumberPlayers * 2) + 4] = newCard
    # Announce the new card.
    if(not trainingMode):
        printCard(newCard)
        print "\n"
        temp = raw_input("Enter anything to continue\n")
        print "\n"

def deal(
    roundNumber, dealerPosition, manualDealing, trainingMode, AIPlayers,
    playerNames, folds, playerCards, communityCards, existingCards):
    # Fill card lists with generated cards between 1 and 52.
    # If manualDealing is True the user is prompted to input cards from
    #a live pack.
    initialNumberPlayers = len(folds)
    if(manualDealing):
        if(roundNumber == 1):
            manualDealRoundOne(
                dealerPosition, initialNumberPlayers, playerNames,
                playerCards)
        elif(roundNumber == 2):
            manualDealRoundTwo(communityCards)
        elif(roundNumber == 3):
            manualDealRoundThree(communityCards)
        elif(roundNumber == 4):
            manualDealRoundFour(communityCards)
    else:
        if(roundNumber == 1):
            autoDealRoundOne(
                dealerPosition, manualDealing, trainingMode, AIPlayers,
                playerNames, folds, playerCards, communityCards,
                existingCards)
        elif(roundNumber == 2):
            autoDealRoundTwo(
                initialNumberPlayers, trainingMode, playerCards,
                communityCards, existingCards)
        elif(roundNumber == 3):
            autoDealRoundThree(
                initialNumberPlayers, trainingMode, playerCards,
                communityCards, existingCards)
        elif(roundNumber == 4):
            autoDealRoundFour(
                initialNumberPlayers, trainingMode, playerCards,
                communityCards, existingCards)

def getDecisionType(decisionRefNumber):
    # Open decisionType.txt file in the relevant folder to find the
    #decision method used by this AI player.
    currentPath = os.getcwd()
    subFolderFile = ("decisionMakers/decisionMaker" + str(decisionRefNumber)
    + "/decisionType.txt")
    filePath = os.path.join(currentPath, subFolderFile)    
    with open(filePath,"r") as decisionfile:
        for line in decisionfile:
            decisionType = line.split(None, 1)[0]
    return decisionType

def getAIBet(
    decisionMakerReference, fileNames, position, bigBlind, roundNumber,
    handStrength, folds, chips, bets, raises):
    decisionMethod = getDecisionType(decisionMakerReference)
    newBet = 0
    if(decisionMethod == "simple"):
        newBet = AIDecisions.simpleAIBet(bigBlind, handStrength, bets, position)
    elif(decisionMethod == "geneticNN"):
        newBet = AIDecisions.geneticNNDecision(
            decisionMakerReference, position, handStrength, roundNumber,
            bigBlind, folds, chips, bets, raises)
    else:
        prompt = (
            "Error in PokerGames module in getAIBet function."
            + "Did not find valid decisionType for decision reference "
            + str(decisionMakerReference))
        temp = raw_input(prompt)
    netBet = int(newBet)
    # Go all-in if newBet is greater than chip count.
    # Bet 0 if newBet is less that zero.
    if(newBet > chips[position]):
        newBet = chips[position]
    elif(newBet < 0):
        newBet = 0
    return newBet

def checkHumanBetValid(newBetString, chipCount, betsMade, callValue, maxBet):
    # Check if new bet is a number.
    if(not newBetString.isdigit()):
        print "That isn't a valid bet\n"
    else:
        newBet = int(float(newBetString))
        if(newBet == 0):
            return newBet
        else:
            # Check if player can afford the full call value.
            if(chipCount + betsMade < maxBet):
                if(newBet < chipCount):
                    # Bet is not valid.
                    print "You must bet all your money or fold\n"
                elif(chipCount < newBet):
                    print "That's more than your chip stack\n"
                else:
                    # Bet is valid.
                    return newBet
            else:
                # Check if bet is valid.
                if(((newBet >= callValue) and (newBet <= chipCount))
                    or (newBet == chipCount)):
                    # Bet is valid.
                    return newBet
                else:
                    # Print problem message and request bet again.
                    if((chipCount > newBet) and (newBet != 0)):
                        print "That's not enough\n"
                    if(chipCount < newBet):
                        print "That's more than your chip stack\n"
    return False    

def getHumanBet(position, playerName, chipCount, bets):
    maxBet = np.amax(bets)
    callValue = maxBet - bets[position]
    pot = sum(bets)
    betsMade = bets[position]
    print("\n" + playerName + " has " + str(int(chipCount))
          + " chips and has bet " + str(int(betsMade)) + " already")
    print("The bet to match is " + str(int(maxBet)) + " for a pot of "
          + str(int(pot)))
    print(playerName + ", the call value is " + str(int(callValue)))
    inputPrompt = "\nHow much are you betting? \n"
    while(True):
        newBetString = raw_input(inputPrompt)
        newBet = checkHumanBetValid(
            newBetString, chipCount, betsMade, callValue, maxBet)
        if(newBet is not False):
            return newBet

def getBet(
    decisionRef, fileNames, position, playerNames, AIPlayers, trainingMode,
    bigBlind, roundNumber, handStrength, folds, chips, bets, calls, raises):
    # Calculate values used in decising bet.
    initialNumberPlayers = len(bets)
    maxBet = np.amax(bets)
    callValue = maxBet - bets[position]
    if(AIPlayers[position]):
        newBet = getAIBet(
            decisionRef, fileNames, position, bigBlind, roundNumber,
            handStrength, folds, chips, bets, raises)
    else:
        newBet = getHumanBet(
            position, playerNames[position], chips[position], bets)
    # Final check to see if bet is valid.
    betValidity = (((newBet >= callValue) and (newBet <= chips[position]))
        or (newBet == chips[position]) or (newBet == 0))
    if(not betValidity):
        raw_input("Error invalid bet made")
    return newBet

def betActionType(newBet, position, chips, bets):
    maxBet = np.amax(bets)
    callValue = maxBet - bets[position]
    hasFolded = ((newBet == 0) and (callValue > 0)
        and (chips[position] > 0))
    if(hasFolded):
        return "fold"
    hasChecked = ((newBet == 0) and (maxBet == bets[position])
        and (chips[position] > 0))
    if(hasChecked):
        return "check"
    canAffordCall = (chips[position] >= callValue)
    if(canAffordCall):
        if(newBet == callValue):
            return "call"
        elif(newBet > callValue):
            return "raise"
    else:
        # If they can't afford a call and they went all in.
        if((chips[position] == 0) and (newBet == 0)):
            return "cannot bet"
        elif(newBet == chips[position]):
            return "all-in call"
    # If newBet does not satisfy any of the above cases then there is an
    #error. Notify user with raw_input so that script pauses.
    raw_input("Error: betActionType cannot determine action")

def postBetUpdate(
    position, newBet, playerNames, chips, bets, calls, raises, active, folds,
    trainingMode):
    # After a new bet update folds, chips, bets, calls, raises lists.
    # Return a list with new pot, playersActive and the type of bet made.
    initialNumberPlayers = len(bets)
    playersActive = initialNumberPlayers - sum(folds)
    oldMaxBet = np.amax(bets)
    callValue = oldMaxBet - bets[position]
    newMaxBet = 0
    betType = betActionType(newBet, position, chips, bets)
    if(betType == "fold"):
        playersActive -= 1
        folds[position] = True
        if(not trainingMode):
            print("\n" + playerNames[position] + " has folded")
    elif(betType == "check"):
        if(not trainingMode):
            print("\n" + playerNames[position] + " has checked")
    elif(betType == "all-in call"):
        calls[position] += newBet
        if(not trainingMode):
            print("\n" + playerNames[position] + " has called")
    elif(betType == "cannot bet"):
        # Do nothing.
        temp = 0
    else:
        # Player calls/raises.
        if(not trainingMode):
            if(betType == "call"):
                print("\n" + playerNames[position] + " has called")
            elif(betType == "raise"):
                print("\n" + playerNames[position] + " has raised")
        calls[position] += oldMaxBet - bets[position]
        raises[position] += bets[position] + newBet - oldMaxBet
        newMaxBet = bets[position] + newBet
    # Remaining updates are independant of whether player
    #folded/checked/raised/called.
    bets[position] += newBet
    chips[position] -= newBet
    active[position] = True
    pot = sum(bets)
    # Announce new bet if it is a raise/call.
    if(betType in ["bet","call"]):
        if(not trainingMode):
            print(
                playerNames[position] + " has bet " + str(int(newBet))
                + ", " + playerNames[position] + " now has "
                + str(int(chips[position])) + " chips.\n")
    # Count the number of players still active.
    updatedValues = []
    updatedValues.append(playersActive)
    # Store betType for later analysis/learning.
    updatedValues.append(betType)
    return updatedValues

def sortBets(bets, positions):
    # Put bets array in order highest to lower and order positions so
    #that they match their bets.
    for i in range(0, len(bets)):
        for j in range(0, len(bets)):
            if(bets[i] > bets[j]):
                temp = bets[i]
                bets[i] = bets[j]
                bets[j] = temp
                temp = positions[i]
                positions[i] = positions[j]
                positions[j] = temp

def adjustHighestBet(bets, chips, raises, bigBlind):
    # Find the player who bet the most; if nobody matched their bet then
    #reduce it to the second highest bet and reduce the raises, ect.
    # Put bets in order and sort positions accordingly.
    initialNumberPlayers = len(bets)
    maxBet = np.amax(bets)
    tempBets = [0] * initialNumberPlayers
    betPositions = range(0, initialNumberPlayers)
    for position in range (0,initialNumberPlayers):
        tempBets[position] = bets[position]
    sortBets(tempBets, betPositions)
    # If one player bet more than everybody adjust the pot, ect.
    highestBetPosition = betPositions[0]
    nextHighestBetPosition = betPositions[0]
    highestBet = tempBets[0]
    nextHighestBet = tempBets[1]
    if(nextHighestBet < bigBlind):
        nextHighestBet = bigBlind
    betDifference = (highestBet - nextHighestBet)
    if(highestBet > nextHighestBet):
       chips[highestBetPosition] += betDifference
       raises[highestBetPosition] -= betDifference
       bets[highestBetPosition] = nextHighestBet

def randomBet(bigBlind, position, chips, bets, callChance = 0.3):
    # Call or raise at random.
    maxBet = np.amax(bets)
    betsMade = bets[position]
    callValue = maxBet - betsMade
    chipCount = chips[position]
    # Decide if calling or raising.
    randomNumber = np.random.random()
    if(randomNumber < callChance):
        # Call the bet.
        if(chipCount < callValue):
            newBet = chipCount
        else:
            newBet = callValue
    else:
        # Make a raise between the Big blind and maxRaise.
        if(chipCount < (maxBet + bigBlind)):
            # Minimum raise is big Blind.
            newBet = chipCount
        else:
            minRaise = maxBet + bigBlind
            maxRaise = chipCount
            randNumber = np.random.random()
            raiseLogMin = math.log(minRaise)
            raiseLogMax = math.log(maxRaise)
            logRandomBet = (raiseLogMin +
                (randNumber * (raiseLogMax - raiseLogMin)))
            newBet = int(2.71828 ** logRandomBet)
    return newBet

def recordGameState(
    position, newBet, bigBlind, roundNumber, chips, bets, raises, calls,
    folds, cardStrength, holeCards, communityCards,
    betRecordFile = "betRecords.csv"):
    # Record the state of the game in a csv file.
    # Calculate extra parameters to be recorded.
    initialNumberPlayers = len(folds)
    # Create a list gameState to contain all information.
    gameState = []
    # First entry is reserved for profit.
    gameState.append(0)
    gameState.append(newBet / bigBlind)
    gameState.append(initialNumberPlayers)
    gameState.append(position)
    gameState.append(roundNumber)
    gameState.append(cardStrength)
    gameState.extend(holeCards)
    gameState.extend(communityCards)
    gameState.extend(x / bigBlind for x in chips)
    gameState.extend(x / bigBlind for x in bets)
    gameState.extend(x / bigBlind for x in raises)
    gameState.extend(x / bigBlind for x in calls)
    gameState.extend(x * 1 for x in folds)
    # Append list as a new line in csv.
    with open(betRecordFile, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(gameState)

def recordProfit(profit, bigBlind, betRecordFile = "betRecords.csv"):
    # Record profit in the final line of the csv file.
    # Open csv as a 2D list.
    csvData = csv.reader(open(betRecordFile))
    lines = [l for l in csvData]
    # Modify profit value of the final line.
    lines[len(lines) -  1][0] = (profit / bigBlind)
    # Save modified list as csv file.
    writer = csv.writer(open(betRecordFile, 'wb'))
    writer.writerows(lines)

def doBetting(
    trainingMode, bigBlind, roundNumber, chips, bets, raises, calls, folds,
    startPosition, playerNames, cardStrengths, AIPlayers, playerCards,
    communityCards, decisionRefs = [], fileNames = [], actionCount = False,
    actionToRecord = False, betRecordFile = "betRecords.csv",
    callChance = 0.3):
    # Conduct a round of betting, update chips and return the final
    #position played from.
    # If actionCount reaches actionToRecord then record the bet.
    recordedPosition = False
    initialNumberPlayers = len(bets)
    playersActive = initialNumberPlayers - sum(folds)
    # Reset round activity.
    roundActive = True    
    # active = True indicates that a player has acted this round.
    active = [False] * initialNumberPlayers
    position = startPosition
    while(roundActive):
        newBet = 0;
        maxBet = np.amax(bets)
        # Check if the round is over.
        # Betting ends if there is 1 player left or if someone who
        #has acted already has nothing to call.
        if((playersActive == 1) or
            (active[position] and (bets[position] == maxBet))):
            roundActive = False
        else:
            if(not folds[position]):
                # Check if player can bet.
                if(chips[position] == 0):
                    if(not trainingMode):
                        print playerNames[position] + " cannot bet"
                else:
                    handStrength = cardStrengths[position]
                    if(len(decisionRefs) <= position):
                        decisionRef = 0
                    else:
                        decisionRef = decisionRefs[position]
                    newBet = getBet(
                        decisionRef, fileNames, position, playerNames,
                        AIPlayers, trainingMode, bigBlind, roundNumber,
                        handStrength, folds, chips, bets, calls, raises)
                    newBet = int(newBet)
                    # If recording this bet then make the bet random,
                    #overwrie old bet made and record game state.
                    if(actionToRecord is not False):
                        actionCount += 1
                        if(actionCount == actionToRecord):
                            newBet = randomBet(
                                bigBlind, position, chips, bets,
                                callChance = callChance)
                            newBet = int(newBet)
                            recordedPosition = position
                            holeCards = [0] * 2
                            holeCards[0] = playerCards[position][0]
                            holeCards[1] = playerCards[position][1]
                            recordGameState(
                                position, newBet, bigBlind, roundNumber, chips,
                                bets, raises, calls, folds,
                                cardStrengths[position], holeCards,
                                communityCards, betRecordFile = betRecordFile)
            # Update all money lists after new bet is made.
            updatedValues = postBetUpdate(
                position, newBet, playerNames, chips, bets, calls, raises,
                active, folds, trainingMode)
            playersActive = updatedValues[0]
            betType = updatedValues[1]
            position = (position + 1) % initialNumberPlayers
    # Round of betting is finished.
    # Find player who bet the most; if nobody matched their bet then
    #reduce their bet to that of the second highest better.
    adjustHighestBet(bets, chips, raises, bigBlind)
    # Return the position where betting will continue next round.
    betInfo = []
    betInfo.append(position)
    betInfo.append(actionCount)
    betInfo.append(recordedPosition)
    return betInfo

def sortHandRanks(handRanks):    
    handRankPositions = range(0,len(handRanks))
    # Put handRanks list in order lowest to highest and order positions
    #so that they match their handRanks.
    # A lower hand rank indicates a better hand.
    for i in range(0, len(handRanks)):
        for j in range(0, len(handRanks)):
            if(handRanks[i] < handRanks[j]):
                temp = handRanks[i]
                handRanks[i] = handRanks[j]
                handRanks[j] = temp
                temp = handRankPositions[i]
                handRankPositions[i] = handRankPositions[j]
                handRankPositions[j] = temp
    return handRankPositions

def rankPositions(playerCards, communityCards, folds, handRanks):
    # Find hand ranks and put players in order of best hand to worst.
    initialNumberPlayers = len(folds)
    holeCards = [0] * 2
    foldedHandRank = 10000 # A score worse than any non-folded outcome.
    board = setUpDeucesCards(communityCards)
    evaluator = Evaluator()
    for position in range(0, initialNumberPlayers):
        if(not folds[position]):
            holeCards[0] = playerCards[position][0]
            holeCards[1] = playerCards[position][1]
            hand = setUpDeucesCards(holeCards)
            # Evaluate hand rank.
            handRanks[position] = evaluator.evaluate(board, hand)
        else:
            handRanks[position] = foldedHandRank
    # Sort positions by hand ranks lowest-highest.
    handRankPositions = sortHandRanks(handRanks)
    return handRankPositions

def giveTiedWinnings(firstTieIndex, bets, chips, handScores, winnerPositions):
    initialNumberPlayers = len(bets)
    # A lower handScore is a better hand.
    # Find how many players have equal hands.
    tiedPlayerCount = 0
    winningHandScore = handScores[firstTieIndex]
    for i in range(firstTieIndex, initialNumberPlayers):
        if(handScores[i] == winningHandScore):
            tiedPlayerCount += 1
    # Find smallest bet by a tied winner.
    minBetHandScoreRank = firstTieIndex
    minBetPosition = winnerPositions[minBetHandScoreRank]
    minBet = bets[minBetPosition]
    for i in range(firstTieIndex, firstTieIndex + tiedPlayerCount):
        if(bets[winnerPositions[i]] < minBet):
            minBetHandScoreRank = i
            minBetPosition = winnerPositions[i]
            minBet = bets[minBetPosition]
    # In the ordered winner positions array swap the first tied player
    #with the minBet tied player.
    winnerPositions[minBetHandScoreRank] = winnerPositions[firstTieIndex]
    winnerPositions[firstTieIndex] = minBetPosition 
    # Now sum up all the chips owed to player in firstTieIndex.
    # Loop through all players to take winnings from.
    winnerBet = bets[minBetPosition]
    sumWinnings = 0
    for i in range(0, initialNumberPlayers):
        # If the winner has a higher bet then take all of the loser's
        #money.
        if(winnerBet >= bets[i]):
            sumWinnings += bets[i]
            bets[i] = 0
        else:
            sumWinnings += winnerBet
            bets[i] -= winnerBet
    # Split the totaled chips between the tied winners' chip stacks.
    splitChips = int(sumWinnings / tiedPlayerCount)
    for i in range(firstTieIndex, firstTieIndex + tiedPlayerCount):
        chips[winnerPositions[i]] += splitChips
    # Chips missed from rounding error are given to the first tied
    #player.
    roundingErrorChips = sumWinnings - (splitChips * tiedPlayerCount)
    chips[winnerPositions[firstTieIndex]] += roundingErrorChips

def giveWinnings(
    chips, bets, folds, playerNames, playerCards, communityCards,
    trainingMode):
    initialNumberPlayers = len(bets)
    # Print all players' cards.
    if(not trainingMode):
        for position in range(0, initialNumberPlayers):
            if(not folds[position]):
                print "\n" + playerNames[position] + "'s cards are"
                for i in range(0,2):
                    suitTemp = cardIndexToSuit(playerCards[position][i])
                    cardNumberTemp = cardIndexToNumber(
                        playerCards[position][i])
                    print suitTemp + cardNumberTemp
    # Find the position of the winner(s) and their hand scores (lower
    #scores are better).
    handScores = [0] * initialNumberPlayers
    winnerPositions = rankPositions(
        playerCards, communityCards, folds, handScores)
    # Loop through all players to collect their winnings.
    for i in range(0, initialNumberPlayers - sum(folds) - 1):
        if(not folds[winnerPositions[i]]):
            sumWinnings = 0
            winnerBet = bets[winnerPositions[i]]
            if(not trainingMode):
                if(winnerBet > 0):
                    print "The",
                    if(i == 0):
                        print "first",
                    else:
                        print "next",
                    print "winner is " + playerNames[winnerPositions[i]] + "\n"
            # Test if it is a split pot.
            thisWinnerHandScore = handScores[i]
            nextWinnerHandScore = handScores[i + 1]
            if(thisWinnerHandScore == nextWinnerHandScore):
                # Split pot.
                # Share the winnings between the tied players.
                giveTiedWinnings(i, bets, chips, handScores, winnerPositions)
            else:
                # Loop through all players to take winnings from.
                for j in range(0, initialNumberPlayers):
                    # If the winner has a higher bet then take all of
                    #the loser's money.
                    if(winnerBet >= bets[j]):
                        sumWinnings += bets[j]
                        bets[j] = 0
                    else:
                        sumWinnings += winnerBet
                        bets[j] -= winnerBet
                # Add the totaled chips to the winner's chip stack.
                chips[winnerPositions[i]] += sumWinnings

def selectActionNumber(initialNumberPlayers):
    # Pick a random move/action based on the expected number of moves
    #that are made in a game.
    # Number of actions made in a game is approx 2.5 for every player.
    actionNumber = -1
    meanActions = -0.73 + (2.5 * initialNumberPlayers)
    stDevActions = -0.52 + (1.43 * initialNumberPlayers)
    while(actionNumber < 1):
        # Minimum actionNumber is 1.
        actionNumber = int(random.gauss(meanActions, stDevActions) + 0.5)
    return actionNumber

def playhand(
    playerNames, initialChips, bigBlind, dealerPosition,
    manualDealing, trainingMode, AIPlayers, decisionRefs = [], fileNames = [],
    recordBets = False, betRecordFile = "betRecords.csv"):
    # playhand takes the players' details and starting chips and plays
    #one hand of poker.
    # If recordBets is True then choose a random time to make a player's
    #bet random and record their resulting profit.
    # Set initial values for the game.
    initialNumberPlayers = len(initialChips)
    activePlayerCount = initialNumberPlayers
    bets = np.zeros(initialNumberPlayers)
    calls = np.zeros(initialNumberPlayers)
    raises = np.zeros(initialNumberPlayers)
    chips = np.zeros(initialNumberPlayers)
    for i in range(0,initialNumberPlayers):
        chips[i] = initialChips[i]
    folds = [False] * initialNumberPlayers
    playerCards = np.zeros((2 * initialNumberPlayers,2))
    communityCards = np.zeros(5)
    existingCards = np.zeros((initialNumberPlayers * 2) + 5)
    # Set the blinds.
    setBlinds(dealerPosition, bets, chips, calls, bigBlind)
    # Set position to start play from.
    actionPosition = (dealerPosition + 3) % initialNumberPlayers
    # Create necessary variables for recording bets
    actionCount = 0
    recordedPosition = False
    storedBetPosition = False
    if(recordBets):
        # Choose a random action on which to record game state and
        #new Bet.
        recordActionNumber = selectActionNumber(initialNumberPlayers)
    else:
        recordActionNumber = False
    # Loop through all rounds of betting.
    for roundNumber in range (1,5):
        # Deal cards for this round and print community cards if not
        #in training mode.
        deal(
        roundNumber, dealerPosition, manualDealing, trainingMode, AIPlayers,
        playerNames, folds, playerCards, communityCards, existingCards)
        # Get the strength of each player's hand.
        cardStrengths = [0] * initialNumberPlayers
        holeCards = [0] * 2
        for position in range(0,initialNumberPlayers):
            # Only check card strength if a player is not human, is not
            #folded, and has chips to bet
            checkCardStrength = ((not folds[position]) and (AIPlayers[position])
                and (chips[position] > 0))
            if(checkCardStrength):
                holeCards[0] = playerCards[position][0]
                holeCards[1] = playerCards[position][1]
                cardStrengths[position] = getHandStrength(
                    holeCards, communityCards, roundNumber)
        # Begin the betting.
        bettingInfo = doBetting(
            trainingMode, bigBlind, roundNumber, chips, bets, raises, calls,
            folds, actionPosition, playerNames, cardStrengths, AIPlayers,
            playerCards, communityCards, decisionRefs = decisionRefs,
            fileNames = fileNames, actionCount = actionCount,
            actionToRecord = recordActionNumber, betRecordFile = betRecordFile)
        actionPosition = bettingInfo[0]
        actionCount = bettingInfo[1]
        recordedPosition = bettingInfo[2]
        if(recordedPosition is not False):
            storedBetPosition = recordedPosition
    # Evaluate winners and give chips to winners.
    giveWinnings(
        chips, bets, folds, playerNames, playerCards, communityCards,
        trainingMode)
    # Record actionCount.
    actionInfo = []
    actionInfo.append(initialNumberPlayers)
    actionInfo.append(actionCount)
    with open("actionCounts.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(actionInfo)
    # Record profit for training data.
    if(storedBetPosition is not False):
        profit = (chips[storedBetPosition] - initialChips[storedBetPosition])
        recordProfit(profit, bigBlind, betRecordFile = betRecordFile)
    return chips
