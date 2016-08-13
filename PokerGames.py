# This script will host poker games to collect data.

from __future__ import division
import numpy as np
import random
from deuces import Card, Evaluator

random.seed()

def setBlinds(dealerPosition, initialNumberPlayers, bets, chips, calls):
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
    pot = bigBlindBet + smallBlindBet    
    # Create list with blind information to return.
    blindInformation = []
    blindInformation.append(smallBlindPosition)
    blindInformation.append(bigBlindPosition)
    blindInformation.append(pot)
    return blindInformation

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

def showHoleCards(playerNames, AIPlayers, playerCards, initialNumberPlayers):
    # Print the hole cards for each player to read.
    for position in range(0, initialNumberPlayers):
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
        # Generate opponent's cards and remaining community cards.
        for j in range(communityCardCount, 5):
            communityCards[j] = dealCard(existingCards)
            existingCards[j + 2] = communityCards[j]
        for j in range(0,2):
            opponentCards[j] = dealCard(existingCards)
            existingCards[7 + j] = opponentCards[j]
        # Put opponent hand and community cards in deuces form.
        opponentHand = setUpDeucesCards(opponentCards)
        board = setUpDeucesCards(communityCards)
        # Evaluate hand ranks.
        evaluator = Evaluator()
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
    dealerPosition, initialNumberPlayers, playernames, playerCards):
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
    roundNumber, dealerPosition, manualDealing, initialNumberPlayers,
    trainingMode, AIPlayers, playerNames, folds, playerCards,
    communityCards, existingCards):
    # Fill card arrays with hole cards.
    for position in range(0,initialNumberPlayers):
        if(not folds[position]):
            for i in range(0,2):
                newCard = dealCard(existingCards)
                playerCards[position][i] = newCard
                existingCards[(2 * position) + i] = newCard
    # Tell players what their cards are.
    showHoleCards(playerNames, AIPlayers, playerCards, initialNumberPlayers)
            
def autoDealRoundTwo(
    initialNumberPlayers, trainingMode, playerCards, communityCards,
    existingCards):
    if(not trainingMode):
        print "The flop cards are:"
    # Generate and store a flop cards.
    for i in range(0,3):
        newCard = dealCard(existingCards)
        communityCards[i] = newCard
        existingCards[(initialNumberPlayers * 2) + i] = newCard
        # Announce the new card.
        if(not trainingMode):
            printCard(newCard)
    if(not trainingMode):
        temp = raw_input("Enter anything to continue\n")
        print "\n"

def autoDealRoundThree(
    initialNumberPlayers, trainingMode, playerCards, communityCards,
    existingCards):
    # Generate and store the turn card.
    if(not trainingMode):
        print "The turn card is:"
    newCard = dealCard(existingCards)
    communityCards[3] = newCard
    existingCards[(initialNumberPlayers * 2) + 3] = newCard
    # Announce the new card.
    if(not trainingMode):
        printCard(newCard)
        temp = raw_input("Enter anything to continue\n")
        print "\n"
    
def autoDealRoundFour(
    initialNumberPlayers, trainingMode, playerCards, communityCards,
    existingCards):
    # Generate and store the river card.
    if(not trainingMode):
        print "The river card is:"
    newCard = dealCard(existingCards)
    communityCards[4] = newCard
    existingCards[(initialNumberPlayers * 2) + 4] = newCard
    # Announce the new card.
    if(not trainingMode):
        printCard(newCard)
        temp = raw_input("Enter anything to continue\n")
        print "\n"

def deal(
    roundNumber, dealerPosition, manualDealing, initialNumberPlayers,
    trainingMode, AIPlayers, playerNames, folds, playerCards,
    communityCards, existingCards):
    # Fills card lists with generated cards between 1 and 52.
    # If manualDealing is True the user is prompted to input cards from
    #a live pack.
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
                roundNumber, dealerPosition, manualDealing,
                initialNumberPlayers, trainingMode, AIPlayers,
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

def getAIBet(callValue, bigBlind, handStrength, chips, position):
    # Use simple betting scheme for now.
    if(handStrength > 0.5):
        newBet = callValue + bigBlind
    elif(handStrength > 0.3):
        newBet = callValue
    else:
        newBet = 0
    # Go all-in if newBet is greater than chip count.
    if(newBet > chips[position]):
        newBet = chips[position]
    return newBet

def checkHumanBetValid(newBetString, chipCount, betsMade, maxBet):
    # Check if new bet is a number
    if(not newBetString.isdigit()):
        print "That wasn't a valid bet\n"
    else:
        newBet = int(float(newBet))
        if(newBet == 0):
            return newBet
        else:
            # Check if player can afford the full call value.
            if(chipCount + betsMade < maxBet):
                if(newBet < chipCount):
                    # Bet is not valid.
                    print "You must bet all your money or fold \n"
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

def getHumanBet(
    position, maxBet, callValue, pot, playerName, chipCount, betsMade,
    trainingMode):
    print(playerName + " has " + str(chipCount) + " chips and has bet "
          + str(betsMade) + " already\n")
    print("The bet to match is " + str(maxBet) + " for a pot of "
          + str(pot) + "\n")
    print(playerName + ", the call value is " + str(callValue))
    inputPrompt = "\nHow much are you betting? \n"
    while(True):
        newBetString = raw_input(inputPrompt)
        newBet = checkHumanBetValid(newBetString, chipCount, betsMade, maxBet)
        if(newBet is not False):
            return newBet

def getBet(
    position, playerNames, AIPlayers, trainingMode, bigBlind,
    handStrength, bets, calls, raises, initialNumberPlayers,
    playersActive):
    # Calculate callValue and maxBet.
    maxBet = np.amax(bets)
    callValue = maxBet - bets[position]
    if(AIPlayers[position]):
        newBet = getAIBet(
            callValue, bigBlind, handStrength, chips, position)
    else:
        newBet = getHumanBet(
            maxBet, callValue, pot, playerNames[position],
            chips[position], bets[position], trainingMode)
    # Final check to see if bet is valid.
    betValidity = (((newBet >= callValue) and (newBet <= chips[position]))
        or (newBet == chips[position]))
    if(not betVaility):
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
    canAffordCall = (chips[position] >= callvalue)
    if(canAffordCall):
        if(newBet == callValue):
            return "call"
        elif(newbet > callValue):
            return "raise"
    else:
        # If they can't afford a call and they went all in.
        if(newbet == chips[position]):
            return "all-in call"
    # If newBet does not satisfy any of the above cases then there is an
    #error. Notify user with raw_input so that script pauses
    raw_input("Error: betActionType cannot determine action")

def postBetUpdate(
    position, newBet, playerNames, chips, bets, calls, raises, active):
    # After a new bet update folds, chips, bets, calls, raises lists.
    # Return a list with new pot and playersActive.
    oldMaxBet = np.amax(bets)
    callValue = maxBet - bets[position]
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
    else:
        # Player calls/raises.
        if(not trainingMode):
            if(betType == "call"):
                print("\n" + playerNames[position] + " has called")
            elif(betType == "raise"):
                print("\n" + playerNames[position] + " has raised")
        calls[position] += maxBet - bets[position]
        raises[position] += bets[position] + newBet - oldMaxBet
        maxbet = bets[position] + newbet
    # Remaining updates are independant of whether player
    #folded/checked/raised/called.
    bets[position] += newBet
    chips[position] -= newBet
    active[position] = True
    pot += newBet
    # Announce new bet if it is a raise/call.
    if(not (hasFolded or hasChecked)):
        if(not trainingMode):
            print(
                "\n" + playerNames[position] + " has bet " + str(newBet)
                + playerNames[position] + " now has "
                + str(chips[position]) + " chips.\n")
    updatedValues = []
    updatedValues.append(playersActive)
    updatedValues.append(maxBet)
    updatedValues.append(pot)
    # Store betType for later analysis/learning
    updatedValues.append(betType)
    return updatedValues

def sortBets(bets, positions):
    # Put bets array in order highest to lower and order positions so
    #that they match their bets
    for i in range(0, len(bets)):
        for j in range(0, len(bets)):
            if(bets[i] > bets[j]):
                temp = bets[i]
                bets[i] = bets[j]
                bets[j] = temp
                temp = positions[i]
                positions[i] = positions[j]
                positions[j] = temp

def adjustHighestBet(bets, chips, raises, maxBet, pot, bigBlind):
    # Find the player who bet the most; if nobody matched their bet then
    #reduce it to the second highest bet and reduce the pot, ect.
    # Put bets in order and sort positions accordingly
    initialNumberPlayers = len(bets)
    tempBets = [0] * initialNumberPlayers
    betPositions = range(0, initialNumberPlayers)
    for position in range (0,initialNumberPlayers):
        tempBets[position] = bets[position]
    sortBets(tempBets, betPositions)
    # If one player bet more than everybody adjust the pot, ect.
    highestBetPosition = betPositions[0]
    nextHighestBetPosition = betPositions[0]
    highestBet = tempbets[0]
    nextHighestbet = tempBets[1]
    if((highestBet > nextHighestbet) and (highestBet > bigBlind)):
       chips[highestBetPosition] += (highestbet - nextHighestBet)
       raises[highestBetPosition] += (highestbet - nextHighestBet)
       bets[highestBetPosition] = bets[nextHighestBetPosition]
       maxBet = nextHighestBet
       pot -= (highestbet - nextHighestBet)
    # Fill list with adjusted info
    adjustedBetInfo = []
    adjustedBetInfo.append(maxBet)
    adjustedBetInfo.append(pot)
    return adjustedBetInfo

def doBetting(
    bigBlind, initialNumberPlayers, chips, bets, raises, calls, folds,
    startPosition, playerNames, cardStrengths, trainingMode):
    # Conduct a round of betting, update chips and return the final
    #position played from.
    # Reset round activity.
    roundActive = True
    
    # active = True indicates that a player has acted this round.
    active = [False] * initialNumberPlayers 
    while(roundActive):
        newBet = 0;
        # Check if the round is over.
        # Betting ends if there is 1 player left or if someone who
        #has acted already has nothing to call.
        if((playersActive == 1) or
            (active[position] and (bets[position] == maxbet))):
            roundActive = False
            position = (position - 1) % initialNumberPlayers
        else:
            if(not folds[position]):
                # Check if player can bet.
                if((chips[position] == 0) and (not trainingMode)):
                    print playerNames[position] + " cannot bet"
                else:
                    handStrength = cardStrengths[position]
                    newBet = getBet(
                        position, playerNames, AIPlayers, trainingMode,
                        bigBlind, handStrength, bets, calls, raises,
                        initialNumberPlayers, playersActive)
            # Update all money lists after new bet is made.
            postBetUpdate(
                position, newBet, playerNames, chips, bets, calls, raises, active)
            playersActive = updatedValues[0]
            maxBet = updatedValues[1]
            pot = updatedValues[2]
            betType = updatedValues[3]
    # Round of betting is finished.
    # Find player who bet the most; if nobody matched their bet then
    #reduce their bet to that of the second highest better.
    adjustedBetInfo = adjustHighestBet(
        bets, chips, raises, maxBet, pot, bigBlind)
    maxBet = adjustedBetInfo[0]
    pot = adjustedBetInfo[1]

def playhand(
    playerNames, initialChips, AIPlayers, bigBlind, dealerPosition,
    manualDealing, trainingMode):
    # playhand takes the players' names and game situation and plays one
    #hand of poker.
    # Set initial values for the game.
    initialNumberPlayers = len(initialChips)
    activePlayerCount = initialNumberPlayers
    chips = np.zeros(initialNumberPlayers)
    bets = np.zeros(initialNumberPlayers)
    calls = np.zeros(initialNumberPlayers)
    raises = np.zeros(initialNumberPlayers)
    for i in range(0,initialNumberPlayers):
        chips[i] = initialChips[i]
    folds = [False] * initialNumberPlayers
    playerCards = np.zeros((2 * initialNumberPlayers,2))
    communityCards = np.zeros(5)
    existingCards = np.zeros((initialNumberPlayers * 2) + 5)
    # Set the blinds.
    blindInfo = setBlinds(dealerPosition, initialNumberPlayers, bets,
        chips, calls)
    # Loop through all rounds of betting.
    for roundNumber in range (1,5):
        # Deal cards for this round and print community cards if not
        #in training mode.
        deal(
        roundNumber, dealerPosition, manualDealing, initialNumberPlayers,
        trainingMode, AIPlayers, playerNames, folds, playerCards,
        communityCards, existingCards)
        # Get the strength of each player's hand.
        cardStrengths = [0] * initialNumberPlayers
        for position in range(0,initialNumberPlayers):
            if(not folds[position]):
                holeCards[0] = playerCards[position][0]
                holeCards[1] = playerCards[position][1]
                cardStrengths[position] = getHandStrength(
                    holeCards, communityCards, roundNumber)
        # Begin the betting.
        doBetting(
        bigBlind, initialNumberPlayers, chips, bets, raises, calls, folds,
        startPosition, playerNames, cardStrengths, trainingMode)

# Play one example game.
playerNames = ["Hugh", "Robin", "Pookey"]
initialChips = [1000,200,50]
AIPlayers = [False, False, False]
bigBlind = 100
dealerPosition = 0
manualDealing = False
trainingMode = False


'''
# Test implementation
playhand(
    playerNames=playerNames, initialChips=initialChips,
    AIPlayers=AIPlayers, bigBlind=100, dealerPosition=dealerPosition,
    manualDealing = manualDealing, trainingMode = trainingMode)
'''
