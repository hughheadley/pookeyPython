# This script will host poker games to collect data.

import numpy as np
import random
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
                # End the testing for uniqueness.
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

def manualDealRoundOne(
    dealerPosition, initialNumberPlayers, playernames, playerCards):
    # Request user input to deal the player's hole cards, nothing to return.
    dealPositionEnd = dealerPosition + initialNumberPlayers + 1
    for i in range (dealerPosition + 1, dealPositionEnd):
        position = i % initialNumberPlayers
        #Get first hole card.
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
                
def playhand(playerNames, initialChips, AIPlayers, bigBlind,
        dealerPosition, manualDealing, trainingMode):
    # playhand takes the players' names and game situation
    #and plays one hand of poker.
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
        # Reset round activity.
        roundActive = True
        # active = True indicates that a player has acted this round.
        active = [False] * initialNumberPlayers 
        # Deal cards for this round.
        deal(
        roundNumber, dealerPosition, manualDealing, initialNumberPlayers,
        trainingMode, AIPlayers, playerNames, folds, playerCards,
        communityCards, existingCards)

# Play one example game.
playerNames = ["Hugh", "Robin", "Pookey"]
initialChips = [1000,200,50]
AIPlayers = [False, False, False]
bigBlind = 100
dealerPosition = 0
manualDealing = False
trainingMode = False

roundNumber = 0
dealerPosition = 0
manualDealing = True
initialNumberPlayers = len(playerNames)
trainingMode = False
folds = [False, False, False]
playerCards = np.zeros((2 * initialNumberPlayers,2))
communityCards = np.zeros(5)
existingCards = np.zeros((initialNumberPlayers * 2) + 5)

for roundNumber in range(1,5):
    deal(
        roundNumber, dealerPosition, manualDealing, initialNumberPlayers,
        trainingMode, AIPlayers, playerNames, folds, playerCards,
        communityCards, existingCards)

'''
playhand(
    playerNames=playerNames, initialChips=initialChips,
    AIPlayers=AIPlayers, bigBlind=100, dealerPosition=dealerPosition,
    manualDealing = manualDealing, trainingMode = trainingMode)
'''
