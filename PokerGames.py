# This script will host poker games to collect data.

import numpy as np

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

def checkCardValidity(cardText):
    # Check if a string is a valid card number.
    # If valid return card number, if invalid return 0
    if(cardText.isdigit()):
        cardNumber = int(float(cardText))
        if((cardNumber > 0) AND (cardNumber < 15)):
            return cardNumber
        else:
            return 0
    elif(any((cardText == "A"), (cardText == "a"), (cardText == "Ace"), (cardText == "ace"), (cardText == "ACE"))):
        return 14
    elif(any((cardText == "K"), (cardText == "k"), (cardText == "King"), (cardText == "king"), (cardText == "KING"))):
        return 13
    elif(any((cardText == "Q"), (cardText == "q"), (cardText == "Queen"), (cardText == "queen"), (cardText == "QUEEN"))):
        return 12
    elif(any((cardText == "J"), (cardText == "j"), (cardText == "Jack"), (cardText == "jack"), (cardText == "JACK"))):
        return 11
    else:
        return 0

def checkSuitValidity(suitText):
    # Check if a string is a valid suit.
    # If valid return suit number as an int, if invalid return 0.
    if(suitText.isdigit()):
        suitumber = int(float(suitText))
        if((suitumber > 0) AND (suitumber < 5)):
            return suitumber
        else:
            return 0
    elif(any((suitText == "c"), (suitText == "C"), (suitText == "club"), (suitText == "clubs"))):
        return 1
    elif(any((suitText == "d"), (suitText == "D"), (suitText == "diamond"), (suitText == "diamonds"))):
        return 2
    elif(any((suitText == "h"), (suitText == "H"), (suitText == "heart"), (suitText == "hearts"))):
        return 3
    elif(any((suitText == "s"), (suitText == "S"), (suitText == "spade"), (suitText == "spades"))):
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
            cardText = raw_input("Enter a valid card number")
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
            suitNumberText = raw_input("Enter a valid suit number")
    return suitNumber

def cardAndSuitToValue(cardNumber, suitNumber):
    # Turn the card and suit number into a value between 1 and 52
    #note that a cardValue of 1 is a two of clubs
    cardValue = cardNumber - 1 + (13 * (suitNumber - 1))
    return cardValue

def manualDealRoundOne(
    dealerPosition, initialNumberPlayers, playernames, playerCards):
    # Request user input to deal the player's hole cards, nothing to return
    dealPositionEnd = dealerPosition + initialNumberPlayers + 1
    for i in range (dealerPosition + 1, dealPositionEnd):
        position = i % initialNumberPlayers
        inputPrompt = "Enter " + playerNames[position] + "'s first card" + "\n"
        cardNumber = getCardNumber(inputPrompt)
        inputPrompt = "Enter " + playerNames[position] + "'s first suit" + "\n"
        cardNumber = getSuitNumber(inputPrompt)
        # Convert the card number and suit to a value between 1 and 52
        cardValue = CardAndSuitToValue(cardNumber, suitNumber)
        playerCards[position][0] = cardValue
        inputPrompt = "Enter " + playerNames[position] + "'s second card" + "\n"
        cardNumber = getCardNumber(inputPrompt)
        inputPrompt = "Enter " + playerNames[position] + "'s second suit" + "\n"
        cardNumber = getSuitNumber(inputPrompt)
        # Convert the card number and suit to a value between 1 and 52
        cardValue = cardAndSuitToValue(cardNumber, suitNumber)
        playerCards[position][1] = cardValue

def manualDealRoundTwo(communityCards):
    # Request user input to deal the flop cards. Nothing to return.
    inputPrompt = "Enter the first flop card"
    flopCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit"
    flopSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(flopCard, flopSuit)
    communityCards[0] = cardValue
    inputPrompt = "Enter the second flop card"
    flopCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit"
    flopSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(flopCard, flopSuit)
    communityCards[1] = cardValue
    inputPrompt = "Enter the third flop card"
    flopCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit"
    flopSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(flopCard, flopSuit)
    communityCards[2] = cardValue

def manualDealRoundThree(communityCards):
    # Request user input to deal the turn card. Nothing to return.
    inputPrompt = "Enter the turn card"
    turnCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit"
    turnSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(turnCard, turnSuit)
    communityCards[3] = cardValue

def manualDealRoundFour(communityCards):
    # Request user input to deal the river card. Nothing to return.
    inputPrompt = "Enter the river card"
    riverCard = getCardNumber(inputPrompt)
    inputPrompt = "Enter that card's suit"
    riverSuit = getSuitNumber(inputPrompt)
    cardValue = cardAndSuitToValue(riverCard, riverSuit)
    communityCards[4] = cardValue

def deal(
    roundNumber, dealerPosition, manualDealing, initialNumberPlayers,
    trainingMode, AIPlayers, playerNames, folds, playerCards,
    communityCards, existingCards):
    # deal fills card lists with generated cards between 1 and 52.
    # If manualDealing is True the user is prompted to input cards from
    #a live pack.
    if(manualDealing):
        if(roundNumber == 1):
            manualDealRoundOne()
        elif(roundNumber == 2):
            manualDealRoundTwo()
        elif(roundNumber == 3):
            manualDealRoundThree()
        elif(roundNumber == 4):
            manualDealRoundFour()
    else:
        #tell human players what their hole cards are
        if(roundNumber == 1):
            if(!trainingMode):
                for i in range(dealerPosition, (initialNumberPlayers + dealerPosition)):
                    position = i % intialNumberPlayers
                    if(!AIPlayers[position]):
                        print playerNames[position] + " your cards are:"
                        showCards(position, playerCards)
                        temp = raw_input("Enter anything to continue")
                        print "\n"
        else:
            dealCommunityCards(
                roundNumber, dealerPosition, manualDealing, numberPlayers,
                trainingMode, AIPlayers, playerNames, folds, playerCards,
                communityCards, existingCards)
                

def playhand(playerNames, initialChips, AIPlayers, bigBlind,
        dealerPosition):
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

    # Set the blinds
    blindInfo = setBlinds(dealerPosition, initialNumberPlayers, bets,
        chips, calls)

    # Loop through all rounds of betting.
    for roundNumber in range (1,5):
        # Reset round activity.
        roundActive = True
        # active = True indicates that a player has acted this round.
        active = [False] * initialNumberPlayers 
        # Deal cards for this round
        deal()

# Play one example game.
playerNames = ["Hugh", "Robin", "Pookey"]
initialChips = [1000,200,50]
AIPlayers = [False, False, False]
bigBlind = 100
dealerPosition = 0

playhand(
    playerNames=playerNames, initialChips=initialChips,
    AIPlayers=aiPlayers, bigBlind=100, dealerPosition=dealerPosition)
