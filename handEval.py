from __future__ import division
import copy
import numpy as np
import random
from deuces import Card, Evaluator
import scipy
from scipy.stats import norm

random.seed()
# Create the poker hand evaluator.
evaluator = Evaluator()

def get_suit_values(cards):
    # Take an array of card indices 1-52 and return suit numbers 1-4.
    numberCards = len(cards)
    suitVals = []
    for i in range(numberCards):
        cardIndex = cards[i]
        suitVals.append(int((cardIndex-1)/13) + 1)
    return suitVals

def get_card_values(cards):
    # Take an array of card indices  1-52 and return card numbers 2-14.
    numberCards = len(cards)
    cardVals = []
    for i in range(numberCards):
        cardIndex = cards[i]
        cardVals.append((cardIndex-1)%13 + 2)
    return cardVals

def dealCard(existingCards):
    # Deal one new card which is not in the existingCards.
    uniqueCard = False
    while not uniqueCard:
        newCard = random.randint(1,52)
        uniqueCard = True
        # Check if card is unique.
        for i in range(len(existingCards)):
            if existingCards[i] == 0:
                # Card is unique, end the testing for uniqueness.
                break
            elif existingCards[i] == newCard:
                uniqueCard = False
                break
    return newCard

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

def sort_cards(cards, suits):
    # Sort cards in descending order, sort the suits accordingly.
    numberCards = len(cards)
    for i in range(numberCards):
        for j in range(numberCards):
            if cards[i] > cards[j]:
                # Swap position of cards and suits.
                temp = cards[i]
                cards[i] = cards[j]
                cards[j] = temp
                temp = suits[i]
                suits[i] = suits[j]
                suits[j] = temp

def add_high_cards(cards, handScore, decimalsAlreadyUsed):
    # Add high cards to the handScore.
    for i in range(5):
        handScore += cards[i] * (0.1**(decimalsAlreadyUsed + 2*(i+1)))
    return handScore

def check_high_card(cards):
    # Calculate the hand score for a high card.
    handScore = 1
    # Add values of high cards as the following decimals.
    for i in range(5):
        handScore += cards[i] * (0.01**(i+1))
    return handScore

def check_pair(cards):
    # Check if there is a pair in cards and return hand score.
    handScore = 0
    for i in range(6):
        if cards[i] == cards[i+1]:
            # Indicate value pair.
            handScore = 2 + cards[i]/100

            # Add first three high cards not used in pair.
            highCardsFound = 0
            for j in range(5):
                if cards[j] != cards[i]:
                    highCardsFound += 1
                    handScore += cards[j] * (0.01**(highCardsFound+1))
                if highCardsFound == 3:
                    break
            break
    return handScore

def check_two_pair(cards):
    # Check if there is a two-pair in cards and return hand score.
    handScore = 0
    twoPairFound = False
    for i in range(4):
        # Search for first pair.
        if cards[i] == cards[i+1]:
            for j in range(i+2, 6):
                # Search for second pair.
                if cards[j] == cards[j+1]:
                    twoPairFound = True
                    handScore = 3 + (cards[i]*0.01) + (cards[j]*0.0001)
                    # Add first high card not used in two pair.
                    for k in range(5):
                        cardUnique = (cards[k] != cards[i]) and (cards[k] != cards[j])
                        if cardUnique:
                            handScore += (cards[k]/1000000)
                            break
                    break
        if(twoPairFound):
            break
    return handScore

def check_three_of_a_kind(cards):
    # Check if there is a three of a kind in cards and return hand score.
    handScore = 0
    tripleFound = False
    for i in range(5):
        if cards[i] == cards[i+1] == cards[i+2]:
            handScore = 4 + cards[i]*0.01
            # Add first two high cards not used in triple.
            highCardsFound = 0
            for j in range(5):
                if cards[j] != cards[i]:
                    highCardsFound += 1
                    handScore += cards[j] * (0.01**(highCardsFound+1))
                if highCardsFound == 2:
                    break
            break
    return handScore

def check_straight(cards):
    # Check if there is a straight in cards and return hand score.
    handScore = 0
    straightFound = False
    for i in range(3):
        topCard = cards[i]
        straightFound = True
        # Check if all sequential cards are held.
        for j in range(1,5):
            if (topCard-j) not in cards:
                straightFound = False
                break
        if straightFound:
            handScore = 5 + (topCard/100)
            break
    return handScore

def check_flush(cards, suits):
    # Check if there is a flush in cards and return hand score.
    handScore = 0
    suitCounts = [0]*4
    suitValues = range(1,5)
    
    # Count the number of each suit.
    for i in range(7):
        suitIndex = suitValues.index(suits[i])
        suitCounts[suitIndex] += 1

    # If there is a flush add high cards.
    if max(suitCounts) >= 5:
        flushSuit = 1 + suitCounts.index(max(suitCounts))
        handScore = 6
        highCardsAdded = 0
        for i in range(7):
            if suits[i] == flushSuit:
                handScore += (cards[i]/100) * (0.01**(highCardsAdded))
                highCardsAdded += 1
    return handScore

def check_full_house(cards):
    # Check if there is a full house in cards and return hand score.
    handScore = 0
    fullHouseFound = False
    # Search for triple.
    tripleScore = check_three_of_a_kind(cards)
    if tripleScore > 0:
        # Card val for triple is in the first two decimals.
        tripleCardVal = int(tripleScore*100) - 100*int(tripleScore)
        for j in range(6):
            # Search for pair distinct from triple.
            if cards[j] != tripleCardVal:
                if cards[j] == cards[j+1]:
                    fullHouseFound = True
                    handScore = 7 + (tripleCardVal/100) + (cards[j]/10000)
            if fullHouseFound:
                break
    return handScore

def check_four_of_a_kind(cards):
    # Check if there is a four of a kind in cards and return hand score.
    handScore = 0
    for i in range(4):
        if cards[i] == cards[i+1] == cards[i+2] == cards[i+3]:
            handScore = 8 + cards[i]*0.01
            # Add first high card not used in quad.
            for j in range(5):
                if cards[j] != cards[i]:
                    handScore += (cards[j]/10000)
                    break
            break
    return handScore

def check_straight_flush(cards, suits):
    # Check if there is a four of a kind in cards and return hand score.
    handScore = 0
    straightFlushFound = False
    for i in range(3):
        topCard = cards[i]
        topSuit = suits[i]
        for j1 in range(i+1, 4):
            cardCorrect = (cards[j1] == (topCard-1)) and (suits[j1] == topSuit)
            if cardCorrect:
                straightFlushFound = True
                break
        if straightFlushFound:
            # Reset straightFlushFound to False and prove that straight
            #flush continues.
            straightFlushFound = False
            for j2 in range(j1+1, 5):
                cardCorrect = (cards[j2] == (topCard-2)) and (suits[j2] == topSuit)
                if cardCorrect:
                    straightFlushFound = True
                    break
        if straightFlushFound:
            # Reset straightFlushFound to False and prove that straight
            #flush continues.
            straightFlushFound = False
            for j3 in range(j2+1, 6):
                cardCorrect = (cards[j3] == (topCard-3)) and (suits[j3] == topSuit)
                if cardCorrect:
                    straightFlushFound = True
                    break
        if straightFlushFound:
            # Reset straightFlushFound to False and prove that straight
            #flush continues.
            straightFlushFound = False
            for j4 in range(j3+1, 7):
                cardCorrect = (cards[j4] == (topCard-4)) and (suits[j4] == topSuit)
                if cardCorrect:
                    straightFlushFound = True
                    break
        if straightFlushFound:
            break
    if straightFlushFound:
        handScore = 9 + (topCard/100)
    return handScore

def get_hand_score(cards):
    # Take seven cards valued 1-52 and return a score for their hand.
    # Higher hand score is better.
    # Get the values of cards 1-12 and suits 1-4.
    cardValues = get_card_values(cards)
    suitValues = get_suit_values(cards)
    
    # Put cards in descending order and sort suits accordingly.
    sort_cards(cardValues, suitValues)

    # Calculate the max frequency of a card value.
    cardValCounts = [0]*13
    for i in range(len(cards)):
        cardValCounts[int(cardValues[i])-2] += 1
    maxCount = max(cardValCounts)

    # Check for card combinations best-worst until one is found.
    """
    handScore = 0
    handScore = check_straight_flush(cardValues, suitValues)
    if handScore == 0:
        handScore = check_four_of_a_kind(cardValues)
    if handScore == 0:
        handScore = check_full_house(cardValues)
    if handScore == 0:
        handScore = check_flush(cardValues, suitValues)
    if handScore == 0:
        handScore = check_straight(cardValues)
    if handScore == 0:
        handScore = check_three_of_a_kind(cardValues)
    if handScore == 0:
        handScore = check_two_pair(cardValues)
    if handScore == 0:
        handScore = check_pair(cardValues)
    if handScore == 0:
        handScore = check_high_card(cardValues)
    """

    handScore = 0
    if maxCount == 4:
        handScore = check_four_of_a_kind(cardValues)
        
    if maxCount == 3:
        handScore = check_three_of_a_kind(cardValues)
        if handScore > 0:
            fullHouseScore = check_full_house(cardValues)
            if fullHouseScore == 0:
                straightFlushScore = check_straight_flush(cardValues, suitValues)
                if straightFlushScore > 0:
                    handScore == straightFlushScore
            else:
                handScore = fullHouseScore
        if handScore < 6:
            handScore = check_flush(cardValues, suitValues)
        if handScore < 5:
            handScore = check_straight(cardValues)

    if maxCount == 2:
        handScore = check_straight_flush(cardValues, suitValues)
        if handScore == 0:
            handScore = check_flush(cardValues, suitValues)
        if handScore == 0:
            handScore = check_straight(cardValues)
        if handScore == 0:
            handScore = check_two_pair(cardValues)
        if handScore == 0:
            handScore = check_pair(cardValues)
    if maxCount == 1:
        handScore = check_straight_flush(cardValues, suitValues)
        if handScore == 0:
            handScore = check_flush(cardValues, suitValues)
        if handScore == 0:
            handScore = check_straight(cardValues)
        if handScore == 0:
            handScore = check_high_card(cardValues)
    
    return handScore;

def get_existing_cards(roundNumber, holeCards, communityCards):
    # Put hole cards and communityCards into an 
    existingCards = np.zeros(9)
    # Find how many community cards are already played.
    roundBoardCards = [0,3,4,5]
    communityCardCount = roundBoardCards[roundNumber-1]
    # Add my cards and community cards to existing card set.
    existingCards = np.append(holeCards, communityCards)
    for i in range(communityCardCount):
        existingCards[i+2] = communityCards[i]
    return existingCards

def wilson_score_samples(proportion, confidence, confRange):
    # Use the Wilson-Score interval to find the sample size required to
    #get a confidence interval for the defined proportion with a defined
    #range.
    ZScore = norm.ppf(0.5 + (confidence/2))

    # Wilson-score gives a quadratic equation to be solved for the
    #sample size. Use quadratic formula with coefficients a,b,c.
    a = 1
    b = (2*(ZScore**2) * (confRange - 2*proportion*(1-proportion))) / confRange
    c = -1*(ZScore**4)*(1-confRange)/confRange
    samples = (-b + ((b**2 - 4*a*c)**0.5)) / (2*a)
    samples = int(samples)
    return samples

def get_deuces_hand_strength(
    holeCards, communityCards, roundNumber, confidence=0.95, confRange=0.1):
    # Calculate the chance of holeCards beating one opponent with random
    #cards.
    # deuces library is used for calculating hand ranks.
    
    # Convert player hole cards and community cards to the form
    #compatiple with the deuces library.
    myHand = setUpDeucesCards(holeCards)
    existingCards = np.append(holeCards, communityCards)
    # Add two spaces for opponent's cards.
    existingCards = np.append(existingCards, [0,0])
    
    roundBoardCards = [0,3,4,5]
    communityCardCount = roundBoardCards[roundNumber-1]

    # Test win/loss outcome repeatedly.
    winCount = 0
    opponentCards = [0]*2
    samples = 0
    # Best case scenario for sampling occurs when proportion=0.
    wilsonSamples = wilson_score_samples(0, confidence, confRange)
    # Round up to nearest 50 samples.
    requiredSamples = int(1 + (wilsonSamples/50))*50
    
    while(samples < requiredSamples):
        # Use copy of communityCards for testing.
        communityCardsCopy = copy.copy(communityCards)
        
        # Generate opponent's cards and remaining community cards.
        for j in range(communityCardCount, 5):
            communityCardsCopy[j] = dealCard(existingCards)
            existingCards[j+2] = communityCardsCopy[j]
        for j in range(2):
            opponentCards[j] = dealCard(existingCards)
            existingCards[j+7] = opponentCards[j]
            
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
        for j in range(communityCardCount+2, 9):
            existingCards[j] = 0

        # Update required samples every 100 samples or when required
        #sample size is met.
        samples+=1

        frequency = 50
        if((samples%frequency == 0) or (samples >= requiredSamples)):
            proportionWins = winCount / samples
            wilsonSamples = wilson_score_samples(proportionWins,
                                                 confidence, confRange)
            # Round up to nearest 50 samples.
            requiredSamples = int(1 + (wilsonSamples/frequency))*frequency
            
    handStrength = (winCount / samples)
    return handStrength

def get_personalised_hand_strength(
    holeCards, communityCards, roundNumber, confidence=0.95, confRange=0.1):
    # Use my personal method to calculate the chance of holeCards beating
    #one opponent with random cards.
    
    # Convert player hole cards and community cards to the form
    #compatiple with the deuces library.
    myHand = setUpDeucesCards(holeCards)
    existingCards = np.append(holeCards, communityCards)
    # Add two spaces for opponent's cards.
    existingCards = np.append(existingCards, [0,0])
    
    if(len(existingCards)!=9):
        print("In handstrength module, ExistingCards is not right length")
    roundBoardCards = [0,3,4,5]
    communityCardCount = roundBoardCards[roundNumber-1]

    # Test win/loss outcome repeatedly.
    winCount = 0
    opponentHoleCards = [0]*2
    samples = 0
    
    # Best case scenario for sampling occurs when proportion=0.
    wilsonSamples = wilson_score_samples(0, confidence, confRange)
    # Round up to nearest 50 samples.
    requiredSamples = int(1 + (wilsonSamples/50))*50
    
    while(samples < requiredSamples):
        # Use copy of communityCards for testing.
        communityCardsCopy = copy.deepcopy(communityCards)
        
        # Generate opponent's cards and remaining community cards.
        for j in range(communityCardCount, 5):
            communityCardsCopy[j] = dealCard(existingCards)
            existingCards[j+2] = communityCardsCopy[j]
        for j in range(2):
            opponentHoleCards[j] = dealCard(existingCards)
            existingCards[j+7] = opponentHoleCards[j]

        # Evaluate hand ranks.
        allMyCards = np.append(holeCards, communityCardsCopy)
        allOpponentCards = np.append(opponentHoleCards, communityCardsCopy)
        myRank = get_hand_score(allMyCards)
        opponentRank = get_hand_score(allOpponentCards)
        
        # Compare hand ranks. Lower rank indicates better hand.
        if(myRank > opponentRank):
            winCount += 1
        elif(myRank == opponentRank):
            winCount += 0.5
            
        # Reset existing cards.
        for j in range(communityCardCount+2, 9):
            existingCards[j] = 0

        # Update required samples.
        samples+=1

        if((samples%50 == 0) or (samples >= requiredSamples)):
            proportionWins = winCount / samples            
            wilsonSamples = wilson_score_samples(proportionWins,
                                                 confidence, confRange)
            # Round up to nearest 50 samples.
            requiredSamples = int(1 + (wilsonSamples/50))*50
            
    handStrength = (winCount / samples)
    return handStrength

def get_hand_strength(
    holeCards, communityCards, roundNumber, confidence=0.95, confRange=0.1,
    method="deuces"):
    # Take cards numbered 1-52 and return chance of beating a random hand.
    if method=="deuces":
        handStrength = get_deuces_hand_strength(holeCards, communityCards,
                                         roundNumber, confidence=confidence,
                                         confRange=confRange)
    elif method=="personal":
        handStrength = get_personalised_hand_strength(holeCards, communityCards,
                                         roundNumber, confidence=confidence,
                                         confRange=confRange)
    else:
        handStrength = False
        print("Method ", method, " for computing hand strength was not found")
    return handStrength
