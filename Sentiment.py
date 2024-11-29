#!/usr/bin/env python
import re, random, math, collections, itertools, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np, warnings

PRINT_ERRORS=1

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = []
        for line in posDictionary:
            if not line.startswith(';'):
                posWordList.extend(re.findall(r"[a-z\-]+", line))
    posWordList.remove('a')

    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = []
        for line in negDictionary:
            if not line.startswith(';'):
                negWordList.extend(re.findall(r"[a-z\-]+", line))

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

  #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    #These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
    print(dataName, correct, total, correctpos, totalpos, correctneg, totalneg)
    calculate_metrics(dataName, correct, total, correctpos, totalpos, correctneg, totalneg)

def calculate_metrics(dataName, correct, total, correct_pos, total_pos, correct_neg, total_neg):
    accuracy = correct / total
    precision_pos = correct_pos / total_pos if total_pos > 0 else 0
    recall_pos = correct_pos / total_pos if total_pos > 0 else 0
    precision_neg = correct_neg / total_neg if total_neg > 0 else 0
    recall_neg = correct_neg / total_neg if total_neg > 0 else 0

    # F1 score for positive class
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    # F1 score for negative class
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    print(f"\n{dataName} \nClassification Results:") 
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision (Positive): {precision_pos*100:.2f}%")
    print(f"Recall (Positive): {recall_pos*100:.2f}%")
    print(f"Precision (Negative): {precision_neg*100:.2f}%")
    print(f"Recall (Negative): {recall_neg*100:.2f}%")
    print(f"F1 Score (Positive): {f1_pos*100:.2f}%")
    print(f"F1 Score (Negative): {f1_neg*100:.2f}%")


def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
    print(dataName, correct, total, correctpos, totalpos, correctneg, totalneg)
    calculate_metrics(dataName, correct, total, correctpos, totalpos, correctneg, totalneg)
 
#Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)
    
    # Check how many of these words are in the sentiment dictionary
    negative_in_dict = sum([word in sentimentDictionary for word in head])
    positive_in_dict = sum([word in sentimentDictionary for word in tail])

    print(f"\nNumber of negative words in sentiment dictionary: {negative_in_dict}")
    print(f"Number of positive words in sentiment dictionary: {positive_in_dict}")

def enhancedSentimentAnalysis(sentencesTest, dataName, sentimentDictionary):
    total = 0
    correct = 0
    negation_scope = 3  # Scope for negation effect

    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+|[!]", sentence.lower())
        score = 0
        negation_flag = False
        negation_countdown = 0
        modifier_multiplier = 1  # Default multiplier for sentiment modification

        for i, word in enumerate(Words):
            # Check for conjunctions and reset score if found
            if word in conjunctionWords:
                score *= 0.5  # Adjusting the score after a conjunction
                negation_flag = False  # Also reset negation flag
                modifier_multiplier = 1  # Reset modifier multiplier
                continue

            # Check for negation and set its countdown
            if word in negationWords:
                negation_flag = True
                negation_countdown = negation_scope
                modifier_multiplier = 1  # Reset modifier multiplier

            # Check for diminishers and intensifiers, adjust the multiplier
            elif word in diminisherWords:
                modifier_multiplier = 0.5
            elif word in intensifierWords:
                modifier_multiplier = 2

            if word in sentimentDictionary:
                sentimentValue = sentimentDictionary[word]

                # Handle negation
                if negation_flag and negation_countdown > 0:
                    sentimentValue *= -1
                    negation_countdown -= 1
                    if negation_countdown == 0:
                        negation_flag = False

                # Apply modifier (diminisher or intensifier)
                sentimentValue *= modifier_multiplier

                score += sentimentValue
                modifier_multiplier = 1  # Reset modifier multiplier after applying it

        total += 1
        predictedSentiment = "positive" if score >= 0 else "negative"
        if predictedSentiment == sentiment:
            correct += 1
        else:
            print ("ERROR (pos classed as neg %0.2f):" %score + sentence)

    accuracy = correct / total * 100
    print(f"\n{dataName} \nEnhanced Rule-Based Classification Results:")
    print(f"Accuracy: {accuracy:.2f}%")


#---------- Main Script --------------------------
negationWords = ['not', 'no', 'never', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'cannot', 'couldn\'t', 'shouldn\'t', 'won\'t', 'wouldn\'t']
diminisherWords = ['slightly', 'somewhat', 'bit', 'little', 'barely', 'hardly', 'just', 'only', 'marginally', 'modestly', 'scarcely', 'faintly', 'minimally', 'mildly', 'partially', 'rarely', 'subtly', 'lightly', 'infrequently', 'occasionally']
intensifierWords = ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely', 'utterly', 'highly', 'exceptionally', 'remarkably', 'really', 'particularly', 'profoundly', 'deeply', 'immensely', 'thoroughly', 'decidedly', 'significantly', 'outstandingly', 'enormously']
conjunctionWords = ['but', 'yet', 'however', 'although', 'though', 'even though', 'while', 'whereas']


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

#run naive bayes classifier on datasets
print ("Naive Bayes")
#testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)



#run sentiment dictionary based classifier on datasets
#testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
#testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)


# print most useful words
#mostUseful(pWordPos, pWordNeg, pWord, 100)

# Example usage
#enhancedSentimentAnalysis(sentencesTrain, "Films (Test Data)", sentimentDictionary)
#enhancedSentimentAnalysis(sentencesTest, "Films (Train Data)", sentimentDictionary)
#enhancedSentimentAnalysis(sentencesNokia, "Nokia (All Data)", sentimentDictionary)


