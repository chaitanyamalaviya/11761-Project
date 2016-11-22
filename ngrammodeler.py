import time
import sys
import os
import math
import logging
import re
from collections import defaultdict
from collections import OrderedDict
from collections import Counter
from itertools import izip
import numpy as np

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

class NgramModeler:
    def __init__(self, articles):
        self.articles = articles
        self.trigramCounts = self.getTrigramsCount(articles)

    def getTrigramsCount(self, articles):
        """
        Return the count for all the trigrams in the set of articles given as an argument
        :param articles:
        :return:
        """
        counterList = []
        trigramsCounter = Counter(counterList)
        for article in articles:
            articleTrigrams = self.getArticleTrigrams(article)
            trigramsCounter = trigramsCounter + articleTrigrams
        return trigramsCounter

    def getArticleTrigrams(self, article):
        """
        Given a list of sentences, which make an article, returns the trigrams for the article
        :param article:
        :return:
        """
        counterList = []
        trigramsCounter = Counter(counterList)
        for sentence in article:
            sentence = "<s> <s> " + sentence + "</s>"
            sentenceAsList = sentence.split()
            sentenceTrigrams = self.findNgrams(sentenceAsList, 3)
            trigramsCounter = trigramsCounter + Counter(sentenceTrigrams)
        return trigramsCounter

    def findNgrams(self, sequenceAsList, n):
        return zip(*[sequenceAsList[i:] for i in range(n)])


    def statesProbabilities(self):
        bigramsProb = {}
        statesProbs = {}
        unigramsProb = {}
        interpolatedProbs = {}
        bigramsList = self.statesBigrams
        unigramsList = self.statesUnigrams

        bigramsCounter = Counter(bigramsList)
        unigramsCounter = Counter(unigramsList)
        bigramTypes = bigramsCounter.keys()
        unigramTypes = unigramsCounter.keys()
        self.statesList = unigramTypes
        numberOfStates = len(unigramTypes)

        # In the slides A has size N+1 x N+1 as it starts counting from 0 to N. In this case N=(numberOfStates-1)
        # because the matrix A starts its indexes also from 0 but up until (numberOfStates-1)
        A = np.zeros([numberOfStates,numberOfStates])

        # for unigram in unigramTypes:
        #     unigramProb = float(unigramsCounter[unigram])/len(unigramsList)
        #     unigramsProb[unigram] = unigramProb

        for bigram in bigramTypes:
            unigram = bigram[0]
            previousState = unigramTypes.index(unigram)
            currentState = unigramTypes.index(bigram[1])
            # The following is the probability of moving from previousState to currentState
            bigramProb = float(bigramsCounter[bigram]) / unigramsCounter[unigram]
            A[previousState][currentState] = bigramProb
            # if state not in statesProbs:
            #     statesProbs[state] = defaultdict(int)
            # statesProbs[state][unigram] = bigramProb
            #bigramsProb[bigram] = bigramProb
        #return bigramsProb
        return A