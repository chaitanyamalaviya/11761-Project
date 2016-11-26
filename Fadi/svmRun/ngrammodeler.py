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
        self.trigramCounts = self.getNgramsCount(articles, 3)
        self.bigramCounts = self.getNgramsCount(articles, 2)
        self.unigramCounts = self.getNgramsCount(articles, 1)
        self.unigramProbabilities = self.computeUnigramsProbs()
        self.bigramProbabilities = self.computeBigramsProbs()
        self.trigramProbabilities = self.computeTrigramsProbs()
        self.l1 = 0.6
        self.l2 = 0.35
        self.l3 = 0.04
        self.l0 = 0.01

    def getNgramsCount(self, articles,n):
        """
        Return the count for all the trigrams in the set of articles given as an argument
        :param articles:
        :return:
        """
        counterList = []
        ngramsCounter = Counter(counterList)
        for article in articles:
            articleNgrams = self.getArticleNgrams(article, n)
            ngramsCounter = ngramsCounter + articleNgrams
        return ngramsCounter

    def getArticleNgrams(self, article, n):
        """
        Given a list of sentences, which make an article, returns the trigrams for the article
        :param article:
        :return:
        """
        counterList = []
        ngramsCounter = Counter(counterList)
        for sentence in article:
            sentence.append('</s>')
            for i in range(n-1):
                sentence.insert(0, '<s>')
            sentenceNgrams = self.findNgrams(sentence, n)
            ngramsCounter = ngramsCounter + Counter(sentenceNgrams)
            for i in range(n-1):
                del sentence[0]
            sentence.pop()
        return ngramsCounter

    def findNgrams(self, sequenceAsList, n):
        return zip(*[sequenceAsList[i:] for i in range(n)])

    def computeTrigramsProbs(self):
        trigramProbabilities = {}
        trigrams = self.trigramCounts
        bigrams = self.bigramCounts
        for trigram in trigrams:
            bigram = (trigram[0], trigram[1])
            if bigram == ('<s>', '<s>'):
                bigram = (trigram[1], trigram[2])
            trigramProb = float(trigrams[trigram]) / bigrams[bigram]
            trigramProbabilities[trigram] = trigramProb
        return trigramProbabilities

    def computeBigramsProbs(self):
        bigramProbabilities = {}
        bigrams = self.bigramCounts
        unigrams = self.unigramCounts
        for bigram in bigrams:
            unigram = (bigram[0])
            if unigram == '<s>':
                unigram = (bigram[1])
            try:
                bigramProb = float(bigrams[bigram]) / unigrams[(unigram,)]
            except:
                print(bigram)
            bigramProbabilities[bigram] = bigramProb
        return bigramProbabilities

    def computeUnigramsProbs(self):
        unigramProbabilities = {}
        unigrams = self.unigramCounts
        N = sum(self.unigramCounts.values())
        for unigram in unigrams:
            unigramProb = float(unigrams[unigram]) / N
            unigramProbabilities[unigram] = unigramProb
        return unigramProbabilities

    def smoothedProbability(self, trigram):
        l0 = self.l0
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        uniformProb = len(self.unigramProbabilities)
        unigram = trigram[2]
        bigram = (trigram[1], trigram[2])
        try:
            trigramProb = self.trigramProbabilities[trigram]
        except:
            trigramProb = 0.0
        try:
            bigramProb = self.bigramProbabilities[bigram]
        except:
            bigramProb = 0.0
        try:
            unigramProb = self.unigramProbabilities[(unigram,)]
        except:
            unigramProb = 0.0
        smoothedProbability = l1*trigramProb+l2*bigramProb+l3*unigramProb+l0*(1.0/uniformProb)
        return(smoothedProbability)

    def computeAverageArticleLogLikelihood(self, article):
        sentencesLL = []
        length = 0
        for sentence in article:
            ll = 0
            sentence.append('</s>')
            for i in range(2):
                sentence.insert(0, '<s>')
            trigrams = self.findNgrams(sentence, 3)
            for trigram in trigrams:
                probability = math.log(self.smoothedProbability(trigram))
                ll = ll + probability
            sentencesLL.append(float(ll)*len(sentence))
            length = length + len(sentence)
#        averageLL = np.mean(sentencesLL)
        averageLL = np.sum(sentencesLL)/float(length)
        return averageLL

    def getTopNgrams(self, n):
        return self.trigramCounts.most_common(n)