from __future__ import division
import pickle
import os
import nltk
import nltk.tokenize
from nltk.corpus import stopwords
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
STOPWORDS = set(stopwords.words('english'))

def saveObj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def importArticles(corpusFileName):
    articles = []
    path = os.getcwd()
    with open(path + '/' + corpusFileName, "r") as f:
        lines = f.readlines()
    article = []
    for line in lines:
        line = line.rstrip()
        if line == "~~~~~":
            if article:
                articles.append(article)
                article = []
        else:
            # Removes the start stop tags for the sentence
            line = line[4:]
            line = line[:-4]
            line = line.rstrip()
            article.append(line)
    articles.append(article)
    return articles

def getFakeGood(labelsFileName):
    path = os.getcwd()
    with open(path + '/' + labelsFileName, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.rstrip()
        labels.append(int(line))
    return labels

def getNumberOfStopwords(article):
    sumStop = 0
    sumLength = 0
    for sentence in article:
        tokenizedSentence = nltk.word_tokenize(sentence.lower())
        stopwords = len([i for i in tokenizedSentence if i in STOPWORDS])
        length = len(tokenizedSentence)
        sumStop += stopwords*length
        sumLength += length
    return float(sumStop)/sumLength


def main():
    articlesPickle = []
    goodArticles = []
    badArticles = []
    articles = importArticles('trainingSet.dat')
    labels = getFakeGood('trainingSetLabels.dat')
    i = 0
    for label in labels:
        if label == 1:
            article = articles[i]
            score = getNumberOfStopwords(article)
            logging.debug("Average number of stopwords in good article: %s" % score)
            goodArticles.append(score)
            articlesPickle.append(score)

        if label == 0:
            article = articles[i]
            score = getNumberOfStopwords(article)
            logging.debug("Average number of stopwords in bad article: %s" % score)
            badArticles.append(score)
            articlesPickle.append(score)
        i = i + 1
    logging.debug("Average number of stopwords in good articles: %f" % (sum(goodArticles)/len(goodArticles)))
    logging.debug("Average number of stopwords in bad articles: %f" % (sum(badArticles)/len(badArticles)))
    saveObj(articlesPickle, 'feature_stopwords')
if __name__ == "__main__": main()






