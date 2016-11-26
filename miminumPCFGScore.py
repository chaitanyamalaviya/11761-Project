from __future__ import division
import os
import re
import sys
os.environ['STANFORD_PARSER'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'
os.environ['STANFORD_MODELS'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'
import nltk
import nltk.tokenize
from nltk.parse import stanford
import ngrammodeler as NG
import pickle
import numpy as np

parser = stanford.StanfordParser(model_path="/root/src/ls11761/ls-project/stanford/englishPCFG.ser.gz")

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


def importScores(fileName):
    articlesList = []
    scoresList = []
    with open(fileName, "r") as f:
        lines = f.readlines()
    for line in lines:
        matchObj = re.match(r'^(\[.*?\]), (\[.*?\])', line, re.M | re.I)
        if matchObj:
            article = matchObj.group(1)
            scores = matchObj.group(2)
            article = eval(article)
            scores = eval(scores)
            scores = [float(i) for i in scores]
            articlesList. append(article)
            scoresList.append(scores)
    return articlesList, scoresList


def computeArticleMinScore(article):
    articleScores = parser.raw_parse_sents_PCFG(article)
    length = 0
    articleScoresList = []
    for i in range(len(article)):
        length = length + len(article[i].split())
        sentenceScore = float(articleScores[i]) * len(article[i].split())
        articleScoresList.append(sentenceScore)
    articleScoresList = [s/float(length) for s in articleScoresList ]
    minScore = min(articleScoresList)
    return minScore

def computeMinimumScoresFromTrainedFile(articles, scores):
    minScoreFeature = []
    for j in range(len(articles)):
        minScore = min(scores[j])
        minScoreFeature.append(minScore)
    return minScoreFeature

def getFakeGood(labelsFileName):
    path = os.getcwd()
    with open(path + '/' + labelsFileName, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.rstrip()
        labels.append(int(line))
    return labels

def computeAccuracy(devLabels, devArticles):
    i = 0
    count = 0
    for label in devLabels:
        article = devArticles[i]
        minPCFGArticle = computeArticleMinScore(article)
        if minPCFGArticle < -70:
            result = 0
        else:
            result = 1
        if result == label:
            count += 1
        i += 1
    print("Good Classifications: %d" % count)
    return 1

def createFeatureFromScoresFiles(badArticles, goodArticles, badScores, goodScores, labels):
    g = 0
    b = 0
    goodMinScoresSum = 0
    badMinScoresSum = 0
    minScoresFeature = []
    for label in labels:
        length = 0
        articleScores = []
        if label == 0:
            for i in range(len(badArticles[b])):
                length = length + len(badArticles[b][i].split())
                sentenceScore = badScores[b][i] * len(badArticles[b][i].split())
                articleScores.append(sentenceScore)
            articleScores = [s / float(length) for s in articleScores]
            minScore = min(articleScores)
            minScoresFeature.append(minScore)
            b += 1
            badMinScoresSum += minScore
        if label == 1:
            for i in range(len(goodArticles[g])):
                length = length + len(goodArticles[g][i].split())
                sentenceScore = goodScores[g][i] * len(goodArticles[g][i].split())
                articleScores.append(sentenceScore)
            articleScores = [s / float(length) for s in articleScores]
            minScore = min(articleScores)
            minScoresFeature.append(minScore)
            g += 1
            goodMinScoresSum += minScore
    #saveObj(minScoresFeature, name)
    print("This is the average min score for the good articles: %f" % (goodMinScoresSum/500.0))
    print("This is the average min score for the bad articles: %f" % (badMinScoresSum/500.0))
    return minScoresFeature

def getFileNames(devFileName):
    if devFileName == 'trainingSet.dat':
        labelsFileName = 'trainingSetLabels.dat'
        goodArticlesFileName = 'goodArticles.txt'
        badArticlesFileName = 'badArticles.txt'
    else:
        labelsFileName = 'developmentSetLabels.dat'
        goodArticlesFileName = 'goodArticlesDevelopment.txt'
        badArticlesFileName = 'badArticlesDevelopment.txt'
    return labelsFileName, goodArticlesFileName, badArticlesFileName


def getFeature(devFileName):
    labelsFileName, goodArticlesFileName, badArticlesFileName = getFileNames(devFileName)
    goodArticles, goodScores = importScores(goodArticlesFileName)
    badArticles, badScores = importScores(badArticlesFileName)
    featureLength = len(goodArticles) + len(badArticles)
    labels = getFakeGood(labelsFileName)
    featureArray = np.zeros([featureLength, 1], dtype=float)
    feature = createFeatureFromScoresFiles(badArticles, goodArticles, badScores, goodScores, labels)
    i = 0
    for f in feature:
        featureArray[i] = f
        i += 1
    return featureArray


def main():
    path = os.getcwd()
    numberOfArguments = len(sys.argv)
    devArticles = importArticles('developmentSet.dat')
    devLabels = getFakeGood('developmentSetLabels.dat')
    labels = getFakeGood('trainingSetLabels.dat')
    if (numberOfArguments != 4  ):
        print("This command takes three arguments: the trained files for the bad and the good articles, and the labels file.")
        exit()
    else:
        goodArticlesFileName = sys.argv[1]
        goodArticlesFilePath = path+'/'+goodArticlesFileName
        badArticlesFileName = sys.argv[2]
        badArticlesFilePath = path+'/'+badArticlesFileName
        labelsFileName = sys.argv[3]
        # labelsFilePath = path+'/'+labelsFileName
        # featureName = sys.argv[4]

    goodArticles, goodScores = importScores(goodArticlesFileName)
    badArticles, badScores = importScores(badArticlesFileName)
    labels = getFakeGood(labelsFileName)
    createFeatureFromScoresFiles(badArticles, goodArticles, badScores, goodScores, labels)
    getFeature('developmentSet.dat')
    #feature = loadObj('min_score_feature')
#    computeAccuracy(devLabels, devArticles)
if __name__ == "__main__": main()
