from __future__ import division

import os
import re
import sys

import numpy as np

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

def computeArticlesAverageScore(articles, scores, threshold):
    count = 0
    avAverage = 0
    skipped = 3
    for j in range(len(articles)):
        articleAvg = []
        length = 0
        for i in range(len(articles[j])):
            #            if len(goodArticles[j][i].split()) >= 1 and len(goodArticles[j][i].split()) <= 10:
            if len(articles[j][i].split()) >= skipped:
                length = length + len(articles[j][i].split())
                sentenceScore = scores[j][i] * len(articles[j][i].split())
                articleAvg.append(sentenceScore)
        average = sum(articleAvg) / length
        avAverage = avAverage + average
        if average <= threshold:
            count = count + 1
#        print("Good Article: %d, average: %f" % (j, average))
    print("Number of articles with score less than %d: %d" % (threshold, count))
    print("Average on all articles: %f" % (avAverage / len(articles)))

def createFeatureFromScoresFiles(badArticles, goodArticles, badScores, goodScores, labels):
    g = 0
    b = 0
    good = open('goodscoresAvg.txt', 'w')
    bad = open('badscoresAvg.txt', 'w')
    pcfgScoresFeature = []
    for label in labels:
        length = 0
        articleScores = []
        if label == 0:
            for i in range(len(badArticles[b])):
                length = length + len(badArticles[b][i].split())
                sentenceScore = badScores[b][i] * len(badArticles[b][i].split())
                articleScores.append(sentenceScore)
            articleScores = [s / float(length) for s in articleScores]
            pcfgScoresFeature.append(sum(articleScores))
            bad.write("%s\n" % sum(articleScores))
            b += 1
        if label == 1:
            for i in range(len(goodArticles[g])):
                length = length + len(goodArticles[g][i].split())
                sentenceScore = goodScores[g][i] * len(goodArticles[g][i].split())
                articleScores.append(sentenceScore)
            articleScores = [s / float(length) for s in articleScores]
            pcfgScoresFeature.append(sum(articleScores))
            good.write("%s\n" % sum(articleScores))
            g += 1
    return pcfgScoresFeature

def getFakeGood(labelsFileName):
    path = os.getcwd()
    with open(path + '/' + labelsFileName, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.rstrip()
        labels.append(int(line))
    return labels

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
    featureList = []
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
        featureList.append((f))
    return featureList

def main():
    path = os.getcwd()
    numberOfArguments = len(sys.argv)
    if (numberOfArguments != 4  ):
        print("This command takes three arguments, which are the bad and the good articles file names and the threshold.")
        exit()
    else:
        goodArticlesFileName = sys.argv[1]
        goodArticlesFilePath = path+'/'+goodArticlesFileName
        badArticlesFileName = sys.argv[2]
        badArticlesFilePath = path+'/'+badArticlesFileName
        threshold = int(sys.argv[3])

    goodArticles, goodScores = importScores(goodArticlesFileName)
    badArticles, badScores = importScores(badArticlesFileName)
    print getFeature('trainingSet.dat')

    computeArticlesAverageScore(goodArticles, goodScores, threshold)
    computeArticlesAverageScore(badArticles, badScores, threshold)

if __name__ == "__main__": main()
