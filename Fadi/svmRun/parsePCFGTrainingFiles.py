from __future__ import division
import os
import re
import sys
import nltk
import nltk.tokenize
from nltk.parse import stanford
import ngrammodeler as NG
import pickle

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
        #         for i in range(len(goodArticles[j])):
        # #            if len(goodArticles[j][i].split()) >= 1 and len(goodArticles[j][i].split()) <= 10:
        #             if len(goodArticles[j][i].split()) >= 1:
        #                 sentenceScore = goodScores[j][i]/float(len(goodArticles[j][i].split()))
        #                 articleAvg.append(sentenceScore)
        #         average = sum(articleAvg)/len(goodArticles[j])
        #         print(average)
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

    computeArticlesAverageScore(goodArticles, goodScores, threshold)
    computeArticlesAverageScore(badArticles, badScores, threshold)

if __name__ == "__main__": main()
