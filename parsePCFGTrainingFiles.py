import os
import re
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

def main():
    goodArticles, goodScores = importScores('goodArticles.txt')
    badArticles, badScores = importScores('badArticles.txt')

    # maxMinScores = []
    # for j in range(len(badArticles)):
    #     print(min(badScores[j]))
    #     for i in range(len(badArticles[j])):
    #         if len(badArticles[j][i].split()) >= 6 and len(badArticles[j][i].split()) <= 10:
    #             maxMinScores.append(min(badScores[j]))
    # print
    # print
    # minMaxScores = []
    for j in range(len(goodArticles)):
        articleAvg = []
        for i in range(len(goodArticles[j])):
#            if len(goodArticles[j][i].split()) >= 1 and len(goodArticles[j][i].split()) <= 10:
            if len(goodArticles[j][i].split()) >= 1:
                sentenceScore = goodScores[j][i]/float(len(goodArticles[j][i].split()))
                articleAvg.append(sentenceScore)
        average = sum(articleAvg)/len(goodArticles[j])
        print(average)

    print
    print

    for j in range(len(badArticles)):
        articleAvg = []
        for i in range(len(badArticles[j])):
            #            if len(goodArticles[j][i].split()) >= 1 and len(goodArticles[j][i].split()) <= 10:
            if len(badArticles[j][i].split()) >= 1:
                sentenceScore = badScores[j][i] / float(len(badArticles[j][i].split()))
                articleAvg.append(sentenceScore)
        average = sum(articleAvg) / len(badArticles[j])
        print(average)
        # if average < 14.0:
        #     print(badArticles[j])




        #    print("%f %f" % (max(maxMinScores), min(minMaxScores)))
        # for i in range(len(badArticles[j])):
        #     if len(badArticles[j][i].split())>=7:
        #         print(badScores[j][i])
        #         if badScores[j][i] > -100:
        #             print(badArticles[j][i])
if __name__ == "__main__": main()
