import os
import nltk
import nltk.tokenize
from nltk.parse import stanford
import ngrammodeler as NG
import pickle
# import plotFunctions as PF
import numpy as np

os.environ['STANFORD_PARSER'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'
os.environ['STANFORD_MODELS'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'

# parser = stanford.StanfordParser(model_path="/root/src/ls11761/ls-project/stanford/englishPCFG.ser.gz")

def saveObj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def posParseArticle(article):
    taggedArticle = []
    for sentence in article:
        sentence = sentence.lower()
        text = nltk.word_tokenize(sentence)
        posTaggedSentence = nltk.pos_tag(text)
        posTaggedSentence = [tag[1] for tag in posTaggedSentence]
        taggedArticle.append(posTaggedSentence)
    return taggedArticle

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

def importDataSet(corpusFileName):
    corpus = []
    path = os.getcwd()
    with open(path + '/' + corpusFileName, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        line = line[4:]
        line = line[:-4]
        line = line.rstrip()
        corpus.append(line)
    return corpus

def posParseLines(corpus, name):
    taggedCorpus = []
    for sentence in corpus:
        sentence = sentence.lower()
        text = nltk.word_tokenize(sentence)
        posTaggedSentence = nltk.pos_tag(text)
        posTaggedSentence = [tag[1] for tag in posTaggedSentence]
        taggedCorpus.append(posTaggedSentence)
    saveObj(taggedCorpus, name)
    return taggedCorpus

def getFakeGood(labelsFileName):
    path = os.getcwd()
    with open(path + '/' + labelsFileName, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.rstrip()
        labels.append(int(line))
    return labels

def posParseArticles(articles, name):
    parsedArticles = []
    for article in articles:
        posParsedArticle = posParseArticle(article)
        parsedArticles.append(posParsedArticle)
    saveObj(parsedArticles, name)
    return parsedArticles

def computeLogLikelihood(article):
    pass

def printArticlesPos(articles, posParsedArticles, labels, gOrB):
    if gOrB:
        labelMark = 1
    else:
        labelMark = 0
    i = 0
    b = 0
    for label in labels:
        if label == labelMark:
            for j in range(len(articles[i])):
                print(articles[i][j])
                print(posParsedArticles[b][j])
            b += 1
        i += 1

def computeLogLikelihood(parsedGoodArticles, parsedBadArticles, trigramModeler):
    y1 = []
    y2 = []
    for article in parsedGoodArticles:
        llArticle = trigramModeler.computeAverageArticleLogLikelihood(article)
        y1.append(llArticle)
        print(llArticle)

    for article in parsedBadArticles:
        llArticle = trigramModeler.computeAverageArticleLogLikelihood(article)
        y2.append(llArticle)
        print(llArticle)
    print
    print(sum(y1)/500.0)
    print(sum(y2)/500.0)
    PF.plotLL(y1, y2)
    return y1, y2

def computeArticleLogLikelihood(parsedArticle, trigramModeler):
    llArticle = trigramModeler.computeAverageArticleLogLikelihood(parsedArticle)
    return llArticle

def createLLFeatureForTraining(parsedGoodArticles, parsedBadArticles, labels, trigramModeler):
    g = 0
    b = 0
    llFeature = []
    for label in labels:
        if label == 0:
            llArticle = computeArticleLogLikelihood(parsedBadArticles[b], trigramModeler)
            ll = llArticle
            llFeature.append(ll)
            b += 1
        if label == 1:
            llArticle = computeArticleLogLikelihood(parsedGoodArticles[g], trigramModeler)
            ll = llArticle
            llFeature.append(ll)
            g += 1
    saveObj(llFeature, 'log_likelihood_feature')
    return 1

def createScores(labels, articles):
    goodArticles = []
    badArticles = []
    fg = open('goodArticles.txt', 'w')
    fb = open('badArticles.txt', 'w')
    i = 0
    for label in labels:
        if label == 1:
            goodArticles.append(articles[i])
            articleScores = parser.raw_parse_sents_PCFG(articles[i])
            sum = 0
            for a in articleScores:
                a = float(a)
                sum = sum + a
            averageScore = sum/len(articleScores)
            fg.write("%s, %s, %f\n" % (articles[i], articleScores, averageScore))
        if label == 0:
            badArticles.append(articles[i])
            articleScores = parser.raw_parse_sents_PCFG(articles[i])
            sum = 0
            for a in articleScores:
                a = float(a)
                sum = sum + a
            averageScore = sum / len(articleScores)
            fb.write("%s, %s, %f\n" % (articles[i], articleScores, averageScore))
        i = i + 1
    fg.close()
    fb.close()

def computeAccuracy(devLabels, devArticles, ngramModeler):
    i = 0
    count = 0
    for label in devLabels:
        article = devArticles[i]
        parsedArticle = posParseArticle(article)
        ll = computeArticleLogLikelihood(parsedArticle, ngramModeler)
        if ll < -57:
            result = 0
        else:
            result = 1
        if result == label:
            count += 1
        i += 1
    print("Good Classifications: %d" % count)
    return 1

def getFeature(devFileName):
    if devFileName == 'trainingSet.dat':
        feature = loadObj('log_likelihood_feature')
        featureArray = np.zeros([len(feature), 1], dtype=float)
        i = 0
        for f in feature:
            featureArray[i] = f
            i += 1
    else:
        ngramModeler = loadObj('smallTrigramModeler')
        articles = importArticles(devFileName)
        featureLength = len(articles)
        featureArray = np.zeros([featureLength, 1], dtype=float)
        i = 0
        for article in articles:
            parsedArticle = posParseArticle(article)
            ll = computeArticleLogLikelihood(parsedArticle, ngramModeler)
            featureArray[i] = ll
            i += 1
    return featureArray



def main():
    #dataSet = importDataSet('LM-train-100MW.txt')
    #parsedCorpus = posParseLines(dataSet, 'corpusPosTagged')
    #parsedCorpus = loadObj('corpusPosTagged')
    articles = importArticles('trainingSet.dat')
    labels = getFakeGood('trainingSetLabels.dat')
    devArticles = importArticles('developmentSet.dat')
    devLabels = getFakeGood('developmentSetLabels.dat')
#    uncomment the next if you want to pos parse the articles again, otherwise it just loads the last parse
#    parsedGoodArticles = posParseArticles(goodArticles, 'posgoodarticles')
#    parsedBadArticles = posParseArticles(badArticles, 'posbadarticles')
    #createScores(labels, articles)
    parsedGoodArticles = loadObj('posgoodarticles')
    parsedBadArticles = loadObj('posbadarticles')
    #printArticlesPos(articles, parsedGoodArticles, labels, True)
    trigramModeler = NG.NgramModeler(parsedGoodArticles)
    saveObj(trigramModeler, 'smallTrigramModeler')
    computeLogLikelihood(parsedGoodArticles, parsedBadArticles, trigramModeler)
#    createLLFeatureForTraining(parsedGoodArticles, parsedBadArticles, labels, trigramModeler)
    computeAccuracy(devLabels, devArticles, trigramModeler)
    computeAccuracy(labels, articles, trigramModeler)
    getFeature('trainingSet.dat')

if __name__ == "__main__": main()

    # trigramModeler2 = NG.NgramModeler(parsedBadArticles)
    # print(set([a[0] for a in trigramModeler.getTopNgrams(20)])-set([a[0] for a in trigramModeler2.getTopNgrams(20)]))
    # print(set([a[0] for a in trigramModeler2.getTopNgrams(20)])-set([a[0] for a in trigramModeler.getTopNgrams(20)]))
