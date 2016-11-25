import os
import nltk
import nltk.tokenize
from nltk.parse import stanford
import ngrammodeler as NG
import pickle
import plotFunctions as PF

os.environ['STANFORD_PARSER'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'
os.environ['STANFORD_MODELS'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'

parser = stanford.StanfordParser(model_path="/root/src/ls11761/ls-project/stanford/englishPCFG.ser.gz")

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


def main():
    goodArticles = []
    badArticles = []
    dataSet = importDataSet('LM-train-100MW.txt')
    parsedCorpus = posParseLines(dataSet, 'corpusPosTagged')
    #parsedCorpus = loadObj('corpusPosTagged')
    articles = importArticles('trainingSet.dat')
    labels = getFakeGood('trainingSetLabels.dat')
    
    # fg = open('goodArticles.txt', 'w')
    # fb = open('badArticles.txt', 'w')
    # i = 0
    # for label in labels:
    #     if label == 1:
    #         goodArticles.append(articles[i])
    #         articleScores = parser.raw_parse_sents_PCFG(articles[i])
    #         sum = 0
    #         for a in articleScores:
    #             a = float(a)
    #             sum = sum + a
    #         averageScore = sum/len(articleScores)
    #         fg.write("%s, %s, %f\n" % (articles[i], articleScores, averageScore))
    #     if label == 0:
    #         badArticles.append(articles[i])
    #         articleScores = parser.raw_parse_sents_PCFG(articles[i])
    #         sum = 0
    #         for a in articleScores:
    #             a = float(a)
    #             sum = sum + a
    #         averageScore = sum / len(articleScores)
    #         fb.write("%s, %s, %f\n" % (articles[i], articleScores, averageScore))
    #     i = i + 1
    # fg.close()
    # fb.close()
    # uncomment the next if you want to pos parse the articles again, otherwise it just loads the last parse
    #parsedGoodArticles = posParseArticles(goodArticles, 'posgoodarticles')
    #parsedBadArticles = posParseArticles(badArticles, 'posbadarticles')
    parsedGoodArticles = loadObj('posgoodarticles')
    parsedBadArticles = loadObj('posbadarticles')
    trigramModeler = NG.NgramModeler(parsedGoodArticles)
    trigramModeler2 = NG.NgramModeler(parsedBadArticles)
    # print(set([a[0] for a in trigramModeler.getTopNgrams(20)])-set([a[0] for a in trigramModeler2.getTopNgrams(20)]))
    # print(set([a[0] for a in trigramModeler2.getTopNgrams(20)])-set([a[0] for a in trigramModeler.getTopNgrams(20)]))
    y1 = []
    for article in parsedGoodArticles:
        llArticle = trigramModeler.computeAverageArticleLogLikelihood(article)
        y1.append(llArticle)
        print(llArticle)
    print(len(y1))
    PF.plotLL(y1,y1)

    parsedBadArticles = loadObj('posbadarticles')
    print("Bad Articles")
    print
    for article in parsedBadArticles:
        #print(article)
        llArticle = trigramModeler.computeAverageArticleLogLikelihood(article)
        print(llArticle)

if __name__ == "__main__": main()
