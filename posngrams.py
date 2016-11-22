import os
import nltk
import nltk.tokenize
import ngrammodeler as NG
import pickle

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
    articles = importArticles('trainingSet.dat')
    labels = getFakeGood('trainingSetLabels.dat')
    i = 0
    for label in labels:
        if label == 1:
            goodArticles.append(articles[i])
        if label == 0:
            badArticles.append(articles[i])
        i = i + 1
    # uncomment the next if you want to pos parse the articles again, otherwise it just loads the last parse
    #parsedGoodArticles = posParseArticles(goodArticles, 'posgoodarticles')
    #parsedBadArticles = posParseArticles(badArticles, 'posbadarticles')
    parsedGoodArticles = loadObj('posgoodarticles')
    trigramModeler = NG.NgramModeler(parsedGoodArticles)
    for article in parsedGoodArticles:
        llArticle = trigramModeler.computeAverageArticleLogLikelihood(article)
        print(llArticle)
    parsedBadArticles = loadObj('posbadarticles')
    print("Bad Articles")
    print
    for article in parsedBadArticles:
        llArticle = trigramModeler.computeAverageArticleLogLikelihood(article)
        print(llArticle)

if __name__ == "__main__": main()
