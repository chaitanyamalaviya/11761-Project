import os
import nltk
import nltk.tokenize
from nltk.parse import stanford
import ngrammodeler as NG
from nltk.tag.stanford import StanfordPOSTagger
import pickle
import plotFunctions as PF
import logging
import numpy as np
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

os.environ['STANFORD_PARSER'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'
os.environ['STANFORD_MODELS'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'
os.environ['STANFORDNER'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars/'
parser = stanford.StanfordParser(model_path="/root/src/ls11761/ls-project/stanford/englishPCFG.ser.gz")
stanford_dir = os.environ['STANFORDNER']
model = stanford_dir + 'english-bidirectional-distsim.tagger'
jarfile = stanford_dir  + 'stanford-postagger.jar'
stanford_pos = StanfordPOSTagger(model, jarfile)

PENNPOSTAGS = [ "CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB" ]

features = [ ]

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
#        posTaggedSentence = stanford_pos.tag(text)
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

def checkLinguisticFeature(parsedSentence):
    """
    Here we need to implement the feature
    :param parsedSentence:
    :return:
    """
#    badTags = [ 'VBD', 'VBG', 'VBP', 'VBZ' ]
    badTags = [ 'RB' ]
    if parsedSentence[0] in badTags:
        return True
    else:
        return False

def checkLinguisticFeatureWithArg(parsedSentence, tag, pos):
    """
    Here we need to implement the feature
    :param parsedSentence:
    :return:
    """
#    badTags = [ 'VBD', 'VBG', 'VBP', 'VBZ' ]
    badTags = [ tag ]
    if parsedSentence[int(pos)] in badTags:
        return True
    else:
        return False




def checkLinguisticFeatureTest(parsedSentence, tag):
    """
    Here we need to implement the feature
    :param parsedSentence:
    :return:
    """
#    badTags = [ 'VBD', 'VBG', 'VBP', 'VBZ' ]
    badTags = [ tag ]
    string = "parsedSentence[0] in badTags"
    #if parsedSentence[0] in badTags:
    if eval(string):
        return True
    else:
        return False

def checkLinguisticFeatureOnArticle(article):
    """
    Returns the number of sentence that have the checked features in the article.
    :param article:
    :return:
    """
    parsedArticle = posParseArticle(article)
    count = 0
    for parsedSentence in parsedArticle:
        feature = checkLinguisticFeature(parsedSentence)
        if feature:
#            count +=1
            count =1
    return count

def checkLinguisticFeatureOnParsedArticle(parsedArticle, tag):
    """
    Returns the number of sentence that have the checked features in the article.
    :param article:
    :return:
    """
    count = 0
    for parsedSentence in parsedArticle:
        feature = checkLinguisticFeatureTest(parsedSentence, tag)
        if feature:
#            count +=1
            count = 1
    return count

def checkLinguisticFeatureOnTraining(labels, parsedGoodArticles, parsedBadArticles):
    max = 0
    for tag in PENNPOSTAGS:
        g = 0
        b = 0
        i = 0
        linguisticFeature = []
        goodCount = 0
        badCount = 0
        for label in labels:
            if label == 0:
                count = checkLinguisticFeatureOnParsedArticle(parsedBadArticles[b], tag)
                linguisticFeature.append(count)
                b += 1
                badCount += count
            if label == 1:
                count = checkLinguisticFeatureOnParsedArticle(parsedGoodArticles[g], tag)
                linguisticFeature.append(count)
                g += 1
                goodCount += count
            i += 1
        diffCount = badCount - goodCount
        if diffCount > max:
            max = diffCount
            maxTag = tag
        logging.debug("This is the average value of the feature for the Good Articles: %f" % (goodCount/500.0))
        logging.debug("This is the average value of the feature for the Bad Articles: %f" % (badCount/500.0))
    print("This is the max Tag: %s, %d" % (maxTag, max))
    saveObj(linguisticFeature, 'handMadeLinguisticFeature')
    return linguisticFeature




def getFeatureSingle(devFileName):
    articles = importArticles(devFileName)
    featureLength = len(articles)
    featureArray = np.zeros([featureLength,1], dtype=float)
    featureList = []
    i = 0
    for article in articles:
        linguisticFeature = checkLinguisticFeatureOnArticle(article)
        featureArray[i] = linguisticFeature
        featureList.append(float(linguisticFeature))
        i += 1
    return featureList

def getFeatureArg(devFileName, tag, position):
    articles = importArticles(devFileName)
    featureLength = len(articles)
    featureArray = np.zeros([featureLength,1], dtype=float)
    featureList = []
    i = 0
    for article in articles:
        linguisticFeature = checkLinguisticFeatureOnArticleWithArg(article, tag, position)
        featureArray[i] = linguisticFeature
        featureList.append(float(linguisticFeature))
        i += 1
    return featureList

def checkLinguisticFeatureOnArticleWithArg(article, tag, position):
    """
    Returns the number of sentence that have the checked features in the article.
    :param article:
    :return:
    """
    parsedArticle = posParseArticle(article)
    count = 0
    for parsedSentence in parsedArticle:
        feature = checkLinguisticFeatureWithArg(parsedSentence, tag, position)
        if feature:
#            count +=1
            count = 1
    return count

def createPosMatrixFeatures(devFileName):
    positions = [-1, 0]
    featureMatrix = []
    for position in positions:
        for tag in PENNPOSTAGS:
            logging.debug("Doing tag: %s" % tag)
            featureList = getFeatureArg(devFileName, tag, position)
            featureMatrix.append(featureList)
    return featureMatrix

def getFeature(devFileName, name):
    """
    To be called to create the overall feature
    :param devFileName:
    :param name:
    :return:
    """
    featureMatrix = createPosMatrixFeatures(devFileName)
    return featureMatrix


def main():
    goodArticles = []
    badArticles = []
    articles = importArticles('developmentSet.dat')
    labels = getFakeGood('developmentSetLabels.dat')

    #parsedGoodArticles = posParseArticles(goodArticles, 'posgoodarticles')
    #parsedBadArticles = posParseArticles(badArticles, 'posbadarticles')
    #parsedGoodArticles = loadObj('posgoodarticles')
    #parsedBadArticles = loadObj('posbadarticles')
    #checkLinguisticFeatureOnTraining(labels, parsedGoodArticles, parsedBadArticles)
    #featureMatrix = createPosMatrixFeatures('developmentSet.dat')
    #saveObj(featureMatrix, 'hmlFeaturesMatrixDev')
    featureMatrix = loadObj('hmlFeaturesMatrix')
    print("Hello")
if __name__ == "__main__": main()
