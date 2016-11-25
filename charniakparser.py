import os
from bllipparser import RerankingParser
from bllipparser.ModelFetcher import download_and_install_model
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

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

def main():
#   model_dir = download_and_install_model('WSJ', '/tmp/models')
    model_dir = download_and_install_model('WSJ+Gigaword-v2', '/tmp/models')
    parser = RerankingParser.from_unified_model_dir(model_dir)
    goodArticles = []
    badArticles = []
    articles = importArticles('trainingSet.dat')
    labels = getFakeGood('trainingSetLabels.dat')
    fg = open('goodArticlesBllip.txt', 'w')
    fb = open('badArticlesBllip.txt', 'w')
    i = 0
    for label in labels:
        if label == 1:
            goodArticles.append(articles[i])
            articleScores = []
            for sentence in articles[i]:
                logging.debug("Looking into good sentence: %s" % sentence)
                sentenceParses = parser.parse(sentence,1)
                sentenceBestScore = sentenceParses[0].parser_score
                logging.debug("Score for good sentence: %s" % sentenceBestScore)
                articleScores.append(sentenceBestScore)
            sum = 0
            for a in articleScores:
                a = float(a)
                sum = sum + a
            averageScore = sum/len(articleScores)
            fg.write("%s, %s, %f\n" % (articles[i], articleScores, averageScore))
        if label == 0:
            badArticles.append(articles[i])
            articleScores = []
            for sentence in articles[i]:
                logging.debug("Looking into bad sentence: %s" % sentence)
                sentenceParses = parser.parse(sentence,1)
                sentenceBestScore = sentenceParses[0].parser_score
                logging.debug("Score for bad sentence: %s" % sentenceBestScore)
                articleScores.append(sentenceBestScore)
            sum = 0
            for a in articleScores:
                a = float(a)
                sum = sum + a
            averageScore = sum / len(articleScores)
            fb.write("%s, %s, %f\n" % (articles[i], articleScores, averageScore))
        i = i + 1
    fg.close()
    fb.close()




if __name__ == "__main__": main()
