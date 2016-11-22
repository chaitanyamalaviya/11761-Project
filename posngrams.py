import os
import nltk
import nltk.tokenize

def posParseArticle(article):
    taggedArticle = []
    for sentence in article:
        text = nltk.word_tokenize(sentence)
        posTaggedSentence = nltk.pos_tag(text)
        #posTaggedSentence = [tag[1] for tag in posTaggedSentence]
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

def main():
    articles = importArticles('trainingSet.dat')
    labels = getFakeGood('trainingSetLabels.dat')
    for article in articles:
        posParsedArticle = posParseArticle(article)
if __name__ == "__main__": main()
