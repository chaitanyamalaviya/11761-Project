#!/usr/bin/python

from math import *
import random
import sys
import re
import numpy as np
from collections import defaultdict
from random import randint
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
import pickle
from random import randint

def getArticles(filename):
	docs = []
	doc = []
	with open(filename, 'r') as myfile:
		next(myfile)          #skip first line
		for l in myfile:
			if l.strip() == "~~~~~":
				docs.append(doc)
				doc = []
			else:
				sent = l.strip()
				doc.append(sent)

		docs.append(doc)
	return docs

def getLabels(filename):
	labels = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			labels.append(int(l.strip()))
	return labels

def produceNewArticles(articles, articleSize, numOfArticles):
	newArticles = []
	filteredArticles = filter(lambda (l,x): len(x) > articleSize, articles)

	for i in range(numOfArticles):
		currentArticle = filteredArticles[i][1]

		if articleSize == 1:
			n = randint(0,len(currentArticle) - articleSize)
			newArticle = currentArticle[n: (n+articleSize)]

			if( len(newArticle[0].strip().split()) < 5):
				n = randint(0,len(currentArticle) - articleSize)
				newArticle = currentArticle[n: (n+articleSize)]

			newArticles.append( (filteredArticles[i][0] , newArticle ) )

		else:
			n = randint(0,len(currentArticle) - articleSize)
			newArticles.append( (filteredArticles[i][0] , currentArticle[n: (n+articleSize) ] ) )

	return newArticles

def printArticlesToFile(articles, articlesFilename, lablesFilename):
	f = open(articlesFilename,'w')
	m = open(lablesFilename,'w')

	for label, article in articles:
		f.write('~~~~~\n')
		m.write(str(label) + '\n')
		for sentence in article:
			f.write(sentence + "\n")

	m.close()
	f.close()

if __name__ == '__main__':

	articles = getArticles("trainingSet.txt")
	labels = getLabels("trainingSetLabels.txt")
	articles = map(lambda (l,x): (l,x), zip(labels,articles) )

	realArticles = filter(lambda (l,x): l == 1, articles)
	fakeArticles = filter(lambda (l,x): l == 0, articles)

	numArticles = 50

	newArticles = []
	newArticles += produceNewArticles(realArticles,1, numArticles * 2)
	newArticles += produceNewArticles(fakeArticles,1, numArticles * 2)
	
	newArticles += produceNewArticles(realArticles,2, numArticles)
	newArticles += produceNewArticles(fakeArticles,2, numArticles)
	
	newArticles += produceNewArticles(realArticles,3, numArticles)
	newArticles += produceNewArticles(fakeArticles,3, numArticles)

	newArticles += produceNewArticles(realArticles,4, numArticles)
	newArticles += produceNewArticles(fakeArticles,4, numArticles)

	newArticles += produceNewArticles(realArticles,5, numArticles)
	newArticles += produceNewArticles(fakeArticles,5, numArticles)

	newArticles += produceNewArticles(realArticles,7, numArticles)
	newArticles += produceNewArticles(fakeArticles,7, numArticles)

	newArticles += produceNewArticles(realArticles,10, numArticles)
	newArticles += produceNewArticles(fakeArticles,10, numArticles)

	newArticles += produceNewArticles(realArticles,15, numArticles)
	newArticles += produceNewArticles(fakeArticles,15, numArticles)

	newArticles += produceNewArticles(realArticles,20, numArticles)
	newArticles += produceNewArticles(fakeArticles,20, numArticles)

	# newArticles += articles

	printArticlesToFile(newArticles, "expandedTrainingSet.txt", "expandedTrainingSetLabels.txt")