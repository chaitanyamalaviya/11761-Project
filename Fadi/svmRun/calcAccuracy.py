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

def calcAccuracy(data):
	total = float(len(data))
	match = 0.0

	for d in data:
		if d[1] == d[2]:
			match += 1.0

	return match / total


if __name__ == '__main__':

	articles = getArticles(sys.argv[1])
	goldLabels = getLabels(sys.argv[2])
	predictedLabels = getLabels(sys.argv[3])

	zipped = zip(articles,goldLabels,predictedLabels)

	for i in [1,2,3,4,5,7,10,15,20]:
		filtered = filter(lambda x: len(x[0]) == i, zipped)
		accuracy = calcAccuracy(filtered)
		print str(i) + "\t" + str(accuracy)


	