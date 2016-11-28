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
import cooccurFeatures
import stopwordsfeature
import miminumPCFGScore
import posngrams
import countLDA
import lda


def toSVMformat(labels, features):
	strs = []

	for featureSet in features:
		if len(labels) != len(featureSet):
			print "Label and feature size don't match"
			exit()

	for i in range(len(labels)):
		label = labels[i]
		strr = str(label) + " "

		featureId = 1
		for featureSet in features:
			for m in range(len(featureSet[i,])):
				strr += str(featureId) + ":"+ str(featureSet[i,m]) + " "
				featureId += 1

		strs.append(strr)

	return strs

def getLabels(filename):
	labels = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			labels.append(int(l.strip()))
	return labels

def printToFile(strs, filename):
	f = open(filename,'w')
	
	for s in strs:
		f.write(s + "\n")

	f.close()

if __name__ == '__main__':

	labels = getLabels(sys.argv[2])

	features = []
	# features.append(cooccurFeatures.getFeatures(sys.argv[1]))
	# features.append(stopwordsfeature.getFeature(sys.argv[1]))
	# features.append(miminumPCFGScore.getFeature(sys.argv[1]))
	# features.append(cooccurFeatures.getFeatures2(sys.argv[1]))
	# features.append(cooccurFeatures.getFeatures3(sys.argv[1]))
	# features.append(cooccurFeatures.getFeatures4(sys.argv[1]))
	# features.append(cooccurFeatures.getFeatures6(sys.argv[1]))
	# features.append(cooccurFeatures.getFeatures7(sys.argv[1]))
	# features.append(cooccurFeatures.getFeatures8(sys.argv[1]))
	features.append(countLDA.feat_lda(sys.argv[1]))

	# features.append(posngrams.getFeature(sys.argv[1]))

	svmFormatLines = toSVMformat(labels,features)
	printToFile(svmFormatLines, sys.argv[3])



