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



def getTrainingLabels(filename):
	labels = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			labels.append(int(l.strip()))
	return labels


def getFeaturesFromFile(filename):
	return pickle.load( open( filename, "rb" ) )

def getFeatures(args):
	features = []

	for filename in args:
		features.append(getFeaturesFromFile(filename))

	return features


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
			strr += str(featureId) + ":"+ str(featureSet[i]) + " "
			featureId += 1

		strs.append(strr)

	return strs

def printToFile(strs, filename):
	f = open(filename,'w')
	
	for s in strs:
		f.write(s + "\n")

	f.close()

if __name__ == '__main__':
	labels = getTrainingLabels(sys.argv[1])
	features = getFeatures(sys.argv[3:])

	strs = toSVMformat(labels,features)

	printToFile(strs, sys.argv[2])

	
