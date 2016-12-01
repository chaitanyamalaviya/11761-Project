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
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn import svm


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

def getFakeGood(labelsFileName):
    path = os.getcwd()
    with open(path + '/' + labelsFileName, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.rstrip()
        labels.append(int(line))
    return labels

def toArray(featureMatrices):
	featuresArr = []

	for i in range(featureMatrices[0].shape[0]):
		newArr = np.array([])
		for featureMatrix in featureMatrices:
			newArr = np.concatenate((newArr,featureMatrix[i,:]))
		featuresArr.append(np.array(newArr))

	return np.array(featuresArr)


if __name__ == '__main__':

	trainDataFile = "trainingSet.dat"
	devDataFile = "developmentSet.dat"

	trainingFeatures = []
	# trainingFeatures.append(cooccurFeatures.getFeatures(trainDataFile))
	trainingFeatures.append(stopwordsfeature.getFeature(trainDataFile))
	trainingFeatures.append(miminumPCFGScore.getFeature(trainDataFile))
	trainingFeatures.append(cooccurFeatures.getFeatures2(trainDataFile))
	trainingFeatures.append(cooccurFeatures.getFeatures3(trainDataFile))
	trainingFeatures.append(cooccurFeatures.getFeatures4(trainDataFile))
	# trainingFeatures.append(cooccurFeatures.getFeatures6(trainDataFile))
	trainingFeatures.append(cooccurFeatures.getFeatures7(trainDataFile))
	trainingFeatures.append(cooccurFeatures.getFeatures8(trainDataFile))
	trainingFeatures.append(cooccurFeatures.getFeatures9(trainDataFile))
	
	# features.append(countLDA.feat_lda(trainDataFile))
	# features.append(posngrams.getFeature(sys.argv[1]))

	labels = np.asarray(getFakeGood('trainingSetLabels.dat'))
	Y = labels
	X = toArray(trainingFeatures)

	bdt = AdaBoostClassifier(svm.SVC(probability=True,kernel='linear'),n_estimators=50, learning_rate=1.0, algorithm='SAMME')
	bdt.fit(X, Y)


	devFeatures = []
	# devFeatures.append(cooccurFeatures.getFeatures(devDataFile))
	devFeatures.append(stopwordsfeature.getFeature(devDataFile))
	devFeatures.append(miminumPCFGScore.getFeature(devDataFile))
	devFeatures.append(cooccurFeatures.getFeatures2(devDataFile))
	devFeatures.append(cooccurFeatures.getFeatures3(devDataFile))
	devFeatures.append(cooccurFeatures.getFeatures4(devDataFile))
	# devFeatures.append(cooccurFeatures.getFeatures6(devDataFile))
	devFeatures.append(cooccurFeatures.getFeatures7(devDataFile))
	devFeatures.append(cooccurFeatures.getFeatures8(devDataFile))
	devFeatures.append(cooccurFeatures.getFeatures9(devDataFile))

	devLabels = np.asarray(getFakeGood('developmentSetLabels.dat'))
	devX = toArray(devFeatures)

	predicted = bdt.predict(devX)
	accuracy = accuracy_score(devLabels, predicted)
	print("Accuracy AdaBoost Classifier: %f" % accuracy)


	classifier = GaussianNB()
	classifier.fit(X, Y)
	predicted = classifier.predict(devX)
	print(predicted)
	accuracy = accuracy_score(devLabels, predicted)
	print("Accuracy Gaussian Naive Bayes Classifier: %f" % accuracy)
