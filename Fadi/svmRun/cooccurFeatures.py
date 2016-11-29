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


def getTrainingData(filename):
	docs = []
	doc = []
	with open(filename, 'r') as myfile:
		next(myfile)          #skip first line
		for l in myfile:
			if l.strip() == "~~~~~":
				docs.append(doc)
				doc = []
			else:
				sent = l.strip().split()[1:-1]
				sent = filter(lambda x: x != "<UNK>", sent)
				doc.append(sent)

		docs.append(doc)

	return docs

def getTrainingLabels(filename):
	labels = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			labels.append(int(l.strip()))
	return labels

def getVocabFromFile(filename):
	vocab = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			vocab.append(l.strip())

	return vocab

def listToDict(vocab):
	dict = {}
	for i in range(len(vocab)):
		dict[vocab[i]] = i

	return dict


def load_sparse_matrix(filename):
    y=np.load(filename)
    z=sparse.coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z


def sentenceLogLikelihood(vocabDict, model, sentence):
	count = 0.0
	logLikelihood = 0.0

	for i1 in range(len(sentence)):
		for i2 in range(len(sentence)):
			if i1 != i2:
				w1 = sentence[i1]
				w2 = sentence[i2]

				logLikelihood += log(model[vocabDict[w1],vocabDict[w2]] + 0.00000000000000001)
				count += 1


	if count == 0:
		print "PROBLEM"
		print sentence
		exit()

	return logLikelihood / count



def docLogLikelihood(vocabDict, model, doc):
	docLen = 0.0
	for s in doc:
		docLen += len(s)

	logLikelihood = 0.0
	
	for sentence in doc:
		if(len(sentence) < 2):
			continue

		sentLen = len(sentence)
		score = sentenceLogLikelihood(vocabDict,model, sentence)

		logLikelihood += (sentLen/docLen) * score

	return logLikelihood

#take the minumum sentence loglik instead of average
def docLogLikelihood2(vocabDict, model, doc):
	docLen = 0.0
	for s in doc:
		docLen += len(s)

	minSentLoglikelihood = 0.0
	minSentLoglikelihood_Normalized = 0.0
	
	for sentence in doc:
		if(len(sentence) < 2):
			continue

		sentLen = len(sentence)
		score = sentenceLogLikelihood(vocabDict,model, sentence)
		score_Normalized = (sentLen/docLen) * score

		if( score_Normalized < minSentLoglikelihood_Normalized):
			minSentLoglikelihood_Normalized = score_Normalized
			minSentLoglikelihood = score

		
	return minSentLoglikelihood	

#standard dev of sentence score
def docLogLikelihood3(vocabDict, model, doc):

	scores = []
	
	for sentence in doc:
		if(len(sentence) < 2):
			continue

		score = sentenceLogLikelihood(vocabDict,model, sentence)
		scores.append(score)

		
	return np.std(scores)

def docLogLikelihood_average_std(vocabDict, model, doc):
	docLen = 0.0
	for s in doc:
		docLen += len(s)

	logLikelihood = 0.0
	scores = []
	
	for sentence in doc:
		if(len(sentence) < 2):
			continue

		sentLen = len(sentence)
		score = sentenceLogLikelihood(vocabDict,model, sentence)
		scores.append(score)

		logLikelihood += (sentLen/docLen) * score

	return (logLikelihood, np.std(scores))		


def run(vocabDict, model, articles):
	scores = np.zeros([len(articles),2], dtype=float)

	for i in range(len(articles)):
		doc = articles[i]
		average, std = docLogLikelihood_average_std(vocabDict,model,doc)
		scores[i,0] = average
		scores[i,1] = std

	return scores

# def getFeatures(filename):
# 	trainingData = getTrainingData(filename)
# 	vocab = getVocabFromFile("./vocab.txt")
# 	vocabDict = listToDict(vocab)
# 	m =  load_sparse_matrix('wordCooccurProb-Final.npy.npz').tolil()

# 	scores = run(vocabDict, m ,trainingData)
# 	return scores


def getFeatures(filename):
	if filename == "trainingSet.dat":
		matrix = pickle.load( open( "./pickles/cooccurTrain.p", "rb" ) )
		# remove std
		# matrix = np.delete(matrix,1,1)
		# remove average
		# matrix = np.delete(matrix,0,1)
		return matrix
	elif filename == "developmentSet.dat":
		matrix = pickle.load( open( "./pickles/cooccurTest.p", "rb" ) )
		# remove std
		# matrix = np.delete(matrix,1,1)
		# remove average
		# matrix = np.delete(matrix,0,1)
		return matrix


def averageWordLength(doc):
	totalWordLength = 0.0
	wordCount = 0.0

	for line in doc:
		for w in line:
			totalWordLength += len(w)
			wordCount += 1.0

	return totalWordLength / wordCount

# average word length
def getFeatures2(filename):
	trainingData = getTrainingData(filename)
	scores = np.zeros([len(trainingData),1], dtype=float)

	for i in range(len(trainingData)):
		currDoc = trainingData[i]
		score = averageWordLength(currDoc)
		scores[i] = score

	return scores


def typeToken(doc):
	types = set()
	wordCount = 0.0

	for line in doc:
		for w in line:
			types.add(w)
			wordCount += 1.0

	return float(len(types)) / wordCount

# type / token 
def getFeatures3(filename):
	trainingData = getTrainingData(filename)
	scores = np.zeros([len(trainingData),1], dtype=float)

	for i in range(len(trainingData)):
		currDoc = trainingData[i]
		score = typeToken(currDoc)
		scores[i] = score

	return scores


def sentenceLengthStd(doc):
	lengths = []

	for line in doc:
		lengths.append(len(line))
		# for w in line:
			# lengths.append(len(w))

	return np.std(lengths)


# std of sentence length
def getFeatures4(filename):
	trainingData = getTrainingData(filename)
	scores = np.zeros([len(trainingData),1], dtype=float)

	for i in range(len(trainingData)):
		currDoc = trainingData[i]
		score = sentenceLengthStd(currDoc)
		scores[i] = score

	return scores


def firstWordTypeToken(doc):
	types = set()
	wordCount = 0.0

	for line in doc:
		if len(line) == 0:
			continue
		types.add(line[0])
		wordCount += 1.0

	return float(len(types)) / wordCount

# type / token for first word in  sentence
def getFeatures5(filename):
	trainingData = getTrainingData(filename)
	scores = np.zeros([len(trainingData),1], dtype=float)

	for i in range(len(trainingData)):
		currDoc = trainingData[i]
		score = firstWordTypeToken(currDoc)
		scores[i] = score

	return scores	


def wordSentenceOccurance(doc):
	vocab = defaultdict(float)
	for line in doc:
		for w in line:
			vocab[w] = 0.0

	for line in doc:
		for key in vocab.keys():
			if key in line:
				vocab[key] += 1.0

	return (sum(vocab.values()) / len(vocab)) / len(doc)

def getFeatures6(filename):
	trainingData = getTrainingData(filename)
	scores = np.zeros([len(trainingData),1], dtype=float)

	for i in range(len(trainingData)):
		currDoc = trainingData[i]
		score = wordSentenceOccurance(currDoc)
		scores[i] = score

	return scores	

def getFeatures7(filename):
	trainingData = getTrainingData(filename)
	scores = np.zeros([len(trainingData),1], dtype=float)

	for i in range(len(trainingData)):
		currDoc = trainingData[i]
		score = len(currDoc)
		scores[i] = score

	return scores	

def getFeatures8(filename):
	if filename == "trainingSet.dat":
		l = pickle.load( open( "./pickles/lm5gramsfeature.pkl", "rb" ) )
		matrix = np.zeros([len(l),1], dtype=float)
		for i in range(len(l)):
			matrix[i,0] = l[i]

		# remove std
		# matrix = np.delete(matrix,1,1)
		# remove average
		# matrix = np.delete(matrix,0,1)
		return matrix
	elif filename == "developmentSet.dat":
		l = pickle.load( open( "./pickles/lm5gramsfeatureDev.pkl", "rb" ) )
		matrix = np.zeros([len(l),1], dtype=float)
		for i in range(len(l)):
			matrix[i,0] = l[i]
		# remove std
		# matrix = np.delete(matrix,1,1)
		# remove average
		# matrix = np.delete(matrix,0,1)
		return matrix


# if __name__ == '__main__':
# 	trainingData = getTrainingData(sys.argv[1])
# 	labels = getTrainingLabels(sys.argv[2])
# 	trainingData = map(lambda (l,x): (l,x), zip(labels,trainingData) )

# 	vocab = getVocabFromFile("./vocab.txt")
# 	vocabDict = listToDict(vocab)
# 	m =  load_sparse_matrix('wordCooccurProb-Final.npy.npz').tolil()

# 	docScores = run(vocabDict, m, trainingData)

# 	pickle.dump( docScores, open( sys.argv[3], "wb" ) )