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



def getVocab(filename):
	vocab = set()
	data = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			l = l.strip().split()
			l = l[1:-1]

			for w in l:
				vocab.add(w)

	return vocab

def getVocabFromFile(filename):
	vocab = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			vocab.append(l.strip())

	return vocab


def printToFile(vocab):
	f = open('vocab.txt','w')

	for v in vocab:
		f.write(v + '\n') 
	f.close()

def listToDict(vocab):
	dict = {}
	for i in range(len(vocab)):
		dict[vocab[i]] = i

	return dict

def buildMatrix2(dataFile, vocabDict):
	# matrix = np.zeros((len(vocabDict),len(vocabDict)))
	matrix = lil_matrix((len(vocabDict), len(vocabDict)), dtype=np.float)
	vocabTotal = [0.0] * len(vocabDict)

	print "Reading File..."
	with open(dataFile, 'r') as myfile:
		for l in myfile:
			print l
			l = l.strip().split()
			l = l[1:-1]

			for i1 in range(len(l)):
				w1 = l[i1]
				vocabTotal[vocabDict[w1]] += 1.0
				for i2 in range(len(l)):
					if i1 != i2:
						w1 = l[i1]						

						if w1 not in vocabDict or w2 not in vocabDict:
							continue

						matrix[vocabDict[w1],vocabDict[w2]] += 1.0

	print "Dividing..."
	for i in range(len(vocabDict)):
		for m in range(len(vocabDict)):
			matrix[i,m] /= vocabTotal[i]

	print "Saving..."
	np.save('wordCooccurProb.npy', matrix)
	np.savetxt('wordCooccurProb.txt', matrix)


def buildMatrix(dataFile, vocabDict):

	cooccurDict = {}
	totalsDict = defaultdict(float)

	for k in vocabDict.keys():
		cooccurDict[k] = {}
		totalsDict[k] = 0.0

	counter = 0
	print "Reading File..."
	with open(dataFile, 'r') as myfile:
		for l in myfile:
			# print l
			counter += 1
			if  counter % 100000 == 0:
				print counter

			l = l.strip().split()
			l = l[1:-1]

			for i1 in range(len(l)):
				w1 = l[i1]
				totalsDict[w1] += 1.0
				for i2 in range(len(l)):
					if i1 != i2:
						w2 = l[i2]
						if w2 in cooccurDict[w1]:
							cooccurDict[w1][w2] += 1.0
						else:
							cooccurDict[w1][w2] = 1.0


	# matrix = np.zeros((len(vocabDict),len(vocabDict)))
	matrix = lil_matrix((len(vocabDict), len(vocabDict)), dtype=np.float)

	print "Dividing..."
	counter = 0
	for key1,dic in cooccurDict.items():
		# print key1
		counter += 1
		if  counter % 100000 == 0:
			print counter

		for key2,count in dic.items():
			matrix[vocabDict[key1],vocabDict[key2]] = count / totalsDict[key1]

	# print "To sparse matrix..."
	# matrix = lil_matrix(matrix)
	print "Saving..."
	# # np.save('wordCooccurProb.npy', matrix)
	# # np.savetxt('wordCooccurProb.txt', matrix)	
	# # print matrix[vocabDict["I"],vocabDict["IRON"]]
	save_sparse_matrix('wordCooccurProb.npy', matrix)

def getMatrixFromFile(filename):
	matrix = np.load(filename, skiprows=0, ndmin=2)
	print matrix[0,0]

def save_sparse_matrix(filename,x):
    x_coo=x.tocoo()
    row=x_coo.row
    col=x_coo.col
    data=x_coo.data
    shape=x_coo.shape
    np.savez(filename,row=row,col=col,data=data,shape=shape)

def load_sparse_matrix(filename):
    y=np.load(filename)
    z=sparse.coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z


if __name__ == '__main__':
	vocab = getVocabFromFile("./vocab.txt")
	vocabDict = listToDict(vocab)
	# buildMatrix("./LM-train-100MW.txt", vocabDict)
	buildMatrix("./testText.txt", vocabDict)
	# getMatrixFromFile('wordCooccurProb.npy')
	m =  load_sparse_matrix('wordCooccurProb.npy.npz').tolil()
	print m[vocabDict["I"],vocabDict["MAN"]]
	# print m[vocabDict["GEORGE"],vocabDict["BUSH"]]
	# print m[vocabDict["BUSH"],vocabDict["GEORGE"]]
	# print m[vocabDict["WHITE"],vocabDict["HOUSE"]]
	# print m[vocabDict["POLICE"],vocabDict["FENCING"]]




