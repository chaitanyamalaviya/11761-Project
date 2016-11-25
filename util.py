from __future__ import division
from collections import defaultdict, Counter
from itertools import count, izip
import numpy as np

class Vocab:
	def __init__(self, w2i = None, termVector = None, avg_sent_len = 0):
		if w2i is None: w2i = defaultdict(count(0).next)
		if termVector is None: termVector = np.zeros(2)
		self.w2i = dict(w2i)
		self.i2w = {i:w for w,i in w2i.iteritems()}
		self.termVector =  termVector
		self.avg_sent_len = avg_sent_len

	def from_corpus(cls, corpus, article_words):
		w2i = defaultdict(count(0).next)
		wordFreq = Counter()
		termVector = np.zeros(len(article_words))
		for sent in corpus:
			[w2i[word] for word in sent if len(word.strip(" "))!=0 ]
			for word in sent.split(" "):
				wordFreq[word] += 1

		sentLenSum = 0
		for sent in corpus:
			sentLenSum += len(sent.split(" "))
			for word in sent.split(" "):
				termVector[article_words.index(word)] = wordFreq[word]

		avg_sent_len = sentLenSum/len(corpus)

		return Vocab(w2i, termVector, avg_sent_len)

	def size(self): return len(self.w2i.keys())

class CorpusReader:
	def __init__(self, fname):
		self.fname = fname
	def __iter__(self):
		for line in file(self.fname):
			line = line.strip().split()
			#line = [' ' if x == '' else x for x in line]
			yield line

class CharsCorpusReader:
	def __init__(self, fname, begin=None):
		self.fname = fname
		self.begin = begin
	def __iter__(self):
		begin = self.begin
		for line in file(self.fname):
			line = list(line)
			if begin:
				line = [begin] + line
			yield line
