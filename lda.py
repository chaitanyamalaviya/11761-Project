#LDA + OTHER COUNT-BASED FEATURES
#TOP 10 MOST COMMON WORDS (BESIDES STOPWORDS)

from __future__ import division
from collections import defaultdict, Counter
from itertools import count, izip
import lda, util, re, pickle
import numpy as np
import matplotlib.pyplot as plt

def importScores(fileName):
	articlesList = []
	scoresList = []
	with open(fileName, "r") as f:
		lines = f.readlines()
	for line in lines:
		matchObj = re.match(r'^(\[.*?\]), (\[.*?\])', line, re.M | re.I)
		if matchObj:
			article = matchObj.group(1)
			scores = matchObj.group(2)
			article = eval(article)
			scores = eval(scores)
			scores = [float(i) for i in scores]
			articlesList. append(article)
			scoresList.append(scores)
	return articlesList, scoresList


def saveModel(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadModel(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

if __name__ == "__main__": 

	goodArticles, goodScores = importScores('goodArticles.txt')
	badArticles, badScores = importScores('badArticles.txt')
	
	good_article_words = []
	for article in goodArticles:
		for sent in article:
			for word in sent.split(" "):
				if word not in good_article_words:
					good_article_words.append(word)

	bad_article_words = []
	for article in badArticles:
		for sent in article:
			for word in sent.split(" "):
				if word not in bad_article_words:
					bad_article_words.append(word)


	docTerm_good = np.zeros((len(goodArticles), len(good_article_words)))
	docTerm_bad = np.zeros((len(badArticles), len(bad_article_words)))


	Vb = util.Vocab()
	
	good_sent_len_sum = 0
	bad_sent_len_sum = 0
	
	type_token_ratios_good = []
	type_token_ratios_bad = []

	i = 0
	for good_article, bad_article in izip(goodArticles, badArticles):
		
		vocab_good = Vb.from_corpus(good_article, good_article_words)
		vocab_bad = Vb.from_corpus(bad_article, bad_article_words)
		
		docTerm_good[i] = vocab_good.termVector
		type_token_ratios_good.append(np.count_nonzero(vocab_good.termVector)/np.sum(vocab_good.termVector))
		
		docTerm_bad[i] = vocab_bad.termVector
		type_token_ratios_bad.append(np.count_nonzero(vocab_bad.termVector)/np.sum(vocab_bad.termVector))
		
		good_sent_len_sum += vocab_good.avg_sent_len
		bad_sent_len_sum += vocab_bad.avg_sent_len
		
		i += 1

	print "Average Sentence Length for good articles: ", good_sent_len_sum/i
	print "Average Sentence Length for bad articles: ", bad_sent_len_sum/i

	print "Type Token ratios (GOOD): ", type_token_ratios_good
	print "Type Token ratios (BAD): ", type_token_ratios_bad

	print "Average Type Token Ratio for good articles: ", sum(type_token_ratios_good)/len(type_token_ratios_good)
	print "Average Type Token Ratio for bad articles: ", sum(type_token_ratios_bad)/len(type_token_ratios_bad)

	# goodModel = lda.LDA(n_topics=50, n_iter=1300, random_state=1)
	# goodModel.fit(docTerm_good.astype('int64'))
	# print "Model ll for good articles", goodModel.loglikelihood()

	# saveModel(goodModel, "lda-goodArticles.model")

	goodModel = loadModel("lda-goodArticles.model")
	print "Model ll for good articles", goodModel.loglikelihood()
	topic_word = goodModel.topic_word_
	n_top_words = 10
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(good_article_words)[np.argsort(topic_dist)][:-n_top_words:-1]
		print('Topic {}: {}'.format(i, ' '.join(topic_words)))


	#OPTIONAL
	# badModel = lda.LDA(n_topics=50, n_iter=1300, random_state=1)
	# badModel.fit(docTerm_bad.astype('int64'))
	# print "Model ll for bad articles", badModel.loglikelihood()
	# saveModel(badModel, "lda-badArticles.model")


	badModel = loadModel("lda-badArticles.model")
	print "Model ll for bad articles", badModel.loglikelihood()
	topic_word = badModel.topic_word_
	n_top_words = 10
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(bad_article_words)[np.argsort(topic_dist)][:-n_top_words:-1]
		print('Topic {}: {}'.format(i, ' '.join(topic_words)))

	plt.plot(goodModel.loglikelihoods_[5:], 'g')
	plt.plot(badModel.loglikelihoods_[5:], 'r')
	plt.ylabel("Log-likelihood")
	plt.xlabel("Iteration")
	plt.show()
	
