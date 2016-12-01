#LDA + OTHER COUNT-BASED FEATURES
#TOP 10 MOST COMMON WORDS (BESIDES STOPWORDS)

from __future__ import division
from sklearn.preprocessing import normalize
from collections import defaultdict, Counter
from itertools import count, izip
from nltk.corpus import stopwords
import lda, util, re, pickle, os
import nltk.tokenize
import numpy as np
import matplotlib.pyplot as plt


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

    return articlesList

def saveModel(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadModel(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)


def feat_type_token_ratio(input_data_file):
	articles = importArticles(input_data_file)
	article_words = []
	for article in articles:
		for sent in article:
			for word in sent.split(" "):
				if word not in article_words:
					article_words.append(word)

	Vb = util.Vocab()
	type_token_ratios = []
	top = 20
	termVectorAll = np.zeros(len(article_words))

	for article in articles:
		vocab = Vb.from_corpus(article, article_words)
		type_token_ratios.append(np.count_nonzero(vocab.termVector)/np.sum(vocab.termVector))

	return np.matrix(type_token_ratios)


def feat_avg_sent_len(input_data_file):

	articles = importArticles(input_data_file)

	article_words = []
	for article in articles:
		for sent in article:	
			for word in sent.split(" "):
				if word not in article_words:
					article_words.append(word)

	avg_sent_len_articles = []

	Vb = util.Vocab()
	for i, article in enumerate(articles):
		vocab = Vb.from_corpus(article, article_words)		
		avg_sent_len_articles.append(vocab.avg_sent_len)		

	return np.transpose(np.matrix(avg_sent_len_articles))


def feat_avg_word_len(input_data_file):

	articles = importArticles(input_data_file)
	avg_word_len_articles = []

	for article in articles:
		article_words = []
		for sent in article:
			for word in sent.split(" "):
				if word not in article_words:
					article_words.append(word)

		avg_word_len_articles.append(sum([len(word) for word in article_words])/len(article_words))

	return np.transpose(np.matrix(avg_word_len_articles))


def feat_lda(input_data_file):

	# RETRIEVE ALL WORDS IN GOOD ARTICLES
	goodArticles = importScores('goodArticles.txt')
	
	good_article_words = []
	for article in goodArticles:
		for sent in article:
			for word in sent.split(" "):
				if word not in good_article_words:
					good_article_words.append(word)

	#LOAD GOOD LDA MODEL
	ldaModel = loadModel("lda-goodArticles.model")
	topic_word = ldaModel.topic_word_
	n_top_words = 20
	n_topics = 50
	topic_words = []

	for i, topic_dist in enumerate(topic_word, start = 0):
		topic_words.append(np.array(good_article_words)[np.argsort(topic_dist)][:-n_top_words:-1])

	# TEST ARTICLE
	articles = importArticles(input_data_file)
	topic_coverage = np.zeros((len(articles), n_topics))
	article_words = []
	
	for article in articles:	
		for sent in article:
			for word in sent.split(" "):
				if word not in article_words:
					article_words.append(word)

	Vb = util.Vocab()
	docTerm = np.zeros((len(articles), len(article_words)))

	# TOPIC COVERAGE CALCULATION
	# for i, article in enumerate(articles):
	# 	vocab = Vb.from_corpus(article, article_words)
	# 	docTerm[i] = vocab.termVector
		
	# 	for j, topic_list in enumerate(topic_words):
	# 		for k, word in enumerate(topic_list):
	# 			if word in vocab.w2i:
	# 				idx = vocab.w2i[word]
	# 				topic_coverage[i, j] += docTerm[i][idx] * (n_top_words - k)

	doc_topic_test = ldaModel.transform(docTerm.astype('int64'))
	normed_doc_topic_test = np.empty((len(articles), n_topics))
	for i in range(doc_topic_test.shape[0]):
		rsum = np.sum(doc_topic_test[i])
		if (rsum==0):
			normed_doc_topic_test[i] = doc_topic_test[i]
			continue
		normed_doc_topic_test[i] = np.array([e/rsum for e in doc_topic_test[i]])
	# normed_matrix = normalize(doc_topic_test, axis=1, norm='l1')

	# topic_coverage[:,n_topics:] = doc_topic_test

	return doc_topic_test


def feat_most_common_words(input_data_file):

	articles = importArticles(input_data_file)
	top = 10

	mcw_feature = np.zeros(shape=(len(articles),2))
	article_words = []

	for article in articles:
		for sent in article:
			for word in sent.split(" "):
				if word not in article_words:
					article_words.append(word)

	Vb = util.Vocab()

	for i, article in enumerate(articles):
		vocab = Vb.from_corpus(article, article_words)
		termVector = vocab.termVector
		idxs = np.argsort(termVector)[::-1]
		top_words = []
		for idx in idxs:
			if len(top_words) == top:
				break
			if article_words[idx].lower() not in (stopwords.words("english") + ["<unk>"]):
				top_words.append(article_words[idx])
		mcw_feature[i][0] = sum([len(word) for word in top_words])/top
		mcw_feature[i][1] = sum([termVector[article_words.index(word)] for word in top_words])/(top * np.count_nonzero(termVector))

	return mcw_feature
	

# def feat_tf_idf(input_data_file):
	
# 	articles = importArticles(input_data_file)
# 	tfidf_articles = np.array(shape=(len(articles),1))
	
# 	article_words = []

# 	for article in articles:
# 		for sent in article:
# 			for word in sent.split(" "):
# 				if word not in article_words:
# 					article_words.append(word)

# 	Vb = util.Vocab()

# 	for article in articles:
# 		vocab = Vb.from_corpus(article, article_words)
# 		termVector = vocab.termVector




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

	top = 20
	termVector_good = np.zeros(len(good_article_words))
	termVector_bad = np.zeros(len(bad_article_words))

	i = 0
	for good_article, bad_article in izip(goodArticles, badArticles):
		
		vocab_good = Vb.from_corpus(good_article, good_article_words)
		vocab_bad = Vb.from_corpus(bad_article, bad_article_words)
		
		docTerm_good[i] = vocab_good.termVector
		type_token_ratios_good.append(np.count_nonzero(vocab_good.termVector)/np.sum(vocab_good.termVector))
		
		termVector_good += vocab_good.termVector

		docTerm_bad[i] = vocab_bad.termVector
		type_token_ratios_bad.append(np.count_nonzero(vocab_bad.termVector)/np.sum(vocab_bad.termVector))

		termVector_bad += vocab_bad.termVector
		
		good_sent_len_sum += vocab_good.avg_sent_len
		bad_sent_len_sum += vocab_bad.avg_sent_len
		
		i += 1

	wordLengthSum_good = sum([len(word) for word in good_article_words])/len(good_article_words)
	wordLengthSum_bad = sum([len(word) for word in bad_article_words])/len(bad_article_words)

	print "Average word length from good articles", wordLengthSum_good
	print "Average word length from bad articles", wordLengthSum_bad

	print "Average Sentence Length for good articles: ", good_sent_len_sum/i
	print "Average Sentence Length for bad articles: ", bad_sent_len_sum/i


	idxs = np.argsort(termVector_good)[::-1]
	top_words_good = []
	for idx in idxs:
		if len(top_words_good) == top:
			break
		if good_article_words[idx].lower() not in (stopwords.words("english") + ["<unk>"]):
			top_words_good.append(good_article_words[idx])

	print "Top 10 words from good articles", top_words_good

	idxs = np.argsort(termVector_bad)[::-1]
	top_words_bad = []
	for idx in idxs:
		if len(top_words_bad) == top:
			break
		if bad_article_words[idx].lower() not in (stopwords.words("english") + ["<unk>"]):
			top_words_bad.append(bad_article_words[idx])

	print "Top 10 words from bad articles", top_words_bad
	
	# print "Type Token ratios (GOOD): ", type_token_ratios_good
	# print "Type Token ratios (BAD): ", type_token_ratios_bad

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
	
