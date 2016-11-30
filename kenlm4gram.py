import os
import kenlm
import numpy as np

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


def feat_kenlm(input_data_file):

    articles = importArticles(input_data_file)
    scoreList = []
    for article in articles:
        cum_article_score = 0.0
        for sent in article:
			cum_article_score = model.score(sentence)
        scoreList.append(cum_article_score/len(article))

    print scoreList
    return np.array(scoreList)

LM = os.path.join(os.path.dirname(__file__), '..','..', 'kenlm-4gram.bin')
model = kenlm.Model(LM)
print('{0}-gram model'.format(model.order))
feat_kenlm('developmentSet.dat')

# sentence = 'language modeling is fun .'.upper()
# print(sentence)
# print(model.score(sentence))
# sentence = 'what apple green boy ?'.upper()
# print(sentence)
# print(model.score(sentence))



# Check that total full score = direct score
def score(s):
    return sum(prob for prob, _, _ in model.full_scores(s))

assert (abs(score(sentence) - model.score(sentence)) < 1e-3)




###### IGNORE BELOW CODE FOR NOW ############



# Show scores and n-gram matches
words = ['<s>'] + sentence.split() + ['</s>']
for i, (prob, length, oov) in enumerate(model.full_scores(sentence)):
    print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
    if oov:
        print('\t"{0}" is an OOV'.format(words[i+1]))

# Find out-of-vocabulary words
for w in words:
    if not w in model:
        print('"{0}" is an OOV'.format(w))

#Stateful query
state = kenlm.State()
state2 = kenlm.State()
#Use <s> as context.  If you don't want <s>, use model.NullContextWrite(state).
model.BeginSentenceWrite(state)
accum = 0.0
accum += model.BaseScore(state, "a", state2)
accum += model.BaseScore(state2, "sentence", state)
#score defaults to bos = True and eos = True.  Here we'll check without the end
#of sentence marker.
assert (abs(accum - model.score("a sentence", eos = False)) < 1e-3)
accum += model.BaseScore(state, "</s>", state2)
assert (abs(accum - model.score("a sentence")) < 1e-3)
