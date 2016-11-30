import os
import numpy as np
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn import svm
import handmadeLinguistic as hml
import arpalmevaluation as alme
import stopwordsfeature as swf

from sklearn.pipeline import Pipeline, FeatureUnion
#import skflow

def loadObj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def getFakeGood(labelsFileName):
    path = os.getcwd()
    with open(path + '/' + labelsFileName, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.rstrip()
        labels.append(int(line))
    return labels

#featureAlme = alme.getFeature1('trainingSet.dat')
featureMinScore = loadObj('min_score_feature')
featureNSents = loadObj('featureNSents')
featureLm5 = loadObj('lm5gramsfeature')
featureLm2 = loadObj('lm2gramsfeature')
featureLm3 = loadObj('lm3gramsfeature')
featurePos = loadObj('posFeatureTrain')
featurePos2 = loadObj('pos2Feature')
featureLm7 = loadObj('lm7gramsfeatureTrain')
featureHml = loadObj('handMadeLinguisticFeature')
featureSw = loadObj('feature_stopwords')
featureHmlTotal = loadObj('hmlFeaturesMatrix')
featureTypeToken = loadObj('featureTyTo')
featureAvgLength = loadObj('featureAvgLength')
#X = np.array([featureAlme, featurePos, featureLm7])
#X = np.array([featureLm5, featureHml, featureSw, featurePos, featureLm2])

HML = np.array([np.array(xi) for xi in featureHmlTotal])
HML = HML.transpose()

X = np.array([featureLm2, featureSw, featurePos, featureLm5, featurePos2, featureLm7, featureTypeToken, featureNSents ])
X = X.transpose()

X = np.column_stack((HML,X))


#devData = np.asarray(loadObj('min_feature_dev'))
#X1 = np.array([featureAlme])
#X1 = X1.reshape([1000,1])

#X2 = np.array([featurePos])
#X2 = X2.reshape([1000,1])

devMinScore = loadObj('min_feature_dev')
devDataLm5 = loadObj('lm5gramsfeatureDev')
devDataPos = loadObj('posFeatureDev')
devDataPos2 = loadObj('pos2FeatureDev')
devDataLm7 = loadObj('lm7gramsfeatureDev')
devDataLm2 = loadObj('lm2gramsfeatureDev')
devDataLm3 = loadObj('lm3gramsfeatureDev')
devDataTyTo = loadObj('featureTyToDev')
#devDataHml = hml.getFeature('developmentSet.dat')
devDataSwf = swf.getFeature('developmentSet.dat')
devDataHml = loadObj('hmlFeaturesMatrixDev')
devDataAvgLength = loadObj('featureAvgLengthDev')
devDataNSents = loadObj('featureNSentsDev')
devDataHml = np.array([np.array(xi) for xi in devDataHml])
devDataHml = devDataHml.transpose()
#devDataAlme = alme.getFeature1('developmentSet.dat')
#devDataAlme = np.array([devDataAlme])
#devDataAlme = devDataAlme.reshape([200,1])
#devDataAlme = alme.getFeature1('developmentSet.dat')
#devDataPos = np.array([devDataPos])
#devX = np.array([devDataAlme, devDataPos, devDataLm7])
#devX = np.array([devDataLm5, devDataHml, devDataSwf, devDataPos, devDataLm2])

devX = np.array([devDataLm2, devDataSwf, devDataPos, devDataLm5, devDataPos2, devDataLm7, devDataTyTo, devDataNSents])
devX = devX.transpose()

devX = np.column_stack((devDataHml,devX))

labels = np.asarray(getFakeGood('trainingSetLabels.dat'))
Y = labels

devLabels = np.asarray(getFakeGood('developmentSetLabels.dat'))

bdt = AdaBoostClassifier(svm.SVC(probability=True,kernel='linear'),n_estimators=50, learning_rate=1, algorithm='SAMME')
#bdt = AdaBoostClassifier(GaussianNB(),n_estimators=50, learning_rate=1.0, algorithm='SAMME')

bdt.fit(X, Y)

#clf = svm.SVC()
#clf.fit(X, Y)



#predicted = clf.predict(devX)

#accuracy = accuracy_score(devLabels, predicted)
#print(accuracy)
predicted = bdt.predict(devX)
#predicted = bdt.predict(X)
print(predicted)
accuracy = accuracy_score(devLabels, predicted)
print("Accuracy AdaBoost Classifier: %f" % accuracy)


classifier = GaussianNB()
classifier.fit(X, Y)
predicted = classifier.predict(devX)
#predicted = classifier.predict(X)

accuracy = accuracy_score(devLabels, predicted)
print("Accuracy Gaussian Naive Bayes Classifier: %f" % accuracy)


# classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
# classifier.fit(X,Y)
# predicted = classifier.predict(devX)
# accuracy = accuracy_score(devLabels, predicted)
# print(accuracy)
