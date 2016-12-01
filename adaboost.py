from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
import countLDA
import os
import re
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

excludedList = ['pcfg', 'mpcfg', 'occ2']

def importSingleFeatures(datasetPath):
    featuresList = os.listdir('features/'+datasetPath+'/single')
    datasetPathPrefix = 'features/'+datasetPath
    i = 0
    singleFeaturesTrain = ''
    singleFeaturesDev = ''
    for feature in featuresList:
        if feature not in excludedList:
            path = datasetPathPrefix+'/single/'+feature
            picklesList = os.listdir(path)
            for file in picklesList:
                match = re.match(r'(.*Dev.*)\.pkl', file)
                if (match):
                    devPickleFileName = match.group(1)
                    devPickle = loadObj(path+'/'+devPickleFileName)
                else:
                    match = re.match(r'(.*)\.pkl', file)
                    if match:
                        trainPickleFileName = match.group(1)
                        trainPickle = loadObj(path+'/'+trainPickleFileName)
            #print("Adding feature: %s" % feature)
            if i == 0:
                singleFeaturesTrain = np.array([trainPickle]).transpose()
                singleFeaturesDev = np.array([devPickle]).transpose()
                i += 1
            else:
                tempTrain = np.array([trainPickle]).transpose()
                tempDev = np.array([devPickle]).transpose()
                singleFeaturesTrain = np.column_stack((singleFeaturesTrain, tempTrain))
                singleFeaturesDev = np.column_stack((singleFeaturesDev, tempDev))
                i += 1
    return singleFeaturesTrain, singleFeaturesDev

def importMultipleFeatures(datasetPath):
    featuresList = os.listdir('features/'+datasetPath+'/multiple')
    datasetPathPrefix = 'features/' + datasetPath
    i = 0
    multipleFeaturesTrain = ''
    multipleFeaturesDev = ''
    for feature in featuresList:
        if feature not in excludedList:
            path = datasetPathPrefix+'/multiple/' + feature
            picklesList = os.listdir(path)
            for file in picklesList:
                match = re.match(r'(.*Dev.*)\.pkl', file)
                if (match):
                    devPickleFileName = match.group(1)
                    devPickle = loadObj(path + '/' + devPickleFileName)
                else:
                    match = re.match(r'(.*)\.pkl', file)
                    if match:
                        trainPickleFileName = match.group(1)
                        trainPickle = loadObj(path + '/' + trainPickleFileName)
            #print("Adding feature: %s" % feature)
            if i == 0:
                multipleFeaturesTrain = np.array([np.array(xi) for xi in trainPickle]).transpose()
                multipleFeaturesDev = np.array([np.array(xi) for xi in devPickle]).transpose()
                i += 1
            else:
                tempTrain = np.array([np.array(xi) for xi in trainPickle]).transpose()
                tempDev = np.array([np.array(xi) for xi in devPickle]).transpose()
                multipleFeaturesTrain = np.column_stack((multipleFeaturesTrain, tempTrain))
                multipleFeaturesDev = np.column_stack((multipleFeaturesDev, tempDev))
                i += 1
    return multipleFeaturesTrain, multipleFeaturesDev

def importArrayFeatures(datasetPath):
    featuresList = os.listdir('features/'+datasetPath+'/arrays')
    i = 0
    datasetPathPrefix = 'features/' + datasetPath
    arraysFeaturesTrain = 0
    arraysFeaturesDev = 0
    for feature in featuresList:
        path = datasetPathPrefix + '/arrays/' + feature
        if feature not in excludedList:
            picklesList = os.listdir(path)
            for file in picklesList:
                match = re.match(r'(.*Dev.*)\.pkl', file)
                if (match):
                    devPickleFileName = match.group(1)
                    devPickle = loadObj(path + '/' + devPickleFileName)
                else:
                    match = re.match(r'(.*)\.pkl', file)
                    if match:
                        trainPickleFileName = match.group(1)
                        trainPickle = loadObj(path + '/' + trainPickleFileName)
            if i == 0:
                match = re.match(r'.*(notranspose).*', file)
                if match:
                    print("I am not transposing the feature %s" % feature)
                    arraysFeaturesTrain = trainPickle
                    arraysFeaturesDev = devPickle
                else:
                    arraysFeaturesTrain = trainPickle.transpose()
                    arraysFeaturesDev = devPickle.transpose()
                i += 1
            else:
                match = re.match(r'.*(notranspose).*', file)
                if match:
                    print("I am not transposing the feature %s" % feature)
                    tempTrain = trainPickle
                    tempDev = devPickle
                    arraysFeaturesTrain = np.column_stack((arraysFeaturesTrain, tempTrain))
                    arraysFeaturesDev = np.column_stack((arraysFeaturesDev, tempDev))

                else:
                    tempTrain = trainPickle.transpose()
                    tempDev = devPickle.transpose()
                    arraysFeaturesTrain = np.column_stack((arraysFeaturesTrain, tempTrain))
                    arraysFeaturesDev = np.column_stack((arraysFeaturesDev, tempDev))
                i += 1
    return arraysFeaturesTrain, arraysFeaturesDev

def buildFeatures(datasetPath):
    singleFeaturesTrain, singleFeaturesDev = importSingleFeatures(datasetPath)
    multipleFeaturesTrain, multipleFeaturesDev = importMultipleFeatures(datasetPath)
    arrayFeaturesTrain, arrayFeaturesDev = importArrayFeatures(datasetPath)

    featuresTrain = np.column_stack((singleFeaturesTrain, multipleFeaturesTrain, arrayFeaturesTrain))
    featuresDev = np.column_stack((singleFeaturesDev, multipleFeaturesDev, arrayFeaturesDev))
    return featuresTrain, featuresDev

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

def adaBoostClassifier(X, Y, devX, devLabels):
    bdt = AdaBoostClassifier(svm.SVC(probability=True, kernel='linear'), n_estimators=50, learning_rate=.1, algorithm='SAMME')
    bdt.fit(X, Y)
    predicted = bdt.predict(devX)
    print(predicted)
    with open('forfadi', 'w') as f:
        for i in range(0,200):
            strP = "%s\n" % str(predicted[i])
            f.write(strP)
    accuracy = accuracy_score(devLabels, predicted)
    print("Accuracy AdaBoost Classifier: %f" % accuracy)

def logisticRegression(X, Y, devX, devLabels):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    logreg_predictions = logreg.predict(devX)
    accuracy = accuracy_score(devLabels, logreg_predictions)
    print("Accuracy LogReg Classifier: %f" % accuracy)

def main():
    featuresTrain, featuresDev = buildFeatures('set1')
    labels = np.asarray(getFakeGood('expandedTrainingSetLabels2.txt'))
    devLabels = np.asarray(getFakeGood('developmentSetLabels.dat'))
    X = featuresTrain
    Y = labels
    devX = featuresDev
    adaBoostClassifier(X, Y, devX, devLabels )
    logisticRegression(X, Y, devX, devLabels )
    adaBoostClassifier(X, Y, X, labels )
    logisticRegression(X, Y, X, labels )

if __name__ == "__main__": main()


"""
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
#featureTypeToken = loadObj('featureTyTo')
featureTypeToken = countLDA.feat_type_token_ratio('trainingSet.dat')
featureAvgSentLength = countLDA.feat_avg_sent_len('trainingSet.dat')
featureAvgLength = loadObj('featureAvgLength')
featureKenLm5 = loadObj('kemlm5')
featureKenLm4 = loadObj('kemlm4')
#X = np.array([featureAlme, featurePos, featureLm7])
#X = np.array([featureLm5, featureHml, featureSw, featurePos, featureLm2])

HML = np.array([np.array(xi) for xi in featureHmlTotal])
HML = HML.transpose()

X = np.array([featureLm2, featureSw, featurePos, featureLm5, featurePos2, featureLm7, featureKenLm5 ])
#X = np.array([featureKenLm4, featureKenLm5, featureSw, featureLm2, featurePos2, featureLm5])
X = X.transpose()

X = np.column_stack((HML,X, featureTypeToken.transpose()))
#X = np.column_stack((X, featureTypeToken.transpose()))



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
#devDataTyTo = loadObj('featureTyToDev')
#devDataHml = hml.getFeature('developmentSet.dat')
devDataSwf = swf.getFeature('developmentSet.dat')
devDataHml = loadObj('hmlFeaturesMatrixDev')
devDataAvgLength = loadObj('featureAvgLengthDev')
devDataNSents = loadObj('featureNSentsDev')
devDataKenLm5 = loadObj('kemlm5Dev')
devDataKenLm4 = loadObj('kemlm4Dev')
devDataTyTo = countLDA.feat_type_token_ratio('developmentSet.dat')
devDataAvgSentLength = countLDA.feat_avg_sent_len('developmentSet.dat')
devDataHml = np.array([np.array(xi) for xi in devDataHml])
devDataHml = devDataHml.transpose()
#devDataAlme = alme.getFeature1('developmentSet.dat')
#devDataAlme = np.array([devDataAlme])
#devDataAlme = devDataAlme.reshape([200,1])
#devDataAlme = alme.getFeature1('developmentSet.dat')
#devDataPos = np.array([devDataPos])
#devX = np.array([devDataAlme, devDataPos, devDataLm7])
#devX = np.array([devDataLm5, devDataHml, devDataSwf, devDataPos, devDataLm2])

devX = np.array([devDataLm2, devDataSwf, devDataPos, devDataLm5, devDataPos2, devDataLm7, devDataKenLm5])
#devX = np.array([devDataKenLm4, devDataKenLm5, devDataSwf, devDataLm2, devDataPos2, devDataLm5])
devX = devX.transpose()

devX = np.column_stack((devDataHml,devX, devDataTyTo.transpose()))
#devX = np.column_stack((devX, devDataTyTo.transpose()))

labels = np.asarray(getFakeGood('trainingSetLabels.dat'))
Y = labels

#X_new = SelectKBest(chi2, k=5).fit_transform(X,Y)



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



logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
logreg_predictions = logreg.predict(devX)
accuracy = accuracy_score(devLabels, logreg_predictions)
print("Accuracy LogReg Classifier: %f" % accuracy)
"""