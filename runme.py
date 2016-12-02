import sys
import os
import re
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn import svm
import pcfgparsing as pcfg
import handmadeLinguistic as hml
import arpalmevaluation as alme
import stopwordsfeature as swf
import posngrams as png
#import typetokenfeature as tyto
import countLDA as clda
import parsePCFGTrainingFiles as PCFG
import miminumPCFGScore as mPCFG
import kenlmfeature as klm
import pickle
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

def loadObj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def saveObj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

from sklearn.pipeline import Pipeline, FeatureUnion
#import skflow

#includedList = ['hml', 'lm2', 'lm3', 'lm4', 'lm5', 'lm7', 'kenlm4', 'kenlm5', 'swf', 'png5', 'png2', 'tyto', 'occ1', 'occ2']
includedList = ['hml', 'lm2', 'lm3', 'lm4', 'lm5', 'lm7', 'swf', 'png5', 'png2', 'tyto', 'klm5']

def getFeatures(trainFileName):
    file = trainFileName
    featuresSet = None
    if 'lm2' in includedList:
        logging.debug("Creating the lm2 feature")
        lm2gramsfeature = alme.getFeature(file, 'LM/lm2grams.binlm')
        featuresSet = stackFeature(featuresSet, lm2gramsfeature, 'single')
        saveObj(lm2gramsfeature, '/tmp/lm2')
    if 'lm3' in includedList:
        logging.debug("Creating the lm3 feature")
        lm3gramsfeature = alme.getFeature(file, 'LM/lm3grams.binlm')
        featuresSet = stackFeature(featuresSet, lm3gramsfeature, 'single')
        saveObj(lm3gramsfeature, '/tmp/lm3')
    if 'lm4' in includedList:
        logging.debug("Creating the lm4 feature")
        lm4gramsfeature = alme.getFeature(file, 'LM/lm4grams.binlm')
        featuresSet = stackFeature(featuresSet, lm4gramsfeature, 'single')
        saveObj(lm4gramsfeature, '/tmp/lm4')
    if 'lm5' in includedList:
        logging.debug("Creating the lm5 feature")
        lm5gramsfeature = alme.getFeature(file, 'LM/lm5grams.binlm')
        featuresSet = stackFeature(featuresSet, lm5gramsfeature, 'single')
        saveObj(lm5gramsfeature, '/tmp/lm5')
    if 'lm7' in includedList:
        logging.debug("Creating the lm7 feature")
        lm7gramsfeature = alme.getFeature(file, 'LM/lm7grams.binlm')
        featuresSet = stackFeature(featuresSet, lm7gramsfeature, 'single')
        saveObj(lm7gramsfeature, '/tmp/lm7')
    if 'png2' in includedList:
        logging.debug("Creating the png2 feature")
        poslm2gramsfeature = png.getFeature(file, 'LM/pos2grams.binlm')
        featuresSet = stackFeature(featuresSet, poslm2gramsfeature, 'single')
        saveObj(poslm2gramsfeature, '/tmp/png2')
    if 'png5' in includedList:
        logging.debug("Creating the png5 feature")
        poslm5gramsfeature = png.getFeature(file, 'LM/pos5grams.binlm')
        featuresSet = stackFeature(featuresSet, poslm5gramsfeature, 'single')
        saveObj(poslm5gramsfeature, '/tmp/png5')
    if 'swf' in includedList:
        logging.debug("Creating the stopword feature")
        stopwordsfeature = swf.getFeature(file)
        featuresSet = stackFeature(featuresSet, stopwordsfeature, 'single')
        saveObj(stopwordsfeature, '/tmp/swf')
    if 'klm4' in includedList:
        logging.debug("Creating the min kenlm5 feature")
        kenlm4feature = klm.getFeature(file, 'kenlm-4gram.bin')
        featuresSet = stackFeature(featuresSet, kenlm5feature, 'single')
        saveObj(kenlm4feature, '/tmp/klm4')
    if 'klm5' in includedList:
        logging.debug("Creating the min kenlm5 feature")
        kenlm5feature = klm.getFeature(file, 'kenlm-5gram.bin')
        featuresSet = stackFeature(featuresSet, kenlm5feature, 'single')
        saveObj(kenlm5feature, '/tmp/klm5')
    if 'hml' in includedList:
        logging.debug("Creating the hml feature")
        hmlFeature = hml.getFeature(file)
        featuresSet = stackFeature(featuresSet, hmlFeature, 'multiple')
        saveObj(hmlFeature, '/tmp/hml')
    if 'tyto' in includedList:
        logging.debug("Creating the typetoken feature")
        # typetokenfeature = tyto.getFeature(file)
        typetokenfeature = clda.feat_type_token_ratio(file)
        featuresSet = stackFeature(featuresSet, typetokenfeature, 'array')
        saveObj(typetokenfeature, '/tmp/tyto')
    if 'pcfg' in includedList:
        logging.debug("Creating the pcgfscore feature")
        pcfgavgfeature = PCFG.getFeature(file)
        featuresSet = stackFeature(featuresSet, pcfgavgfeature, 'single')
    if 'mpcfg' in includedList:
        logging.debug("Creating the min pcgfscore feature")
        minpcfgscorefeature = mPCFG.getFeature(file)
        featuresSet = stackFeature(featuresSet, minpcfgscorefeature, 'single')
    if 'klm4' in includedList:
        logging.debug("Creating the min kenlm4 feature")
        kenlm4feature = klm.getFeature(file, 'kenlm-4gram.bin')
        featuresSet = stackFeature(featuresSet, kenlm4feature, 'single')
    return featuresSet

def getFakeGood(labelsFileName):
    path = os.getcwd()
    with open(path + '/' + labelsFileName, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.rstrip()
        labels.append(int(line))
    return labels

def logisticRegression(X, Y, devX, devLabels):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    logreg_predictions = logreg.predict(devX)
    accuracy = accuracy_score(devLabels, logreg_predictions)
    logging.debug("Accuracy LogReg Classifier: %f" % accuracy)

def stackFeature(featureSet, feature, featureType):
    if featureType == 'single':
        if featureSet == None:
            stackedFeature = np.array([feature]).transpose()
        else:
            tempStackedFeature = np.array([feature]).transpose()
            stackedFeature = np.column_stack((featureSet, tempStackedFeature))
    if featureType == 'multiple':
        if featureSet == None:
            stackedFeature = np.array([np.array(xi) for xi in feature]).transpose()
        else:
            tempStackedFeature = np.array([np.array(xi) for xi in feature]).transpose()
            stackedFeature = np.column_stack((featureSet, tempStackedFeature))
    if featureType == 'array':
        if featureSet == None:
            stackedFeature = feature.transpose()
        else:
            tempStackedFeature = feature.transpose()
            stackedFeature = np.column_stack((featureSet, tempStackedFeature))
    return stackedFeature

def main():
    numberOfArguments = len(sys.argv)
    path = os.getcwd()
    if (numberOfArguments < 3):
        print("This command takes two arguments: testSet.dat and testOutput.dat ")
        exit()
    else:
        testSetFileName = sys.argv[1]
        testOutputFileNAme = sys.argv[2]
        # logging.debug("This is the trainset file name: %s" % trainFileName)
        # logging.debug("This is the test file name: %s" % testFileName)
    if not (os.path.isfile('./'+testSetFileName)):
        print("The testSet file %s is not present in the directory %s" % (testSetFileName, path+'/'+testSetFileName))
        print("You are supposed to give the name of the file only, not the path.")
        exit()

    testX = getFeatures(testSetFileName )
    saveObj(testX, '/tmp/testX')
    numberOfArticles = testX.shape[0]
    bdt = loadObj('adaBoostTrained93.model')
    predicted = bdt.predict(testX)

    with open(testOutputFileNAme, 'w') as f:
        for i in range(0,numberOfArticles):
            strP = "%s\n" % str(predicted[i])
            f.write(strP)

    devLabels = np.asarray(getFakeGood('developmentSetLabels.dat'))
    accuracy = accuracy_score(devLabels, predicted)
    print(accuracy)

if __name__ == "__main__": main()



"""

"""