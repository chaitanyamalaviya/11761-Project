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

def saveObj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

"""
featureMinScore = loadObj('min_score_feature')
featureNSents = loadObj('featureNSents')
featureHmlTotal = loadObj('hmlFeaturesMatrix')
featureTypeToken = loadObj('featureTyTo')
featureAvgLength = loadObj('featureAvgLength')
"""

def main():

    trainingFileName = 'expandedTrainingSet3.txt'
    devsetFileName = 'developmentSet.dat'
    fileNames = [ trainingFileName, devsetFileName ]
    datasetPath = 'features/set3/'
    i = 0
    for file in fileNames:
        print("Creating the hml feature")
        hmlFeature = hml.getFeature(file)
        print("Creating the lm2 feature")
        lm2gramsfeature = alme.getFeature(file, 'LM/lm2grams.binlm')
        print("Creating the lm3 feature")
        lm3gramsfeature = alme.getFeature(file, 'LM/lm3grams.binlm')
        print("Creating the lm4 feature")
        lm4gramsfeature = alme.getFeature(file, 'LM/lm4grams.binlm')
        print("Creating the lm5 feature")
        lm5gramsfeature = alme.getFeature(file, 'LM/lm5grams.binlm')
        print("Creating the lm7 feature")
        lm7gramsfeature = alme.getFeature(file, 'LM/lm7grams.binlm')
        print("Creating the pos5gram feature")
        poslm5gramsfeature = png.getFeature(file, 'LM/pos5grams.binlm')
        print("Creating the pos2grams feature")
        poslm2gramsfeature = png.getFeature(file, 'LM/pos2grams.binlm')
        print("Creating the stopword feature")
        stopwordsfeature = swf.getFeature(file)
        print("Creating the typetoken feature")
        #typetokenfeature = tyto.getFeature(file)
        typetokenfeature = clda.feat_type_token_ratio(file)
        #print("Creating the pcfgAverage feature")
        #pcfgavgfeature = pcfg.getFeature(file)
        #print("Creating the minpcfg feature")
        #minpcfgscorefeature = pcfg.getFeature(file)
        print("Creating the kenlm4 feature")
        kenlm4feature = klm.getFeature(file, 'kenlm-4gram.bin')
        print("Creating the kenlm5 feature")
        kenlm5feature = klm.getFeature(file, 'corpuslm5grams.binary')

        if i == 1:
            file = file+'Dev'

        saveObj(hmlFeature, datasetPath+'/multiple/hml/'+file)
        saveObj(lm2gramsfeature, datasetPath+'/single/lm2/'+file)
        saveObj(lm3gramsfeature, datasetPath+'/single/lm3/'+file)
        saveObj(lm4gramsfeature, datasetPath+'/single/lm4/'+file)
        saveObj(lm5gramsfeature, datasetPath+'/single/lm5/'+file)
        saveObj(lm7gramsfeature, datasetPath+'/single/lm7/'+file)
        saveObj(poslm2gramsfeature, datasetPath+'/single/png2/'+file)
        saveObj(poslm5gramsfeature, datasetPath+'/single/png5/'+file)
        saveObj(stopwordsfeature, datasetPath+'/single/swf/'+file)
        saveObj(typetokenfeature, datasetPath+'/arrays/tyto/'+file)
        #saveObj(pcfgavgfeature, datasetPath+'/single/pcfg/' + file)
        #saveObj(minpcfgscorefeature, datasetPath+'/single/mpcfg/' + file)
        saveObj(kenlm4feature, datasetPath+'/single/klm4/' + file)
        saveObj(kenlm5feature, datasetPath+'/single/klm5/' + file)
        i += 1

if __name__ == "__main__": main()

