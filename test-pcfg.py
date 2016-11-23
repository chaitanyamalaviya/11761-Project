import os
from nltk.parse import stanford
os.environ['STANFORD_PARSER'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'
os.environ['STANFORD_MODELS'] = '/root/src/ls11761/ls-project/stanford/stanford-parser-full/jars'

parser = stanford.StanfordParser(model_path="/root/src/ls11761/ls-project/stanford/englishPCFG.ser.gz")
sents = ["Hello, My name is Melroy.", "What is your name?"]
sentences = parser.raw_parse_sents_PCFG(sents)
print(sentences)



# GUI
# for line in sentences:
#     for sentence in line:
#         sentence.draw()