import os
import sys

# Paths
trainIntFile = "./srl_data/train_int.txt"
devIntFile = "./srl_data/dev_int.txt"
testIntFile = "./srl_data/test_int.txt"

trainFile = "./srl_data/train.txt"
devFile = "./srl_data/dev.txt"
testFile = "./srl_data/test.txt"

vocabFile = "./srl_data/word_int_map.txt"
labelsFile = "./srl_data/labels_int_map.txt"

modelPath = "./trained_rnn_models/" + sys.argv[1]
bestModelPath = "./trained_rnn_models/" + sys.argv[2]

predDevFile = "./srl_output/" + sys.argv[3]
predTestFile = "./srl_output/" + sys.argv[4]


cmd = "th trainRNN.lua {} {} {} {} {}".format(trainIntFile, devIntFile, devFile,
                                              vocabFile, labelsFile,
                                              modelPath, bestModelPath,
                                              predDevFile)
                                              
                                              
                                              
os.system(cmd)
