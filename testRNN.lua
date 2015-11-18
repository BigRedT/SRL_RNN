require 'rnn'
require 'cunn'

local netIO = require('netIO')
local featIO = require('readFeatures')

-- file paths
local featureFile = "./pos_tag_data/feat.txt"
local vocabFile = "./pos_tag_data/vocab.txt"
local labelsFile = "./pos_tag_data/labels.txt"
local modelPath = "./rnnPOS.net"
local outputFile = "./train_pred_memory_debug2.txt"

-- read data
local sens = featIO.readFeatureFile(featureFile)
local id2word, word2id, vocabSize = featIO.readVocab(vocabFile) 
local allLabels, numLabels = featIO.readLabels(labelsFile)


-- print stats
print('Number of sequences: ' .. #sens)
print('Vocabulary size: ' ..vocabSize)
print('Number of labels: ' ..numLabels)

rnn = torch.load(modelPath)
rnn = rnn:cuda()

netIO.genPOSTags(sens, rnn, id2word, vocabSize, allLabels, outputFile)
