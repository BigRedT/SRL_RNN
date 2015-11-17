require 'rnn'
require 'cunn'

local featIO = require('readFeatures')


-- file paths
local featureFile = "./pos_tag_data/feat.txt"
local vocabFile = "./pos_tag_data/vocab.txt"
local labelsFile = "./pos_tag_data/labels.txt"


-- read training data
local train_sens = featIO.readFeatureFile(featureFile)
local id2word, word2id, vocabSize = featIO.readVocab(vocabFile) 
local allLabels, numLabels = featIO.readLabels(labelsFile)


-- print stats
print('Number of training sequences: ' .. #train_sens)
print('Vocabulary size: ' ..vocabSize)
print('Number of labels: ' ..numLabels)


-- extract a example feature to determine inputSize
local exFeat = featIO.getOneHotFeature(1,vocabSize)
print('Feature size: ' ..tostring(exFeat:size(2)))


-- Network parameters
local numClasses = numLabels
local inputSize = exFeat:size(2)
local hiddenSize = 100
local rho = 1  -- maximum number of steps to BPTT

-- Define network input layer
local inputLayer = nn.TemporalConvolution(inputSize,hiddenSize,1,1)

-- Define network feedback
local feedbackLayer = nn.Linear(hiddenSize,hiddenSize)

-- Define non-linear transfer layer
local transferLayer = nn.Sigmoid()

-- Define recurrent network
local rnn = nn.Sequential()
rnn:add(nn.Recurrent(hiddenSize, inputLayer, feedbackLayer, transferLayer, rho))
rnn:add(nn.Linear(hiddenSize,numClasses))
rnn:add(nn.LogSoftMax())
rnn = rnn:cuda()

-- Define criterion
local criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()


-- Iterate over the training sentences
for i = 1,#train_sens do
   print(i)
   local sentence = train_sens[i]
   for j = 1,#sentence do 
      local input = featIO.getOneHotFeature(sentence[j].wordId,vocabSize)
      input = input:cuda()
      local target = torch.Tensor{tonumber(sentence[j].label)+1}
      target = target:cuda()
      local output = rnn:forward(input)
      local gradOutput = criterion:backward(output,target)
      rnn:backward(input,gradOutput)
      rnn:backwardThroughTime()
      rnn:updateParameters(0.1)
      rnn:zeroGradParameters()
      rnn:forget()
   end
end
