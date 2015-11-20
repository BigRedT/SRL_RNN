require 'rnn'
require 'cunn'

local featIO = require('readFeatures')
local netIO = require('netIO')

-- local py = require 'python'
-- local sum_from_python = py.import "sum".sum_from_python
-- print(sum_from_python(2,3))


-- file paths
-- resume from word embeddings and arguments
local word_int_file = arg[1]
local label_int_file = arg[2]
local wordInt_embeddings_file = arg[3]


local train_int_file = arg[4]
local dev_int_file = arg[5]
local test_int_file = arg[6]

local train_file = arg[7]
local dev_file = arg[8]
local test_file = arg[9]

local modelPath = arg[10]
local outputFile = arg[11]

local predDevFile = arg[12]
local predTestFile = arg[13]

-- read training data
-- local train_sens = featIO.readFeatureFile(featureFile)
-- local id2word, word2id, vocabSize = featIO.readVocab(vocabFile) 
-- local allLabels, numLabels = featIO.readLabels(labelsFile)
local train_sens, id2word, word2id, vocabSize, allLabels, numLabels, word_embeddings  = featIO.CompleteData(word_int_file, label_int_file, wordint_embeddings_file, train_int_file)
local dev_sens = featIO.readIntData(dev_int_file)
local test_sens = featIO.readIntData(test_int_file)


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
local hiddenSize = 1000
local rho = 100  -- maximum number of steps to BPTT
local max_epochs = 10
local learning_rate = 0.01


-- Define recurrent network architecture
local inputLayer = nn.TemporalConvolution(inputSize,hiddenSize,1,1)
local feedbackLayer = nn.Linear(hiddenSize,hiddenSize)
local transferLayer = nn.Sigmoid()
local rnn = nn.Sequential()
rnn:add(nn.Recurrent(hiddenSize, inputLayer, feedbackLayer, transferLayer, rho))
rnn:add(nn.Linear(hiddenSize,numClasses))
rnn:add(nn.LogSoftMax())
rnn = rnn:cuda()


-- Define criterion / loss function
local criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()


-- Iterate over the training sentences
for epoch = 1,max_epochs  do
   print('Epoch: ' .. tostring(epoch))
   print('#Sequence: ')
   
   for i = 1,#train_sens do 
      
      if i % 1000 == 0 then
	     print('    ' .. tostring(i))
      end
      
      local sentence = train_sens[i]
      -- local label_int_index = sentence[1]
      local updateCounter = 0
      for j = 1,#sentence do 
      -- for j = 2,#sentence do    To incorporate that first is PRED
         local input = featIO.getOneHotFeature(sentence[j].wordId,vocabSize)
         -- local input = featIO.getOneHotFeature(sentence[j].wordId,vocabSize) Get concatenation of PRED and word Embedding
      	input = input:cuda()
      	local target = torch.Tensor{tonumber(sentence[j].label)+1}
      	target = target:cuda()
      	local output = rnn:forward(input)
      	local gradOutput = criterion:backward(output,target)
      	rnn:backward(input,gradOutput)
      	updateCounter = updateCounter + 1
      end
      rnn:backwardThroughTime()
      rnn:updateParameters(learning_rate)
      rnn:zeroGradParameters()
      rnn:forget()
   end
   torch.save(modelPath,rnn)
   netIO.genPOSTags(train_sens, rnn, id2word, vocabSize, allLabels, outputFile)
end
