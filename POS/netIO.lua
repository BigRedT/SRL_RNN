local P = {}

require 'rnn'
require 'cunn'
require 'readFeatures'

local featIO = require('readFeatures')

local function netPred(logits, labels)
   local maxLogit, id = torch.max(logits,2)
   id = id - 1
   if labels[id[{1,1}]]=='PRED' then
      return 'NIL', -1
   end
   return labels[id[{1,1}]], id
end

local function genSRLTags(sens, rnn, id2word, vocabSize, allLabels, word_embeddings, fileName)
   local file = io.open(fileName,"w")
   for i = 1,#sens do
      local sentence = sens[i]
      local predIntId = sentence[1]
      for j = 2,#sentence do 
	 if allLabels[tonumber(sentence[j].label)] ~= 'PRED' then
	    local input = featIO.concatenate(word_embeddings[predIntId.wordId], word_embeddings[sentence[j].wordId])
	    input = input:cuda()
	    local output = rnn:forward(input)
	    local predLabel, predLabelId = netPred(output, allLabels)
	    file:write(id2word[sentence[j].wordId] .. '_' .. predLabel .. ' ')
	 else
	    file:write(id2word[sentence[j].wordId] .. '_' .. 'PRED' .. ' ')
	 end
      end
      rnn:forget()
      file:write('\n')
   end
   file:close()
end

local function genSequencerSRLTags(sens, rnn, id2word, vocabSize, allLabels, word_embeddings, fileName)
   local file = io.open(fileName,"w")
   for i = 1,#sens do
      local sentence = sens[i]
      local predIntId = sentence[1]
      local inputs = {}
      for j = 2,#sentence do 
	 local input = featIO.concatenate(word_embeddings[predIntId.wordId], word_embeddings[sentence[j].wordId])
	 input = input:cuda()
	 table.insert(inputs,input)
      end
      local outputs = rnn:forward(inputs)
      for j = 2,#sentence do
	 local predLabel, predLabelId = netPred(outputs[j-1], allLabels)
	 file:write(id2word[sentence[j].wordId] .. '_' .. predLabel .. ' ')
      end
      rnn:forget()
      file:write('\n')
   end
   file:close()
end

local function genPOSTags(sens, rnn, id2word, vocabSize, allLabels, word_embeddings, fileName)
   local file = io.open(fileName,"w")
   for i = 1,#sens do
      local sentence = sens[i]
      --local predIntId = sentence[1]
      for j = 1,#sentence do 
	 --if allLabels[tonumber(sentence[j].label)] ~= 'PRED' then
	 local input = word_embeddings[sentence[j].wordId]
	 input = input:cuda()
	 local output = rnn:forward(input)
	 local predLabel, predLabelId = netPred(output, allLabels)
	 file:write(id2word[sentence[j].wordId] .. '_' .. predLabel .. ' ')
	 --else
	 --file:write(id2word[sentence[j].wordId] .. '_' .. 'PRED' .. ' ')
	 --end
      end
      rnn:forget()
      file:write('\n')
   end
   file:close()
end

local function genSequencerPOSTags(sens, rnn, id2word, vocabSize, allLabels, word_embeddings, fileName)
   local file = io.open(fileName,"w")
   for i = 1,#sens do
      local sentence = sens[i]
      --local predIntId = sentence[1]
      local inputs = {}
      for j = 1,#sentence do 
	 local input = word_embeddings[sentence[j].wordId]
	 input = input:cuda()
	 table.insert(inputs,input)
      end
      local outputs = rnn:forward(inputs)
      for j = 1,#sentence do
	 local predLabel, predLabelId = netPred(outputs[j], allLabels)
	 file:write(id2word[sentence[j].wordId] .. '_' .. predLabel .. ' ')
      end
      rnn:forget()
      file:write('\n')
   end
   file:close()
end

P.netPred = netPred
P.genSRLTags = genSRLTags
P.genSequencerSRLTags = genSequencerSRLTags
P.genPOSTags = genPOSTags
P.genSequencerPOSTags = genSequencerPOSTags

return P
