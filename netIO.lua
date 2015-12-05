local P = {}

require 'rnn'
require 'cunn'
require 'readFeatures'

local featIO = require('readFeatures')

local function netPred(logits, labels)
   local maxLogit, id = torch.max(logits,2)
   id = id - 1
   local label = ''
   local labelID = -1
   l = labels[id[{1,1}]]	
   if l=='PRED' then
      label = 'NIL'	
   elseif l~=nil  then
      label = labels[id[{1,1}]]
      labelID = id
   else
      label = 'DUMLAB'
   end   
   return label, labelID
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
	 if allLabels[tonumber(sentence[j].label)] ~= 'PRED' then
	 	local predLabel, predLabelId = netPred(outputs[j-1], allLabels)
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

P.netPred = netPred
P.genSRLTags = genSRLTags
P.genSequencerSRLTags = genSequencerSRLTags

return P
