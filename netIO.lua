local P = {}

require 'rnn'
require 'cunn'
require 'readFeatures'

local featIO = require('readFeatures')

local function netPred(logits, labels)
   local maxLogit, id = torch.max(logits,2)
   id = id - 1
   return labels[id[{1,1}]], id
end

local function genPOSTags(sens, rnn, id2word, vocabSize, allLabels, fileName)
   local file = io.open(fileName,"w")
   for i = 1,#sens do
      print(i)
      local sentence = sens[i]
      for j = 1,#sentence do 
	 local input = featIO.getOneHotFeature(sentence[j].wordId,vocabSize)
	 input = input:cuda()
	 local output = rnn:forward(input)
	 local predLabel, predLabelId = netPred(output, allLabels)
	 file:write(id2word[sentence[j].wordId] .. '_' .. predLabel .. ' ')
      end
      rnn:forget()
      file:write('\n')
   end
   file:close()
end

P.netPred = netPred
P.genPOSTags = genPOSTags

return P
