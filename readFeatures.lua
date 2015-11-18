local P = {}
local function readFeatureFile(fileName)
   local file = io.input(fileName, "r")
   if file == nil then
      print("Could not read file")
   end

   local sens = {}
   local labels = {}
   local sens_id = 1
   while true do
      local line = io.read("*line")

      if line == nil then break end

      if line == "{" then 
	 sentence = {}
	 word_id = 1

      elseif line == "}" then
	 sens[sens_id] = sentence
	 sens_id = sens_id + 1

      else
	 local line_split = string.split(line," ")
	 local line_info = {}
	 line_info["wordId"] = tonumber(line_split[1])
	 line_info["label"] = tonumber(line_split[2])
	 sentence[word_id] = line_info
	 word_id = word_id + 1
      end
   end
   return sens
end


local function readVocab(fileName)
   local file = io.input(fileName, "r")
   if file == nil then
      print("Could not read file")
   end

   local id2word = {}
   local word2id = {}
   local wordCount = 0
   while true do
      local line = io.read("*line")
      
      if line == nil then break end
      
      local line_split = string.split(line," ")
      local word_id = tonumber(line_split[2])
      id2word[word_id] = line_split[1]
      word2id[line_split[2]] = word_id
      wordCount = wordCount + 1
   end
   local vocabSize = wordCount
   return id2word, word2id, vocabSize      
end


local function readLabels(fileName)
   local file = io.input(fileName, "r")
   if file == nil then
      print("Could not read file")
   end

   local labels = {}
   local numLabels = 0
   while true do
      local line = io.read("*line")

      if line == nil then break end
      
      local line_split = string.split(line," ")
      labels[tonumber(line_split[2])] = line_split[1]
      numLabels = numLabels + 1
   end
   return labels, numLabels
end

-- Converts a wordId (0 based indexing) into a one hot feature vector
local function getOneHotFeature(wordId, numWords)
   local feat = torch.zeros(1,numWords)
   local wordId_ = wordId + 1
   feat[{1,wordId_}] = 1
   return feat
end   

P.readFeatureFile = readFeatureFile
P.readVocab = readVocab
P.getOneHotFeature = getOneHotFeature
P.readLabels = readLabels
-- local sens = readFeatureFile("/home/tgupta6/pos_tag_data/feat.txt")
-- local vocab, vocabSize = readVocab("/home/tgupta6/pos_tag_data/vocab.txt")

-- print(sens[1][1].wordId)
-- print(vocabSize)
-- print(getOneHotFeature(sens[1][1].wordId,vocabSize))


return P
