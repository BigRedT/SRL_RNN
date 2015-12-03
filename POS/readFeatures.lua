local P = {}

local function readIntData(fileName, labels)
   local file = io.input(fileName, "r")
   if file == nil then
      print("Could not read file")
   end

   local sens = {}
   local sens_id = 1
   local PRED_int_id = tonumber(labels["PRED"])

   while true do
      local line = io.read("*line")

      if line == nil then break end

      if line == "{" then 
         sentence = {}
         word_id = 1

      elseif line == "}" then
         sens[sens_id] = sentence
         sens_id = sens_id + 1
         -- Put a check for label == label["PRED"]. If yes, put word here as well as word_id = 1 location 

      else
         local line_split = string.split(line," ")
         local line_info = {}
         local wid = tonumber(line_split[1])
         local lid = tonumber(line_split[2])
         line_info["wordId"] = wid
         line_info["label"] = lid

         sentence[word_id] = line_info
         if lid == PRED_int_id then
            sentence[1] = line_info
	 end
	 word_id = word_id + 1	    
      end
   end
   return sens
end

-- To append pred with each word, what we can do is start sentence from position 2. First pass through the sentence and find out the PRED using the label map and append at start of sentence.
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
       -- Here put word_id = 2

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
      local word = line_split[1]
      id2word[word_id] = word
      word2id[word] = word_id
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
   local label2id = {}
   local numLabels = 0
   while true do
      local line = io.read("*line")

      if line == nil then break end
      
      local line_split = string.split(line," ")
      labels[tonumber(line_split[2])] = line_split[1]
      label2id[line_split[1]] = tonumber(line_split[2])
      numLabels = numLabels + 1
   end
   return labels, label2id, numLabels
end

local function readEmbeddings(fileName)
   local file = io.input(fileName, "r")
   if file == nil then
      print("Could not read file")
   end

   local embedding_size = 300
   local word_embeddings = {}

   while true do
      local line = io.read("*line")

      if line == nil then break end
      
      local line_split = string.split(line," ")
      
      word_int = tonumber(line_split[1])
      embedding = torch.zeros(1,embedding_size)
      for i = 2, 301 do
	 embedding[{1,i-1}] = tonumber(line_split[i]) 
      end

      word_embeddings[word_int] = embedding
      
   end
   return word_embeddings
end

-- Converts a wordId (0 based indexing) into a one hot feature vector
local function getOneHotFeature(wordId, numWords)
   local feat = torch.zeros(1,numWords)
   local wordId_ = wordId + 1
   feat[{1,wordId_}] = 1
   return feat
end   


local function concatenate(vec1, vec2)
   return torch.cat(vec1,vec2, 2)
end

local function readCompleteData(word_int_file, label_int_file, word_Embeddings_File, data_int_file)
   id2word, word2id, vocabSize = readVocab(word_int_file)
   labels, label2id, numLabels = readLabels(label_int_file)
   word_embeddings = readEmbeddings(word_Embeddings_File)
   sens = readIntData(data_int_file, label2id)
   -- sens = readFeatureFile(data_int_file)

   return sens, id2word, word2id, vocabSize, labels, label2id, numLabels, word_embeddings
end



P.readFeatureFile = readFeatureFile
P.readVocab = readVocab
P.getOneHotFeature = getOneHotFeature
P.readLabels = readLabels
P.readCompleteData = readCompleteData
P.readIntData = readIntData
P.readEmbeddings = readEmbeddings
P.concatenate = concatenate
-- local sens = readFeatureFile("/home/tgupta6/pos_tag_data/feat.txt")
-- local vocab, vocabSize = readVocab("/home/tgupta6/pos_tag_data/vocab.txt")

-- print(sens[1][1].wordId)
-- print(vocabSize)
-- print(getOneHotFeature(sens[1][1].wordId,vocabSize))


return P
