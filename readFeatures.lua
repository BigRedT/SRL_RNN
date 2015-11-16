function readFeatureFile(fileName)
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
	 line_info["feat_id"] = line_split[1]
	 line_info["label"] = line_split[2]
	 sentence[word_id] = line_info
	 word_id = word_id + 1
      end
   end
   return sens
end

function readVocab(fileName)
   local file = io.input(fileName, "r")
   if file == nil then
      print("Could not read file")
   end

   local words = {}
   local word_id = 0
   while true do
      local line = io.read("*line")
      
      if line == nil then break end
      
      local line_split = string.split(line," ")
      words[word_id] = line_split[2]
      word_id = word_id + 1
   end
   return words      
end



sens = readFeatureFile("/home/tgupta6/pos_tag_data/feat.txt")
vocab = readVocab("/home/tgupta6/pos_tag_data/vocab.txt")

print(sens[1])
