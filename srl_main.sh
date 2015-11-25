trainIntFile="./srl_data/train_int.txt"
devIntFile="./srl_data/dev_int.txt"
testIntFile="./srl_data/test_int.txt"

trainFile="./srl_data/train.txt"
devFile="./srl_data/dev.txt"
testFile="./srl_data/test.txt"

word_int_File="./srl_data/word_int_map.txt"
label_int_File="./srl_data/label_int_map.txt"
wordint_embeddings="./srl_data/wordint_embeddings.txt"

modelPath="./trained_rnn_models/"$1
bestModelPath="./trained_rnn_models/"$2

predDevFile="./srl_output/"$3
predTestFile="./srl_output/"$4

th trainRNN.lua $word_int_File $label_int_File $wordint_embeddings $trainIntFile $devIntFile $testIntFile $trainFile $devFile $testFile $modelPath $bestModelPath $predDevFile $predTestFile
