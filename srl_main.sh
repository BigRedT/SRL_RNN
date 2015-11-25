data_dir=$1
output_dir=$2
layer=$3
hiddenFrac=$4

trainIntFile=$data_dir"/train_int.txt"
devIntFile=$data_dir"/dev_int.txt"
testIntFile=$data_dir"/test_int.txt"

trainFile=$data_dir"/train.txt"
devFile=$data_dir"/dev.txt"
testFile=$data_dir"/test.txt"

word_int_File=$data_dir"/word_int_map.txt"
label_int_File=$data_dir"/label_int_map.txt"
wordint_embeddings=$data_dir"/wordint_embeddings.txt"

modelsDir=$output_dir"/models"

predTrainFilesDir=$output_dir"/predTrainFiles"
predDevFilesDir=$output_dir"/predDevFiles"
predTestFilesDir=$output_dir"/predTestFiles"

if [ ! -d $modelsDir ]; then
    mkdir $modelsDir
fi

if [ ! -d $predTrainFilesDir ]; then
    mkdir $predTrainFilesDir
fi

if [ ! -d $predDevFilesDir ]; then
    mkdir $predDevFilesDir
fi

if [ ! -d $predTestFilesDir ]; then
    mkdir $predTestFilesDir
fi

th trainRNN.lua $word_int_File $label_int_File $wordint_embeddings $trainIntFile $devIntFile $testIntFile $trainFile $devFile $testFile $modelsDir $predTrainFilesDir $predDevFilesDir $predTestFilesDir $layer $hiddenFrac
