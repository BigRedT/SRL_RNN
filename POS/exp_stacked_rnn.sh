# ./srl_main.sh testModel.net bestTestModel.net testPred.txt testTest.txt

data_dir="./posdata"
output_dir="./stacked_rnn_output"

if [ ! -d $data_dir ]; then
    echo "Data directory not found: " $data_dir
fi

if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

layers=(1 2 3)
hiddenFracs=(2)

for layer in ${layers[*]}
do
    for hiddenFrac in ${hiddenFracs[*]}
    do
	echo "Number of layers: " $layer " Hidden layer fraction: " $hiddenFrac

	# Create sub directory
	output_sub_dir=$output_dir"/layers_"$layer"_hiddenFrac_"$hiddenFrac
	if [ ! -d $output_sub_dir ]; then
	    mkdir $output_sub_dir
	fi

	# Train and test RNN
	./pos_main.sh $data_dir $output_sub_dir $layer $hiddenFrac "stacked"

    done
done




