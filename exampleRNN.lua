require 'rnn'

-- function readData()
--    return tables
-- end

-- Network parameters
numClasses = 10
inputSize = 60
hiddenSize = 100
rho = 1  -- maximum number of steps to BPTT

-- Define network input layer
inputLayer = nn.TemporalConvolution(inputSize,hiddenSize,1,1)

-- Define network feedback
feedbackLayer = nn.Linear(hiddenSize,hiddenSize)

-- Define non-linear transfer layer
transferLayer = nn.Sigmoid()

-- Define recurrent network
mlp = nn.Sequential()
mlp:add(nn.Recurrent(hiddenSize, inputLayer, feedbackLayer, transferLayer, rho))
mlp:add(nn.Linear(hiddenSize,numClasses))
mlp:add(nn.LogSoftMax())
--rnn = nn.Sequencer(mlp)

-- Define criterion
criterion = nn.ClassNLLCriterion()

-- input i
input = torch.rand(3,inputSize)
target = torch.Tensor({1, 2, 3});

for iter = 1,100 do
   for i = 1,3 do -- analogous to word
      output = mlp:forward(input[{{i},{}}])
      -- err = criterion:forward(output, target[{i}])
      gradOutput = criterion:backward(output,target[{i}])
      mlp:backward(input,gradOutput)
      mlp:backwardThroughTime()
      mlp:updateParameters(0.1)
      mlp:zeroGradParameters()
      mlp:forget()
   end
end
print(mlp:forward(input):exp())

-- Define loss function/ training criterion
--criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())





