--[[
http://yann.lecun.com/exdb/mnist/
http://rnduja.github.io/2015/10/13/torch-mnist/
http://rnduja.github.io/2015/10/01/deep_learning_with_torch/
http://mdtux89.github.io/2015/12/11/torch-tutorial.html
--]]
require 'torch'
require 'nn'
dofile "train.lua"

function load_data()
  print 'Loading data...'
  tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

  if not paths.dirp('mnist.t7') then
    os.execute('wget ' .. tar)
    os.execute('tar xvf ' .. paths.basename(tar))
  end

  train_file = 'mnist.t7/train_32x32.t7'
  test_file = 'mnist.t7/test_32x32.t7'

  train = torch.load(train_file,'ascii')
  train_size = #train['labels']
  function train:size() return train_size end

  test = torch.load(test_file,'ascii')
  test_size = #test['labels']
  function test:size() return test_size end
  return {train, test}
end


function init_model()
  model = nn:Sequential()
  model:add(nn.SpatialConvolutionMap())
  model:add(nn.ReLU())
  model:add(nn.SpatialLPPooling())
  model:add(nn.SpatialDropout())

  model:add(nn.Reshape())
  model:add(nn.Linear())
  model:add(nn.LogSoftMax())
  return model
end


function run_network(data, model)
  criterion = nn.ClassNLLCriterion()
  trainer = nn.StochasticGradient(model, criterion)
  trainer:train(data[1])

  print("Evaluate test set results")
  result = evaluateModel(model, data[2], "result.csv")
end
