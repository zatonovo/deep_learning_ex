--[[
Author: Brian Lee Yung Rowe

An example of approximating a function with a neural network
--]]
require 'torch'
require 'nn'
dofile "train.lua"

--[[
The function we want to approximate
--]]
function z(x,y) return 2*x^2 - 3*y^2 + 1 end

--[[
The only requirement for the dataset is that it needs to be a Lua table 
with method size() returning the number of elements in the table.
Each element will be a subtable with two elements: the input (a Tensor of 
size 1 x input_size) and the target class (a Tensor of size 1 x 1).

http://mdtux89.github.io/2015/12/11/torch-tutorial.html

Example
makeTestSet(range(-4,4))
--]]
function makeTrainSet(a)
  local dataset = {}  
  local i = 1
  -- Generate each row of data
  fn = function(x,y)
    --print(string.format("x=%s, y=%s", x,y))
    local zi = z(x,y)
    local input = torch.Tensor({x,y})
    local output = torch.Tensor({zi})
    dataset[i] = {input, output}
    i = i + 1
  end
  each(function(t) fn(t[1], t[2]) end, cartprod(a,a))
  function dataset:size() return (i - 1) end
  return dataset
end



print("Make train set")
trainset = loadTrainSet("train_w_unif.csv")

print("Build network")
-- Design your network architecture
model = nn.Sequential()
model:add(nn.Linear(2, 20))
model:add(nn.Tanh())
model:add(nn.Linear(20, 1))

print("Train network")
-- Choose which cost function you want to use
criterion = nn.SmoothL1Criterion()
-- Choose your learning algorithm and set parameters
trainer = nn.StochasticGradient(model, criterion)
trainer:train(trainset)

print("Evaluate in-sample results")
result = evaluateModel(model, trainset, "result_w_unif.csv")


