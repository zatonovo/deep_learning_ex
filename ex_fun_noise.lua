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



print("Make train set")
trainset = loadTrainSet("train_w_unif.csv")

print("Build network")
-- Design your network architecture
model = nn.Sequential()
model:add(nn.Linear(2, 10))
model:add(nn.Tanh())
model:add(nn.Linear(10, 1))

print("Train network")
-- Choose which cost function you want to use
criterion = nn.SmoothL1Criterion()
-- Choose your learning algorithm and set parameters
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 1e-3
trainer.learningRateDecay = 1e-4
trainer:train(trainset)

print("Evaluate in-sample results")
result = evaluateModel(model, trainset, "result_w_norm_10.csv")


