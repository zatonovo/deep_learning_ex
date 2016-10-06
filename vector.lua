--[[
Author: Brian Lee Yung Rowe

These functions are helpers to make it easier to use the fun package for
people coming from R.
--]]

require "fun" ()
csv = require "simplecsv"

--[[
Take an iterator representing an array/vector and return a vector

Example
tovec(map(function(x) return x+1 end, {1,2,3}))
--]]
function tovec(x)
  local y = { }
  local i = 1
  each(function(...) y[i] = ...; i = i+1 end, x)
  return y
end


--[[
Take an iterator representing a matrix and return a multi-dimensional
array.
--]]
function tomat(x)
  local y = { }
  local i = 1
  each(function(...) y[i] = {...}; i = i+1 end, x)
  return y
end



--[[
Flatten deep lists into single list

Source:
http://svn.wildfiregames.com/public/ps/trunk/build/premake/premake4/src/base/table.lua
--]]
function flatten(arr)
  local result = { }
  local function flatten(arr)
    for _, v in ipairs(arr) do
      if type(v) == "table" then
        flatten(v)
      else
        table.insert(result, v)
      end
    end
  end
  flatten(arr)
  return result
end


--[[
Compute the cartesian product of two vectors

Example
cartprod(range(4), range(6))
--]]
function cartprod(a,b)
  local o = map(function(i) 
    return tomat(map(function(j) return i,j end, a)) end, b)
  local t = { }
  local i = 1
  each(function(x) each(function(y) t[i] = y; i = i+1 end, x) end, o)
  return t
end


--[[
Read a CSV and construct a Torch input dataset.
--]]
function loadTrainSet(path)
  local o = csv:read(path)
  local ncol = #o[1]

  local dataset = {}  
  local i = 1
  -- Generate each row of data
  fn = function(row)
    --print(string.format("x=%s, y=%s", x,y))
    -- TODO: Select first n-1 elements
    local input = torch.Tensor({x,y})
    local output = torch.Tensor({row[ncol]})
    dataset[i] = {input, output}
    i = i + 1
  end
  each(function(row) fn(row) end, o)
  function dataset:size() return (i - 1) end
  return dataset
end


--[[
Convert a Torch training set to a Tensor for batch evaluation of the model.

The training set looks like this:
{
  1: {inputTensor_1, outputTensor_1},
  2: {inputTensor_2, outputTensor_2},
  ...
  n: {inputTensor_n, outputTensor_n}
}

The batch tensor looks like:
Tensor({inputTensor_1, inputTensor_2, ..., inputTensor_n})


The function output is batchTensor, output,
where output is a standard vector.
--]]
function tobatch(data)
  local batch = tovec(map(function(x) return torch.totable(x[1]) end, data))
  local output = flatten(tovec(map(function(x) return torch.totable(x[2]) end, data)))
  return batch, output
end
