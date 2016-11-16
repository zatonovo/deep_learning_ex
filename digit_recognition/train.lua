--[[
Author: Brian Lee Yung Rowe

Provide some convenient wrappers around model training and evaluation.
--]]
dofile "vector.lua"


--[[
Read a CSV and construct a Torch input dataset.
--]]
function loadTrainSet(path)
  local o = csv.read(path)
  local ncol = #o[1]

  local dataset = {}  
  local i = 1
  -- Generate each row of data
  fn = function(row)
    --print(string.format("row: ", row))
    local trow = torch.Tensor(row)
    dataset[i] = { trow[{{1,ncol-1}}], torch.Tensor({trow[ncol]}) }
    i = i + 1
  end
  each(function(row) fn(row) end, o)
  function dataset:size() return (i - 1) end
  return dataset
end


--[[
Evaluate a model against the given dataset and optionally write to the
outfile.
--]]
function evaluateModel(model, data, outfile)
  batch, expected = tobatch(data)
  output = flatten(torch.totable(model:forward(torch.Tensor(batch))))
  if outfile then
    local ncol = #batch[1]
    for i=1,#batch do
      batch[i][ncol+1] = expected[i]
      batch[i][ncol+2] = output[i]
    end
    print(string.format("Writing CSV to %s", outfile))
    csv.write(outfile, batch)
  end
  return output
end
