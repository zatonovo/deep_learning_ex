--[[
Author: Brian Lee Yung Rowe

Provide some convenient wrappers around model training and evaluation.
--]]
dofile "vector.lua"

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
