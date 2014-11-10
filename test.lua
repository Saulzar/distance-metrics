require 'cunn'
local metrics = require 'metrics'



function differenceMatrix(x, y)
  
  local nx = x:size(1)
  local ny = y:size(1)  
  
  local d = x:size(2)
  
  local result = x.new(nx, ny, d)
  
  x = x:clone():resize(nx, 1, d):expand(nx, ny, d)
  y = y:clone():resize(1, ny, d):expand(nx, ny, d)
  
  result:copy(x)
  result:add(-1, y)
  
  return result
end


local function distanceL2(x, y)
  
  local d = differenceMatrix(x, y)
  d:cmul(d)
  
  return d:sum(3):resize(x:size(1), y:size(1))
end

local function distanceL1(x, y)
  
  local d = differenceMatrix(x, y)
  d:abs()
  
  return d:sum(3):resize(x:size(1), y:size(1))
end


local function distanceLp(x, y, p)
  
  local d = differenceMatrix(x, y)
  d:abs()
  d:pow(p)
  
  return d:sum(3):resize(x:size(1), y:size(1))
end


local function assert_eq(name, d1, d2, tolerance)
  
-- print( d2, d1)
  
  d1:add(-1, d2)
  
  local err = d1:abs():max()
  assert(err < tolerance, string.format("%s: failed: error = %f", name, err))
    
end


function test()

  torch.manualSeed(1)
  
  local tolerance = 1e-2
  
  local types = {"torch.CudaTensor", "torch.FloatTensor", "torch.DoubleTensor" }

  for _, tensorType in pairs(types) do
    for i = 1, 50 do
    
      local dim = torch.random(200)
      local n = torch.random(200)
      local m = torch.random(200)
      
      local p = torch.uniform(0.5, 2)
      
      local x = torch.FloatTensor(n, dim):uniform():type(tensorType)
      local y = torch.FloatTensor(m, dim):uniform():type(tensorType)
      
      assert_eq("distanceL2 "..tensorType, distanceL2(x, x), metrics.distancesL2(x, x), tolerance)
      assert_eq("distanceL1 "..tensorType, distanceL1(x, y), metrics.distancesL1(x, y), tolerance)
      assert_eq("distanceLP "..tensorType, distanceLp(x, y, p), metrics.distancesLp(p, x, y), tolerance)

    end
    
    print ("Passed 50 tests.", tensorType)
  end
  

  
end
  

test()