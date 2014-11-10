
require 'torch'
require 'sys'
require 'paths'
require 'dok'


-- load C lib
require 'cutorch'
require 'libmetrics'


local function showFlags (t)
  keys = {}
  for k, _ in pairs(t) do
    table.insert(keys, k)
  end
  
  return string.format("(%s)", table.concat(keys, ", "))
end



local metrics = {}

local commonArgs = { 
      {arg='ref', type='torch.FloatTensor | torch.DoubleTensor | torch.CudaTensor',
       help='reference points (m x h) 2d tensor', req=true},
       
      {arg='query', type='torch.FloatTensor | torch.DoubleTensor | torch.CudaTensor',
       help='query point(s) (n x h) 2d tensor or (h) 1d tensor', req=true}
    }

       


local function checkDimensions(query, ref)
  
  assert(torch.type(query) == torch.type(ref), "query and ref must be of the same tensor type (Float, Double or Cuda)")
  
   if(query:dim() == 1) then
     query = query:resize(1, query:size(1))
   end   
     
   assert(query:dim() == 2 and ref:dim() == 2, "query must be 1d or 2d tensor (h or n x h), ref must be a 2d (h x m) tensor")
   assert(query:size(2) == ref:size(2), "query and ref must have equal size features")
  
end
    

function metrics.distancesL2(...)

   local _, ref, query = dok.unpack(
      {...},
      'metrics.distancesL2',
      [[Compute l2 distances matrix]],
      unpack(commonArgs)
   )
   
   checkDimensions(query, ref)
     
   local distances = query.libmetrics.distances(ref:contiguous(), query:contiguous(), libmetrics.metric.l2)
   return distances
end

function metrics.distancesL1(...)

   local _, ref, query = dok.unpack(
      {...},
      'metrics.distancesL1',
      [[Compute l1 distances matrix]],
      unpack(commonArgs)
   )
   
   checkDimensions(query, ref)
     
   local distances = query.libmetrics.distances(ref:contiguous(), query:contiguous(), libmetrics.metric.l1)
   return distances
end


function metrics.distancesLp(...)

   local _, p, ref, query = dok.unpack(
      {...},
      'metrics.distancesLp',
       [[Compute lp distances matrix]],
      {arg='p', type='number',
       help='parameter for lp distance metric', req=true},
      unpack(commonArgs)
      
   )
   
   checkDimensions(query, ref)
--    assert(p and p > 0, "p must be a positive number")
   
     
   local distances = query.libmetrics.distances(ref:contiguous(), query:contiguous(), libmetrics.metric.lp, p)
   return distances
end


return metrics