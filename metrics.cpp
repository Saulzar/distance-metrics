#include <TH.h>
#include <luaT.h>
#include <THC/THC.h>

#include <stdexcept>
#include <map>
#include <string>

#include "metrics.h"


inline THCState* getCutorchState(lua_State* L)
{
    lua_getglobal(L, "cutorch");
    lua_getfield(L, -1, "getState");
    lua_call(L, 0, 1);
    THCState *state = (THCState*) lua_touserdata(L, -1);
    lua_pop(L, 2);
    return state;
}


inline void luaAssert (bool condition, const char *message) {
 
  if(!condition)
    throw std::invalid_argument(message);  
}



enum DistanceMetric {
  L1 = 0,
  L2,
  LP
};


static int distances(lua_State* L) {
  THCudaTensor *distances = NULL;
  THCState *state = getCutorchState(L);
 
  try {
    
    
    THCudaTensor *ref =   (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
    THCudaTensor *query =   (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");  
    DistanceMetric metric = (DistanceMetric)lua_tonumber(L, 3);
    
    if(ref->nDimension != 2 || query->nDimension != 2) 
      throw std::invalid_argument("distances: expected 2d tensor of reference and query points");
    
    size_t features = ref->size[1];
    if(features != query->size[1])
      throw std::invalid_argument("distances: query and reference points must have the same size");
    
    distances = THCudaTensor_newWithSize2d(state, ref->size[0], query->size[0]);

    switch(metric) {
      case L1:  
        distanceL1(THCudaTensor_data(state, ref), ref->size[0], THCudaTensor_data(state, query), query->size[0], features, THCudaTensor_data(state, distances));
        break;
      case L2:
        distanceL2(THCudaTensor_data(state, ref), ref->size[0], THCudaTensor_data(state, query), query->size[0], features, THCudaTensor_data(state, distances));
        break;
      case LP: {
        float p = lua_tonumber(L, 4);    
        distanceLP(THCudaTensor_data(state, ref), ref->size[0], THCudaTensor_data(state, query), query->size[0], features, THCudaTensor_data(state, distances), p);
        break;
      }
      default:
        throw std::invalid_argument("distances: bad distance metric type");        
    }
    
    THCudaTensor_retain(state, distances);   
    luaT_pushudata(L, distances, "torch.CudaTensor");
    
    return 1;
  
  } catch (std::exception const &e) {
    
    if(distances) {
      THCudaTensor_free(state, distances);
    }
    
    
    luaL_error(L, e.what());
  }  
  
  
  return 1;
}

inline void pushValue(lua_State* L, std::string const &s) {
  lua_pushstring(L, s.c_str());
}

inline void pushValue(lua_State* L, int v) {
  lua_pushnumber(L, v);
}

template<typename K, typename V>
inline void pushValue(lua_State* L, std::map<K, V> const &t) {
  lua_newtable(L);
  int top = lua_gettop(L);

  for (typename std::map<K, V>::const_iterator it = t.begin(); it != t.end(); ++it) {

    pushValue(L, it->first);
    pushValue(L, it->second);
    lua_settable(L, top);
  } 
  
}

template<typename K, typename V>
void setField(lua_State* L, std::string const &table, K const &key,  V const &value) {
  lua_getglobal(L, table.c_str());
  pushValue(L, key);
  pushValue(L, value);
  lua_settable(L, -3);
  lua_pop(L, 1);  
}





std::map<std::string, int> metrics() {
  std::map<std::string, int> m;
  
  m["l1"] = L1; 
  m["l2"] = L2;
  m["lp"] = LP;
  
  return m;
};



#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor        TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define libmetrics_(NAME) TH_CONCAT_3(libmetrics_, Real, NAME)

#include "generic/metrics.cpp"
#include "THGenerateFloatTypes.h"






//============================================================
// Register functions in LUA
//
static const luaL_reg libmetrics_init [] =
{  
  {"distances",   distances},
  {NULL,NULL}
};


extern "C" {

  DLL_EXPORT int luaopen_libmetrics(lua_State *L)
  {


    libmetrics_FloatMain_init(L);
    libmetrics_DoubleMain_init(L);

    luaL_register(L, "libmetrics", libmetrics_init);    
    setField(L, "libmetrics", "metric", metrics());      
    
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, libmetrics_init, "libmetrics");
    lua_pop(L,1); 
    
//     libmetrics_ByteMain_init(L);
//     libmetrics_CharMain_init(L);
//     libmetrics_IntMain_init(L); 
//     libmetrics_LongMain_init(L);
 

    return 1;
  }

}