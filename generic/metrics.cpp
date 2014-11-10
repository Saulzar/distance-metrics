#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/metrics.cpp"
#else


#include <luaT.h>
#include <TH.h>
#include <TH/generic/THTensorMath.h>


inline THTensor *libmetrics_(checkTensor)(lua_State* L, int arg) {
  return (THTensor*)luaT_checkudata(L, arg, torch_Tensor);  
}

template<typename M>
static int libmetrics_(distances) (THTensor * ref, THTensor * query, THTensor * distances, M const &metric) {

  int n = ref->size[0];
  int m = query->size[0];
  
  int dim = ref->size[1];
  
  for(int i = 0; i < n; ++i) {
    THTensor * row = THTensor_(newSelect)(ref, 0, i);    
    
    for(int j = 0; j < m; ++j) {
      THTensor * col = THTensor_(newSelect)(query, 0, j);
      
      real sum = 0.0;
      
      TH_TENSOR_APPLY2(real, row, real, col, sum += metric(*row_data, *col_data); );
    
      THTensor_(set2d)(distances, i, j, sum);
      THTensor_(free)(col);  
    }
    
    THTensor_(free)(row);
  }

}



struct libmetrics_(Lp) {

  const real p;
  
   libmetrics_(Lp) (const real &p) : p(p) { }
  
   real operator()(const real& x, const real& y) const {
    return pow(fabs(x - y), p);
  }
};


struct libmetrics_(L1) {
   
   real operator()(const real& x, const real& y) const {
    return fabs(x - y);
  }
};


struct libmetrics_(Sign) {
   
   real operator()(const real& x, const real& y) const {
    return x - y > 0 ? 1 : -1;
  }
};


struct libmetrics_(L2) {
   
  real operator()(const real& x, const real& y) const {
    real d = x - y;
    return d * d;
  }  
};



static int libmetrics_(distances) (lua_State *L) {
  
  THTensor *distances = NULL;
  
  try {
    
    THTensor * ref = libmetrics_(checkTensor)(L, 1);
    THTensor * query = libmetrics_(checkTensor)(L, 2);
    
    DistanceMetric metric = (DistanceMetric)lua_tonumber(L, 3);
    
    if(ref->nDimension != 2 || query->nDimension != 2) 
      throw std::invalid_argument("distances: expected 2d tensor of reference and query points");
    
    size_t features = ref->size[1];
    if(features != query->size[1])
      throw std::invalid_argument("distances: query and reference points must have the same size");

    distances = THTensor_(newWithSize2d)(ref->size[0], query->size[0]);

    switch(metric) {
      case L1:  
        libmetrics_(distances)(ref, query, distances, libmetrics_(L1)());
        break;
      case L2:
        libmetrics_(distances)(ref, query, distances, libmetrics_(L2)());
        break;
      case LP: {
        double p = lua_tonumber(L, 4);    
        libmetrics_(distances)(ref, query, distances, libmetrics_(Lp)(p));
        break;
      }
      default:
        throw std::invalid_argument("distances: bad distance metric type");        
    }
    
    THTensor_(retain)(distances);   
    luaT_pushudata(L, distances, torch_Tensor);
    
    return 1;
  
  } catch (std::exception const &e) {
    
    if(distances) {
      THTensor_(free)(distances);
    }
    
    luaL_error(L, e.what());
  }  
  
  
  return 1; 
  
}



//============================================================
// Register functions in LUA
//
static const luaL_reg libmetrics_(Main__) [] =
{
  {"distances",           libmetrics_(distances)},
  {NULL, NULL}  /* sentinel */
};


extern "C" {

  DLL_EXPORT int libmetrics_(Main_init) (lua_State *L) {
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, libmetrics_(Main__), "libmetrics");
    lua_pop(L,1); 
    return 1;
  }

}
#endif
