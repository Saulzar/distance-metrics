CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 4.0 REQUIRED)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

# SET(src knn.cpp)s
SET(src-cuda metrics.cu)
SET(src-cpp metrics.cpp)
SET(luasrc init.lua)

CUDA_ADD_LIBRARY(metrics MODULE ${src-cuda} ${src-cpp})

CUDA_ADD_CUBLAS_TO_TARGET(metrics)

# ADD_TORCH_PACKAGE(knn "${src}" "${luasrc}" "Machine Learning")
TARGET_LINK_LIBRARIES(metrics luaT TH THC)


SET_TARGET_PROPERTIES(metrics PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")
    
INSTALL(TARGETS metrics
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}

  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/metrics")