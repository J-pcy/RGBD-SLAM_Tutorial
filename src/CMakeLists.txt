FIND_PACKAGE(PCL REQUIRED COMPONENTS common io visualization filters)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_DEFINITIONS(${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
SET(G2O_ROOT /usr/local/include/g2o)
FIND_PACKAGE(G2O REQUIRED)

FIND_PACKAGE(CSparse REQUIRED)
INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})


ADD_EXECUTABLE(generate_pointcloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES(generate_pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES})


ADD_LIBRARY(slambase slamBase.cpp)
TARGET_LINK_LIBRARIES(slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})


ADD_EXECUTABLE(detectFeatures detectFeatures.cpp)
TARGET_LINK_LIBRARIES(detectFeatures slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})


ADD_EXECUTABLE(joinPointCloud joinPointCloud.cpp)
TARGET_LINK_LIBRARIES(joinPointCloud slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})


ADD_EXECUTABLE(visualOdometry visualOdometry.cpp)
TARGET_LINK_LIBRARIES(visualOdometry slambase ${OpenCV_LIBS} ${PCL_LIBRSRIES})


ADD_EXECUTABLE(slamEnd slamEnd.cpp)
TARGET_LINK_LIBRARIES(slamEnd slambase ${OpenCV_LIBS} ${PCL_LIBRARIES} g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})

