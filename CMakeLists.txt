cmake_minimum_required(VERSION 2.4)

# OpenCV Config
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

FIND_LIBRARY(TINYSPLINE NAMES libtinysplinecpp.so PATHS ./tinyspline/build/lib)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

ADD_EXECUTABLE(
  stipplecurves
  FastNoise.cpp
  stipplecurves.cpp
)

TARGET_LINK_LIBRARIES(
  stipplecurves
  ${TINYSPLINE}
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)
