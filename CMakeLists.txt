cmake_minimum_required(VERSION 2.4)

# OpenCV Config
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

ADD_EXECUTABLE(
  messycurves
  PerlinNoise.cpp
  FastNoise.cpp
  messycurves.cpp
)

ADD_EXECUTABLE(
  messycurves_random
  PerlinNoise.cpp
  FastNoise.cpp
  messycurves_random.cpp
)

TARGET_LINK_LIBRARIES(
  messycurves
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)

TARGET_LINK_LIBRARIES(
  messycurves_random
  ${OPENCV_LIBRARIES}
  ${OpenCV_LIBS}
)
