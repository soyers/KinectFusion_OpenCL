# Compile with cmake .. -DOpenCV_DIR=<Path to OpenCVConfig.cmake>
# CMake version 2.8.12 introduced add_compile_options
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)


#############################################################################
##################### Set up project and cmake options ######################
#############################################################################
PROJECT(KinectFusion_OpenCL)
SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#To use a custom opencv version it is also possible to set the opencv path here
#SET(CMAKE_PREFIX_PATH /path/to/opencv)


#############################################################################
############################# Include libraries #############################
#############################################################################
#OpenNI
set(OPENNI2_DEFINITIONS ${PC_OPENNI_CFLAGS_OTHER})
FIND_LIBRARY( OPENNI2_LIBRARY
             NAMES OpenNI2
             HINTS ${PC_OPENNI2_LIBDIR} ${PC_OPENNI2_LIBRARY_DIRS} ${PROJECT_SOURCE_DIR}/libs/OpenNI-Linux-x64-2.2/Redist
             PATHS "$ENV{PROGRAMFILES}/OpenNI2/Lib${OPENNI2_SUFFIX}" "$ENV{PROGRAMW6432}/OpenNI2/Lib${OPENNI2_SUFFIX}" "$ENV{PROGRAMW6432}/OpenNI2"
             PATH_SUFFIXES lib lib64
)
FIND_PATH( OPENNI2_INCLUDE_DIR OpenNI.h
          HINTS ${PC_OPENNI2_INCLUDEDIR} ${PC_OPENNI2_INCLUDE_DIRS} 
                  ${PROJECT_SOURCE_DIR}/libs/OpenNI-Linux-x64-2.2/Include /usr/include/ni2
                  PATHS "$ENV{PROGRAMFILES}/OpenNI2/include" "$ENV{PROGRAMW6432}/OpenNI2/include"
          PATH_SUFFIXES openni2 ni2)

link_directories(${OPENNI2_LIBRARY})
INCLUDE_DIRECTORIES(${OPENNI2_INCLUDE_DIR})

#Eigen (Since Eigen is header-only, it is enough to just include the directory)
SET(EIGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/libs/)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

#Boost
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

#OpenCV (Set custom opencv dir with -DOpenCV_DIR=/path/where/OpenCVConfig.cmake/lies)
FIND_PACKAGE(OpenCV 3.1.0 REQUIRED)

#OpenCL
FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})


#############################################################################
########################### set compiler options ############################
#############################################################################
#Set c++11 standard and optimization options- Disable vectorization to prevent alignment issues
add_compile_options(-O3 -std=gnu++0x -funroll-loops -DEIGEN_DONT_VECTORIZE)

#Pass project directory to C++ program
#See http://stackoverflow.com/questions/32947974/file-path-with-cmake-add-definitions
ADD_DEFINITIONS(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")

#Source files to compile
FILE(GLOB includes "src/*.hpp")
FILE(GLOB sources "src/*.cpp")

#Compile
ADD_EXECUTABLE(${PROJECT_NAME} ${includes} ${sources})
#Link
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${OPENNI2_LIBRARY})
