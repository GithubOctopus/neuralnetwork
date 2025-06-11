cmake_minimum_required(VERSION 3.29.2)
project(numbernn)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_subdirectory(fuzzymatch)
add_executable(${PROJECT_NAME} main.cpp)

add_library(neuron SHARED neuron.cpp)
add_library(neuronlayer SHARED neuronlayer.cpp)
add_library(neuralnetwork SHARED neuralnetwork.cpp)


target_link_libraries(${PROJECT_NAME} PRIVATE
    neuron
    neuronlayer
    neuralnetwork
)
