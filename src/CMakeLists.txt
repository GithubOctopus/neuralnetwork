cmake_minimum_required(VERSION 3.29.2)
project(octoneuralnetwork)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(${PROJECT_NAME}_make_lib TRUE)

add_library(neuron STATIC neuron.cpp)
add_library(neuronlayer STATIC neuronlayer.cpp)
add_library(neuralnetwork STATIC neuralnetwork.cpp)

if(NOT ${PROJECT_NAME}_make_lib)
    add_executable(${PROJECT_NAME} main.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        neuron
        neuronlayer
        neuralnetwork
    )
else()
    add_library(${PROJECT_NAME}
        #neuronlayer.cpp
        #neuron.cpp
        neuralnetwork.cpp
    )
endif()
