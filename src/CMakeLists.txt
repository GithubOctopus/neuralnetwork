cmake_minimum_required(VERSION 3.29.2)
project(numbernn)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_subdirectory(fuzzymatch)
add_executable(${PROJECT_NAME} main.cpp)

#target_link_libraries(${PROJECT_NAME} PRIVATE
#
#)
