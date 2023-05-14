cmake_minimum_required(VERSION 3.24)
project(transformer)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp )

add_subdirectory(model)
set(MODEL_DIR ${PROJECT_SOURCE_DIR}/model)
include_directories(${MODEL_DIR}/include)
target_link_libraries(${PROJECT_NAME} model)

add_subdirectory(test)
set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)
