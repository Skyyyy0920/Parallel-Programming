cmake_minimum_required(VERSION 3.5)
set(GOOGLETEST_VERSION 1.11.0)
project(Test VERSION ${GOOGLETEST_VERSION} LANGUAGES CXX C)

add_executable(Test
        model/nn/TestTensor.cpp
        model/nn/TestVar.cpp model/Function/TestLossFunction.cpp)

add_subdirectory(googletest)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/googletest/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../model/include
)
link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/../model/src
        ${CMAKE_CURRENT_SOURCE_DIR}/googletest
)

target_link_libraries(${PROJECT_NAME} gtest gtest_main)
target_link_libraries(${PROJECT_NAME} -pthread)
target_link_libraries(${PROJECT_NAME} model)

find_package(Threads REQUIRED)
if (THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_OPTIONS "-pthread")
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_COMPILE_OPTIONS "-pthread")
endif()
if (CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif()