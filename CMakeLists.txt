cmake_minimum_required(VERSION 3.16)
project(LinearMath VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(LinearMath INTERFACE)
target_include_directories(LinearMath INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

option(BUILD_TESTING "Build tests" ON)

if(BUILD_TESTING)
    enable_testing()

    add_executable(test_linear_math testing/test_linear_math.cpp)
    target_link_libraries(test_linear_math PRIVATE LinearMath)
    target_include_directories(test_linear_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/testing)
    if (MSVC)
        target_compile_options(test_linear_math PRIVATE /EHsc /W4)
    else()
        target_compile_options(test_linear_math PRIVATE -Wall -Wextra)
    endif()

    add_test(NAME LinearMathTest COMMAND test_linear_math)
endif()
