cmake_minimum_required(VERSION 3.16)

project(tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mavx2 -m64 -Ofast -flto")

set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(DIAGNOSTIC_MODE)
add_compile_definitions(BENCHMARK_MODE)

add_executable(tests main.cpp
    utils.h
    vxor.h
    vxor.cpp
    peval.h
    mmul.h
    mmul.cpp
    rational.h
)

include(GNUInstallDirs)
install(TARGETS tests
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
