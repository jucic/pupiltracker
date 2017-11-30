project (pupiltracker)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_DEBUG_POSTFIX "d")

if (NOT MSVC)
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        if (GCC_VERSION VERSION_LESS 4.7)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else ()
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        endif ()
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
endif ()


find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED)
find_package (TBB REQUIRED)

add_library (
    pupiltracker

    src/PupilTracker.cpp
    src/cvx.cpp
    src/utils.cpp
)

include_directories(
    SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include    
)

install(TARGETS pupiltracker
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include
        DESTINATION "")


add_subdirectory (cmd)
