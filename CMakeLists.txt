cmake_minimum_required(VERSION 3.1)

project(cis565_stream_compaction_test)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBUGINFO -lineinfo)

# Crucial magic for CUDA linking
find_package(Threads REQUIRED)
find_package(CUDA 8.0 REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/osx")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/linux" "/usr/lib64")
elseif(WIN32)
    if(${MSVC_VERSION} MATCHES "1915")
        set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/win/vc2017")
	elseif(${MSVC_VERSION} MATCHES "1900")
        set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/win/vc2015")
    elseif(${MSVC_VERSION} MATCHES "1800")
        set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/win/vc2013")
    elseif(${MSVC_VERSION} MATCHES "1700")
        set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/win/vc2012")
    else()
        MESSAGE("Error: unsupported MSVC_VERSION: " ${MSVC_VERSION})
    endif()
endif()

include_directories(.)
add_subdirectory(stream_compaction)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    "src/main.cpp"
    )

target_link_libraries(${CMAKE_PROJECT_NAME}
    stream_compaction
    ${CORELIBS}
    )
