cmake_minimum_required(VERSION 3.18)
project(GpuIcpProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found - building GPU version")
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found - building CPU version only")
endif()

# Try to find Open3D (for local Windows system)
set(Open3D_DIR "D:/Open3D/open3d-devel-windows-amd64-0.19.0/CMake")
find_package(Open3D QUIET)

# Try to find Eigen (used this instead open3d in remote gpu server)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/eigen-3.4.0")
    set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/eigen-3.4.0)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Using local Eigen")
endif()

# ===== CPU Executables (with Open3D if available) =====
if(Open3D_FOUND)
    add_executable(generate_data src/generate_data.cpp)
    target_link_libraries(generate_data PRIVATE Open3D::Open3D)

    add_executable(icp_engine src/icp_engine.cpp)
    target_link_libraries(icp_engine PRIVATE Open3D::Open3D)

    add_executable(icp_vis src/visualizer.cpp)
    target_link_libraries(icp_vis PRIVATE Open3D::Open3D)

    # Copy DLLs on Windows
    if(WIN32)
        add_custom_command(TARGET icp_vis POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "D:/Open3D/open3d-devel-windows-amd64-0.19.0/bin/Open3D.dll"
            "D:/Open3D/open3d-devel-windows-amd64-0.19.0/bin/tbb12.dll"
            $<TARGET_FILE_DIR:icp_vis>)
    endif()
    message(STATUS "Built Open3D executables")
endif()

# ===== CUDA Executable (if CUDA available) =====
if(CUDA_AVAILABLE AND (Eigen3_FOUND OR EIGEN3_INCLUDE_DIR))
    add_executable(icp_cuda icp_cuda.cu)
    
    if(Eigen3_FOUND)
        target_link_libraries(icp_cuda Eigen3::Eigen)
    else()
        target_include_directories(icp_cuda PRIVATE ${EIGEN3_INCLUDE_DIR})
    endif()
    
    set_target_properties(icp_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
    endif()
    
    target_compile_options(icp_cuda PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>: -O3 --use_fast_math>
    )
    
    message(STATUS "Built CUDA executable")
endif()