cmake_minimum_required(VERSION 3.10)
project(GpuIcpProject)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Locate Open3D (Point this to your D: drive folder)
# Make sure this path contains the Open3DConfig.cmake file
set(Open3D_DIR "D:/Open3D/open3d-devel-windows-amd64-0.19.0/CMake")
find_package(Open3D REQUIRED)

# 2. Executable: Data Generator
add_executable(generate_data src/generate_data.cpp)
target_link_libraries(generate_data PRIVATE Open3D::Open3D)

# 3. Executable: ICP Engine (The Math)
add_executable(icp_engine src/icp_engine.cpp)
target_link_libraries(icp_engine PRIVATE Open3D::Open3D)

# 4. Executable: Visualizer (The Replay)
add_executable(icp_vis src/visualizer.cpp)
target_link_libraries(icp_vis PRIVATE Open3D::Open3D)

# Copy DLLs to build folder automatically (Optional but helpful)
if(WIN32)
    add_custom_command(TARGET icp_vis POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/Open3D/open3d-devel-windows-amd64-0.19.0/bin/Open3D.dll"
        "D:/Open3D/open3d-devel-windows-amd64-0.19.0/bin/tbb12.dll"
        $<TARGET_FILE_DIR:icp_vis>)
endif()