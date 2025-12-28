# 3D Point Cloud Alignment (ICP)
A C++ and CUDA implementation/comparison of the Iterative Closest Point (ICP) algorithm using Open3D for visualization and Eigen for SVD-based rigid body transformations. 
This project demonstrates the step-by-step alignment of two point clouds (source and target).
The goal is to show the advantage of utilizing parallel computing by ICP algorithm

### Project Structure (for now)

generate_data: Takes a base .ply file and creates a transformed "Target" version to act as the ground truth.

icp_engine: The core mathematical engine. It calculates the alignment and saves intermediate snapshots (frames). Uses cpu. (CUDA to be implemented)

icp_vis: A replay tool that reads saved frames and plays them back like a GIF/animation.

### Prerequisites
Windows 10/11 (with latest AMD/NVIDIA/Intel graphics drivers for OpenGL support).

Visual Studio 2022 (with C++ Desktop Development workload).

CMake 3.10+.

Open3D libraries installed (e.g., at D:/Open3D/open3d-devel-windows-amd64-0.19.0/CMake).

### How to build?
Open the x64 Native Tools Command Prompt for VS 2022.

Navigate to the project root and use following commands:

mkdir build
cd build
cmake ..
cmake --build . --config Release

### How to run?
Follow these steps in order from the project root directory:

##### 1. Generate Target Data
Create a misaligned "Target" file from your original source file:

.\build\Release\generate_data.exe bun000.ply
Output: Creates bun000_target.ply.

##### 2. Run ICP Engine
Compute the transformation. This creates a frames/ folder containing the alignment steps:

CUDA version will be implemented.

.\build\Release\icp_engine.exe bun000.ply bun000_target.ply
Output: Fills frames/ with iter_0.ply, iter_5.ply, etc.

##### 3. Visualize the Alignment
Play back the results in an interactive Open3D window:

.\build\Release\icp_vis.exe bun000_target.ply
