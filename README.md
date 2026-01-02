# 3D Point Cloud Alignment (ICP)

A C++ and CUDA implementation of the Iterative Closest Point (ICP) algorithm. This project demonstrates point cloud alignment and compares CPU vs GPU performance using parallel computing.

## Features
- **CPU Implementation**: Uses Eigen for SVD-based rigid transformations
- **CUDA Implementation**: GPU-accelerated version with parallel nearest neighbor search, centroid computation, and transformation (Tested on remote GPU server in Linux only)
- **Visualization**: Open3D-based replay tool (Tested on Windows only)
- **Performance Comparison**: Demonstrates 300x speedup with GPU acceleration on large point clouds

## Project Structure

- **generate_data**: Creates a transformed "target" point cloud from a source .ply file
- **icp_engine**: CPU-based ICP implementation with frame-by-frame snapshots
- **icp_cuda**: GPU-accelerated ICP using CUDA
- **icp_vis**: Interactive visualization tool for replaying alignment steps

## Prerequisites

### For Windows 
- Windows 10/11
- Visual Studio 2022 (C++ Desktop Development workload)
- CMake 3.18+
- Open3D libraries (e.g., `D:/Open3D/open3d-devel-windows-amd64-0.19.0/CMake`)

### For Linux 
- CUDA Toolkit 11.0+ with compatible NVIDIA GPU
- CMake 3.18+
- Eigen3 library 

## Installation

### Eigen3 and OPEN3D Setup (for Linux)

```bash
sudo apt-get install libeigen3-dev
sudo apt-get install libopen3d-dev
```
If the sudo command does not work, download pre-built release (https://github.com/isl-org/Open3D/releases)

```bash
cd ~
wget https://github.com/isl-org/Open3D/releases/download/v0.18.0/open3d-devel-linux-x86_64-cxx11-abi-0.18.0.tar.xz
tar -xf open3d-devel-linux-x86_64-cxx11-abi-0.18.0.tar.xz
```
Update CMakeLists.txt to point to it, e.g:
set(Open3D_DIR "$ENV{HOME}/open3d-devel-linux-x86_64-cxx11-abi-0.18.0/lib/cmake/Open3D")

### OPEN3D Setup (for Windows)
Follow the guide here 
https://www.open3d.org/docs/release/getting_started.html#c

(Note: Eigen is also installed with Open3D, no need for Eigen installation if you are on Windows)

Update CMakeLists.txt to point to it, e.g:
set(Open3D_DIR "D:/Open3D/open3d-devel-windows-amd64-0.19.0/CMake")

## Build Instructions

### Windows 
```cmd
# Open x64 Native Tools Command Prompt for VS 2022
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Linux 
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### 1. Generate Target Data
Create a transformed target point cloud:

**Windows:**
```cmd
.\build\Release\generate_data.exe bun000.ply
```

**Linux:**
```bash
./build/generate_data bun000.ply
```

**Output:** `bun000_target.ply`

### 2. Run ICP Alignment

**CPU Version:** Remove kd_tree argument for brute force nearest neighbor search

**Windows:**

```cmds
.\build\Release\icp_engine.exe bun000.ply bun000_target.ply kd_tree
```

**Linux:**
```bash
./build/icp_engine bun000.ply bun000_target.ply kd_tree
```

**CUDA Version:**

Not tested on Windows.

**Linux:**
```bash
./build/icp_cuda bun000.ply bun000_target.ply
```

```bash
./build/icp_cuda_kdtree bun000.ply bun000_target.ply
```

**Output:** Creates `frames/` directory with intermediate alignment steps (`iter_0.ply`, `iter_5.ply`, etc.)

### 3. Visualize Results 
```cmd
.\build\Release\icp_vis.exe bun000_target.ply
```

Not tested on Linux.

Opens an interactive Open3D window showing the alignment animation.


## Algorithm Details

### ICP Pipeline
1. **Nearest Neighbor Search**: Find closest target point for each source point
2. **Centroid Computation**: Calculate centers of mass for both point sets
3. **SVD Transformation**: Compute optimal rigid transformation (rotation + translation)
4. **Apply Transform**: Update source points
5. **Iterate**: Repeat until convergence (50 iterations)

### CUDA Optimizations
- Parallel nearest neighbor search (each thread processes one source point)
- Shared memory reductions for centroid computation
- GPU-accelerated covariance matrix calculation
- Parallel point transformation

