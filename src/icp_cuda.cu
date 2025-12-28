#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <Eigen/Dense>

struct Point3D {
    float x, y, z;
};

// ============================================================================
// CUDA KERNELS
// ============================================================================

// Kernel: Find nearest neighbor for each source point
__global__ void findNearestNeighbors(
    const float3* source, int n_source,
    const float3* target, int n_target,
    float3* matched_target, float* distances)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_source) return;

    float3 src = source[idx];
    float min_dist = 1e10f;
    float3 best_match = target[0];

    // Each thread searches through all target points
    for (int i = 0; i < n_target; ++i) {
        float3 tgt = target[i];
        float dx = src.x - tgt.x;
        float dy = src.y - tgt.y;
        float dz = src.z - tgt.z;
        float dist = dx*dx + dy*dy + dz*dz;
        
        if (dist < min_dist) {
            min_dist = dist;
            best_match = tgt;
        }
    }

    matched_target[idx] = best_match;
    distances[idx] = min_dist;
}

// Kernel: Compute centroid using parallel reduction
__global__ void computeCentroid(const float3* points, int n, float3* partial_sums) {
    __shared__ float3 shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (idx < n) {
        shared_sum[tid] = points[idx];
    } else {
        shared_sum[tid] = make_float3(0.0f, 0.0f, 0.0f);
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid].x += shared_sum[tid + stride].x;
            shared_sum[tid].y += shared_sum[tid + stride].y;
            shared_sum[tid].z += shared_sum[tid + stride].z;
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

// Kernel: Compute covariance matrix H
__global__ void computeCovarianceMatrix(
    const float3* source, const float3* target, int n,
    float3 center_src, float3 center_tgt,
    float* H_elements)  // 9 elements (3x3 matrix)
{
    __shared__ float shared_H[9][256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = 0; i < 9; ++i) {
        shared_H[i][tid] = 0.0f;
    }
    
    if (idx < n) {
        float3 s = make_float3(
            source[idx].x - center_src.x,
            source[idx].y - center_src.y,
            source[idx].z - center_src.z
        );
        float3 t = make_float3(
            target[idx].x - center_tgt.x,
            target[idx].y - center_tgt.y,
            target[idx].z - center_tgt.z
        );
        
        // Compute outer product: s * t^T
        shared_H[0][tid] = s.x * t.x;
        shared_H[1][tid] = s.x * t.y;
        shared_H[2][tid] = s.x * t.z;
        shared_H[3][tid] = s.y * t.x;
        shared_H[4][tid] = s.y * t.y;
        shared_H[5][tid] = s.y * t.z;
        shared_H[6][tid] = s.z * t.x;
        shared_H[7][tid] = s.z * t.y;
        shared_H[8][tid] = s.z * t.z;
    }
    __syncthreads();
    
    // Reduction for each matrix element
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int i = 0; i < 9; ++i) {
                shared_H[i][tid] += shared_H[i][tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        for (int i = 0; i < 9; ++i) {
            atomicAdd(&H_elements[i], shared_H[i][0]);
        }
    }
}

// Kernel: Apply transformation to source points
__global__ void applyTransform(float3* points, int n, const float* transform) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float3 p = points[idx];
    
    // Apply 4x4 transformation matrix
    float x = transform[0]*p.x + transform[1]*p.y + transform[2]*p.z + transform[3];
    float y = transform[4]*p.x + transform[5]*p.y + transform[6]*p.z + transform[7];
    float z = transform[8]*p.x + transform[9]*p.y + transform[10]*p.z + transform[11];
    
    points[idx] = make_float3(x, y, z);
}

// ============================================================================
// CPU HELPER FUNCTIONS
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Cross-platform directory creation
bool createDirectory(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        #ifdef _WIN32
            return _mkdir(path) == 0;
        #else
            return mkdir(path, 0700) == 0;
        #endif
    }
    return true;  // Directory already exists
}

std::vector<Point3D> loadPLY(const std::string& filename) {
    std::vector<Point3D> points;
    std::ifstream file(filename);
    if (!file.is_open()) return points;

    std::string line;
    bool header = true;
    while (std::getline(file, line)) {
        if (header) {
            if (line == "end_header") header = false;
            continue;
        }
        std::stringstream ss(line);
        Point3D p;
        if (ss >> p.x >> p.y >> p.z) points.push_back(p);
    }
    return points;
}

void savePLY(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream file(filename);
    file << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n";
    file << "property float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto& p : points) 
        file << p.x << " " << p.y << " " << p.z << "\n";
}

// Convert Point3D vector to float3 vector
std::vector<float3> toFloat3(const std::vector<Point3D>& points) {
    std::vector<float3> result(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        result[i] = make_float3(points[i].x, points[i].y, points[i].z);
    }
    return result;
}

// Convert float3 vector back to Point3D
std::vector<Point3D> toPoint3D(const std::vector<float3>& points) {
    std::vector<Point3D> result(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        result[i] = {points[i].x, points[i].y, points[i].z};
    }
    return result;
}

// Compute centroid on GPU
float3 computeCentroidGPU(float3* d_points, int n) {
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float3* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(float3)));
    
    computeCentroid<<<numBlocks, blockSize>>>(d_points, n, d_partial_sums);
    
    // Copy partial sums back and finish reduction on CPU
    std::vector<float3> partial_sums(numBlocks);
    CUDA_CHECK(cudaMemcpy(partial_sums.data(), d_partial_sums, 
                          numBlocks * sizeof(float3), cudaMemcpyDeviceToHost));
    
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    for (const auto& ps : partial_sums) {
        centroid.x += ps.x;
        centroid.y += ps.y;
        centroid.z += ps.z;
    }
    centroid.x /= n;
    centroid.y /= n;
    centroid.z /= n;
    
    CUDA_CHECK(cudaFree(d_partial_sums));
    return centroid;
}

// Compute rigid transform using Eigen SVD (on CPU)
Eigen::Matrix4f findRigidTransform(float3* d_source, float3* d_matched, int n,
                                   float3 center_src, float3 center_tgt) {
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Compute covariance matrix on GPU
    float* d_H;
    CUDA_CHECK(cudaMalloc(&d_H, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_H, 0, 9 * sizeof(float)));
    
    computeCovarianceMatrix<<<numBlocks, blockSize>>>(
        d_source, d_matched, n, center_src, center_tgt, d_H);
    
    float H_elements[9];
    CUDA_CHECK(cudaMemcpy(H_elements, d_H, 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_H));
    
    // Build Eigen matrix
    Eigen::Matrix3f H;
    H << H_elements[0], H_elements[1], H_elements[2],
         H_elements[3], H_elements[4], H_elements[5],
         H_elements[6], H_elements[7], H_elements[8];
    
    // SVD on CPU
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
    
    if (R.determinant() < 0) {
        Eigen::Matrix3f V = svd.matrixV();
        V.col(2) *= -1;
        R = V * svd.matrixU().transpose();
    }
    
    Eigen::Vector3f t = Eigen::Vector3f(center_tgt.x, center_tgt.y, center_tgt.z) - 
                        R * Eigen::Vector3f(center_src.x, center_src.y, center_src.z);
    
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;
}

// ============================================================================
// MAIN ICP LOOP
// ============================================================================

int main(int argc, char* argv[]) {
    std::string source_file = (argc > 1) ? argv[1] : "bun000.ply";
    std::string target_file = (argc > 2) ? argv[2] : "bun000_target.ply";

    auto source_cpu = loadPLY(source_file);
    auto target_cpu = loadPLY(target_file);

    if (source_cpu.empty() || target_cpu.empty()) {
        std::cerr << "Error: Could not load PLY files!" << std::endl;
        return -1;
    }

    std::cout << "Loaded " << source_cpu.size() << " source points, "
              << target_cpu.size() << " target points" << std::endl;

    // Create frames directory
    if (!createDirectory("frames")) {
        std::cerr << "Warning: Could not create frames directory" << std::endl;
    }

    // Convert to float3
    auto source_f3 = toFloat3(source_cpu);
    auto target_f3 = toFloat3(target_cpu);

    int n_source = source_f3.size();
    int n_target = target_f3.size();

    // Allocate GPU memory
    float3 *d_source, *d_target, *d_matched;
    float *d_distances;
    
    CUDA_CHECK(cudaMalloc(&d_source, n_source * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_target, n_target * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_matched, n_source * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_distances, n_source * sizeof(float)));

    // Copy target to GPU (stays constant)
    CUDA_CHECK(cudaMemcpy(d_target, target_f3.data(), 
                          n_target * sizeof(float3), cudaMemcpyHostToDevice));

    // Copy source to GPU
    CUDA_CHECK(cudaMemcpy(d_source, source_f3.data(), 
                          n_source * sizeof(float3), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    int numBlocks = (n_source + blockSize - 1) / blockSize;

    // ICP iterations
    for (int iter = 0; iter <= 50; ++iter) {
        // Save intermediate frames
        if (iter % 5 == 0) {
            CUDA_CHECK(cudaMemcpy(source_f3.data(), d_source, 
                                  n_source * sizeof(float3), cudaMemcpyDeviceToHost));
            savePLY("frames/iter_" + std::to_string(iter) + ".ply", 
                    toPoint3D(source_f3));
            std::cout << "Iteration " << iter << " completed" << std::endl;
        }

        // 1. Find nearest neighbors (GPU)
        findNearestNeighbors<<<numBlocks, blockSize>>>(
            d_source, n_source, d_target, n_target, d_matched, d_distances);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Compute centroids (GPU)
        float3 center_src = computeCentroidGPU(d_source, n_source);
        float3 center_tgt = computeCentroidGPU(d_matched, n_source);

        // 3. Compute transformation (hybrid GPU/CPU)
        Eigen::Matrix4f T = findRigidTransform(d_source, d_matched, n_source,
                                               center_src, center_tgt);

        // 4. Apply transformation (GPU)
        float h_transform[12];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                h_transform[i * 4 + j] = T(i, j);
            }
        }

        float* d_transform;
        CUDA_CHECK(cudaMalloc(&d_transform, 12 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_transform, h_transform, 12 * sizeof(float), 
                              cudaMemcpyHostToDevice));

        applyTransform<<<numBlocks, blockSize>>>(d_source, n_source, d_transform);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_transform));
    }

    // Final result
    CUDA_CHECK(cudaMemcpy(source_f3.data(), d_source, 
                          n_source * sizeof(float3), cudaMemcpyDeviceToHost));
    savePLY("frames/final_aligned.ply", toPoint3D(source_f3));

    // Cleanup
    CUDA_CHECK(cudaFree(d_source));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_matched));
    CUDA_CHECK(cudaFree(d_distances));

    std::cout << "ICP alignment completed!" << std::endl;
    return 0;
}