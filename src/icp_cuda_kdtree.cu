#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <chrono>
#include <algorithm>
#include <Eigen/Dense>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

struct Point3D {
    float x, y, z;
    unsigned char r, g, b;
    
    Point3D() : x(0), y(0), z(0), r(255), g(255), b(255) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_), r(255), g(255), b(255) {}
    Point3D(float x_, float y_, float z_, unsigned char r_, unsigned char g_, unsigned char b_) 
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
    
    __host__ __device__ float& operator[](int idx) {
        return (idx == 0) ? x : (idx == 1) ? y : z;
    }
    
    __host__ __device__ const float& operator[](int idx) const {
        return (idx == 0) ? x : (idx == 1) ? y : z;
    }
};

struct KDNode {
    float3 point;
    int left;
    int right;
    int axis;
};

// ============================================================================
// CUDA KD-TREE KERNELS
// ============================================================================

__device__ void searchKDTree(const KDNode* nodes, int nodeIdx, const float3& query,
                             float3& best, float& bestDist) {
    if (nodeIdx == -1) return;
    
    const KDNode& node = nodes[nodeIdx];
    
    // Check current node distance
    float dx = query.x - node.point.x;
    float dy = query.y - node.point.y;
    float dz = query.z - node.point.z;
    float dist = dx*dx + dy*dy + dz*dz;
    
    if (dist < bestDist) {
        bestDist = dist;
        best = node.point;
    }
    
    // Determine which side to search
    float query_val = (node.axis == 0) ? query.x : (node.axis == 1) ? query.y : query.z;
    float node_val = (node.axis == 0) ? node.point.x : (node.axis == 1) ? node.point.y : node.point.z;
    float diff = query_val - node_val;
    
    int nearIdx = (diff < 0) ? node.left : node.right;
    int farIdx = (diff < 0) ? node.right : node.left;
    
    // Search near side first
    searchKDTree(nodes, nearIdx, query, best, bestDist);
    
    // Check if we need to search far side
    if (diff * diff < bestDist) {
        searchKDTree(nodes, farIdx, query, best, bestDist);
    }
}

__global__ void findNearestKDTree(
    const float3* source, int n_source,
    const KDNode* kdtree, int tree_size,
    float3* matched_target)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_source) return;
    
    float3 query = source[idx];
    float3 best = kdtree[0].point;
    float bestDist = 1e10f;
    
    searchKDTree(kdtree, 0, query, best, bestDist);
    
    matched_target[idx] = best;
}

// ============================================================================
// OTHER CUDA KERNELS
// ============================================================================

__global__ void computeCentroid(const float3* points, int n, float3* partial_sums) {
    __shared__ float3 shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        shared_sum[tid] = points[idx];
    } else {
        shared_sum[tid] = make_float3(0.0f, 0.0f, 0.0f);
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid].x += shared_sum[tid + stride].x;
            shared_sum[tid].y += shared_sum[tid + stride].y;
            shared_sum[tid].z += shared_sum[tid + stride].z;
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

__global__ void computeCovarianceMatrix(
    const float3* source, const float3* target, int n,
    float3 center_src, float3 center_tgt,
    float* H_elements)
{
    __shared__ float shared_H[9][256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int i = 0; i < 9; ++i) {
                shared_H[i][tid] += shared_H[i][tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        for (int i = 0; i < 9; ++i) {
            atomicAdd(&H_elements[i], shared_H[i][0]);
        }
    }
}

__global__ void applyTransform(float3* points, int n, const float* transform) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float3 p = points[idx];
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

bool createDirectory(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        #ifdef _WIN32
            return _mkdir(path) == 0;
        #else
            return mkdir(path, 0700) == 0;
        #endif
    }
    return true;
}

// PLY Header parser
struct PLYHeader {
    int vertex_count = 0;
    bool is_binary = false;
    bool has_color = false;
};

PLYHeader parsePLYHeader(std::ifstream& file) {
    PLYHeader header;
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;
        
        if (token == "format") {
            std::string format_type;
            ss >> format_type;
            header.is_binary = (format_type.find("binary") != std::string::npos);
        }
        else if (token == "element") {
            std::string element_type;
            int count;
            ss >> element_type >> count;
            if (element_type == "vertex") {
                header.vertex_count = count;
            }
        }
        else if (token == "property") {
            std::string prop_type, prop_name;
            ss >> prop_type >> prop_name;
            if (prop_name.find("red") != std::string::npos || 
                prop_name.find("diffuse_red") != std::string::npos) {
                header.has_color = true;
            }
        }
        else if (token == "end_header") {
            break;
        }
    }
    
    return header;
}

// Load PLY with color support
std::vector<Point3D> loadPLY(const std::string& filename) {
    std::vector<Point3D> points;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return points;
    }
    
    PLYHeader header = parsePLYHeader(file);
    
    if (header.vertex_count == 0) {
        std::cerr << "Error: No vertices found in PLY header" << std::endl;
        return points;
    }
    
    points.reserve(header.vertex_count);
    
    if (header.is_binary) {
        // Binary format
        for (int i = 0; i < header.vertex_count; ++i) {
            Point3D p;
            file.read(reinterpret_cast<char*>(&p.x), sizeof(float));
            file.read(reinterpret_cast<char*>(&p.y), sizeof(float));
            file.read(reinterpret_cast<char*>(&p.z), sizeof(float));
            
            if (header.has_color) {
                file.read(reinterpret_cast<char*>(&p.r), sizeof(unsigned char));
                file.read(reinterpret_cast<char*>(&p.g), sizeof(unsigned char));
                file.read(reinterpret_cast<char*>(&p.b), sizeof(unsigned char));
            }
            
            points.push_back(p);
        }
    } else {
        // ASCII format
        std::string line;
        while (std::getline(file, line) && points.size() < static_cast<size_t>(header.vertex_count)) {
            std::stringstream ss(line);
            Point3D p;
            
            if (ss >> p.x >> p.y >> p.z) {
                if (header.has_color) {
                    int r, g, b;
                    if (ss >> r >> g >> b) {
                        p.r = static_cast<unsigned char>(r);
                        p.g = static_cast<unsigned char>(g);
                        p.b = static_cast<unsigned char>(b);
                    }
                }
                points.push_back(p);
            }
        }
    }
    
    file.close();
    std::cout << "Loaded " << points.size() << " vertices from " << filename << std::endl;
    return points;
}

// Save PLY with color support
void savePLY(const std::string& filename, const std::vector<Point3D>& points, bool save_colors = true) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    
    if (save_colors) {
        file << "property uchar diffuse_red\n";
        file << "property uchar diffuse_green\n";
        file << "property uchar diffuse_blue\n";
    }
    
    file << "end_header\n";
    
    for (const auto& p : points) {
        file << p.x << " " << p.y << " " << p.z;
        if (save_colors) {
            file << " " << static_cast<int>(p.r) 
                 << " " << static_cast<int>(p.g) 
                 << " " << static_cast<int>(p.b);
        }
        file << "\n";
    }
    
    file.close();
}

std::vector<float3> toFloat3(const std::vector<Point3D>& points) {
    std::vector<float3> result(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        result[i] = make_float3(points[i].x, points[i].y, points[i].z);
    }
    return result;
}

std::vector<Point3D> toPoint3D(const std::vector<float3>& points, const std::vector<Point3D>& original) {
    std::vector<Point3D> result(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        result[i].x = points[i].x;
        result[i].y = points[i].y;
        result[i].z = points[i].z;
        // Preserve original colors
        if (i < original.size()) {
            result[i].r = original[i].r;
            result[i].g = original[i].g;
            result[i].b = original[i].b;
        }
    }
    return result;
}

// CPU KD-Tree builder (uses only spatial coordinates, ignores color)
int buildKDTreeCPU(std::vector<Point3D>& points, std::vector<int>& indices, 
                   std::vector<KDNode>& nodes, int start, int end, int depth) {
    if (start >= end) return -1;
    
    int axis = depth % 3;
    int mid = start + (end - start) / 2;
    
    std::nth_element(
        indices.begin() + start,
        indices.begin() + mid,
        indices.begin() + end,
        [&](int a, int b) { return points[a][axis] < points[b][axis]; }
    );
    
    int nodeIdx = nodes.size();
    KDNode node;
    node.point = make_float3(points[indices[mid]].x, points[indices[mid]].y, points[indices[mid]].z);
    node.axis = axis;
    nodes.push_back(node);
    
    nodes[nodeIdx].left = buildKDTreeCPU(points, indices, nodes, start, mid, depth + 1);
    nodes[nodeIdx].right = buildKDTreeCPU(points, indices, nodes, mid + 1, end, depth + 1);
    
    return nodeIdx;
}

float3 computeCentroidGPU(float3* d_points, int n) {
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float3* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(float3)));
    
    computeCentroid<<<numBlocks, blockSize>>>(d_points, n, d_partial_sums);
    
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

Eigen::Matrix4f findRigidTransform(float3* d_source, float3* d_matched, int n,
                                   float3 center_src, float3 center_tgt) {
    const int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    float* d_H;
    CUDA_CHECK(cudaMalloc(&d_H, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_H, 0, 9 * sizeof(float)));
    
    computeCovarianceMatrix<<<numBlocks, blockSize>>>(
        d_source, d_matched, n, center_src, center_tgt, d_H);
    
    float H_elements[9];
    CUDA_CHECK(cudaMemcpy(H_elements, d_H, 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_H));
    
    Eigen::Matrix3f H;
    H << H_elements[0], H_elements[1], H_elements[2],
         H_elements[3], H_elements[4], H_elements[5],
         H_elements[6], H_elements[7], H_elements[8];
    
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
// MAIN ICP LOOP WITH KD-TREE
// ============================================================================

int main(int argc, char* argv[]) {
    std::string source_file = (argc > 1) ? argv[1] : "data/bun000.ply";
    std::string target_file = (argc > 2) ? argv[2] : "data/bun000_target.ply";

    auto source_cpu = loadPLY(source_file);
    auto target_cpu = loadPLY(target_file);

    if (source_cpu.empty() || target_cpu.empty()) {
        std::cerr << "Error: Could not load PLY files!" << std::endl;
        return -1;
    }

    std::cout << "=== CUDA ICP BENCHMARK (with KD-Tree) ===" << std::endl;
    std::cout << "Source points: " << source_cpu.size() << std::endl;
    std::cout << "Target points: " << target_cpu.size() << std::endl;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << std::endl;

    if (!createDirectory("frames")) {
        std::cerr << "Warning: Could not create frames directory" << std::endl;
    }

    auto source_f3 = toFloat3(source_cpu);
    auto target_f3 = toFloat3(target_cpu);

    int n_source = source_f3.size();
    int n_target = target_f3.size();

    // Timing variables
    double kdtree_build_time = 0.0;
    double kdtree_upload_time = 0.0;
    double total_nn_time = 0.0;
    double total_centroid_time = 0.0;
    double total_transform_time = 0.0;
    double total_apply_time = 0.0;
    double total_memcpy_time = 0.0;

    // Allocate GPU memory
    float3 *d_source, *d_matched;
    
    CUDA_CHECK(cudaMalloc(&d_source, n_source * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_matched, n_source * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_source, source_f3.data(), 
                          n_source * sizeof(float3), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    int numBlocks = (n_source + blockSize - 1) / blockSize;

    auto program_start = std::chrono::high_resolution_clock::now();

    // ICP iterations
    for (int iter = 0; iter <= 50; ++iter) {

        if (iter % 5 == 0) {
            auto save_start = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpy(source_f3.data(), d_source, 
                                  n_source * sizeof(float3), cudaMemcpyDeviceToHost));
            savePLY("frames/iter_" + std::to_string(iter) + ".ply", 
                    toPoint3D(source_f3, source_cpu));
            auto save_end = std::chrono::high_resolution_clock::now();
            total_memcpy_time += std::chrono::duration<double, std::milli>(save_end - save_start).count();
        }

        // Build KD-Tree on CPU
        auto kdtree_start = std::chrono::high_resolution_clock::now();
        std::vector<KDNode> kdtree_nodes;
        kdtree_nodes.reserve(target_cpu.size());
        std::vector<int> indices(target_cpu.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        buildKDTreeCPU(target_cpu, indices, kdtree_nodes, 0, target_cpu.size(), 0);
        auto kdtree_end = std::chrono::high_resolution_clock::now();
        kdtree_build_time += std::chrono::duration<double, std::milli>(kdtree_end - kdtree_start).count();

        // Upload KD-Tree to GPU
        auto upload_start = std::chrono::high_resolution_clock::now();
        KDNode* d_kdtree;
        CUDA_CHECK(cudaMalloc(&d_kdtree, kdtree_nodes.size() * sizeof(KDNode)));
        CUDA_CHECK(cudaMemcpy(d_kdtree, kdtree_nodes.data(), 
                              kdtree_nodes.size() * sizeof(KDNode), cudaMemcpyHostToDevice));
        auto upload_end = std::chrono::high_resolution_clock::now();
        kdtree_upload_time += std::chrono::duration<double, std::milli>(upload_end - upload_start).count();

        // 1. Find nearest neighbors using KD-Tree
        auto nn_start = std::chrono::high_resolution_clock::now();
        findNearestKDTree<<<numBlocks, blockSize>>>(
            d_source, n_source, d_kdtree, kdtree_nodes.size(), d_matched);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto nn_end = std::chrono::high_resolution_clock::now();
        total_nn_time += std::chrono::duration<double, std::milli>(nn_end - nn_start).count();

        CUDA_CHECK(cudaFree(d_kdtree));

        // 2. Compute centroids
        auto centroid_start = std::chrono::high_resolution_clock::now();
        float3 center_src = computeCentroidGPU(d_source, n_source);
        float3 center_tgt = computeCentroidGPU(d_matched, n_source);
        auto centroid_end = std::chrono::high_resolution_clock::now();
        total_centroid_time += std::chrono::duration<double, std::milli>(centroid_end - centroid_start).count();

        // 3. Compute transformation
        auto transform_start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix4f T = findRigidTransform(d_source, d_matched, n_source,
                                               center_src, center_tgt);
        auto transform_end = std::chrono::high_resolution_clock::now();
        total_transform_time += std::chrono::duration<double, std::milli>(transform_end - transform_start).count();

        // 4. Apply transformation
        auto apply_start = std::chrono::high_resolution_clock::now();
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
        auto apply_end = std::chrono::high_resolution_clock::now();
        total_apply_time += std::chrono::duration<double, std::milli>(apply_end - apply_start).count();
    }

    auto program_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(program_end - program_start).count();

    // Final result
    CUDA_CHECK(cudaMemcpy(source_f3.data(), d_source, 
                          n_source * sizeof(float3), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_source));
    CUDA_CHECK(cudaFree(d_matched));

    // Print statistics
    std::cout << "\n=== PERFORMANCE REPORT ===" << std::endl;
    std::cout << "Total execution time:    " << total_time << " ms (" << total_time/1000.0 << " s)" << std::endl;
    std::cout << "Average time per iter:   " << total_time/51.0 << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "KD-Tree build total:     " << kdtree_build_time << " ms (" << (kdtree_build_time/total_time)*100 << "%)" << std::endl;
    std::cout << "KD-Tree upload total:    " << kdtree_upload_time << " ms (" << (kdtree_upload_time/total_time)*100 << "%)" << std::endl;
    std::cout << "Nearest Neighbor total:  " << total_nn_time << " ms (" << (total_nn_time/total_time)*100 << "%)" << std::endl;
    std::cout << "Centroid compute total:  " << total_centroid_time << " ms (" << (total_centroid_time/total_time)*100 << "%)" << std::endl;
    std::cout << "Transform compute total: " << total_transform_time << " ms (" << (total_transform_time/total_time)*100 << "%)" << std::endl;
    std::cout << "Apply transform total:   " << total_apply_time << " ms (" << (total_apply_time/total_time)*100 << "%)" << std::endl;
    std::cout << "File I/O & memcpy total: " << total_memcpy_time << " ms (" << (total_memcpy_time/total_time)*100 << "%)" << std::endl;
    std::cout << std::endl;
    std::cout << "Avg KD-Tree per iter:    " << kdtree_build_time/51.0 << " ms" << std::endl;
    std::cout << "Avg Upload per iter:     " << kdtree_upload_time/51.0 << " ms" << std::endl;
    std::cout << "Avg NN per iter:         " << total_nn_time/51.0 << " ms" << std::endl;

    return 0;
}