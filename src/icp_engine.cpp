#include "icp_utils.h"
#include "kdtree.h"
#include <sys/stat.h>
#include <chrono>
#ifdef _WIN32
#include <direct.h>  
#endif

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
    return true;
}

int main(int argc, char* argv[]) {
    std::string source_file = (argc > 1) ? argv[1] : "bun000.ply";
    std::string target_file = (argc > 2) ? argv[2] : "bun000_target.ply";
    std::string method = (argc > 3) ? argv[3] : "brute";  // "brute" or "kdtree"

    auto source = loadPLY(source_file);
    auto target = loadPLY(target_file);

    if (source.empty() || target.empty()) {
        std::cerr << "Error: Could not load bunny files!" << std::endl;
        return -1;
    }

    bool use_kdtree = (method == "kd_tree");

    std::cout << "=== CPU ICP BENCHMARK (" << (use_kdtree ? "KD-Tree" : "Brute-Force") << ") ===" << std::endl;
    std::cout << "Source points: " << source.size() << std::endl;
    std::cout << "Target points: " << target.size() << std::endl;
    std::cout << "Method: " << (use_kdtree ? "O(N log N) with KD-Tree" : "O(N^2) Brute-Force") << std::endl;
    std::cout << std::endl;

    if (!createDirectory("frames")) {
        std::cerr << "Warning: Could not create frames directory" << std::endl;
    }

    // Timing variables
    double kdtree_build_time = 0.0;
    double total_nn_time = 0.0;
    double total_transform_time = 0.0;
    double total_apply_time = 0.0;
    
    auto program_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter <= 50; ++iter) {

        if (iter % 5 == 0) {
            savePLY("frames/iter_" + std::to_string(iter) + ".ply", source);
        }

        std::vector<Point3D> matched_target;

        if (use_kdtree) {
            // ========== KD-TREE METHOD ==========
            
            // Build KD-Tree for target
            auto kdtree_start = std::chrono::high_resolution_clock::now();
            KDTree kdtree;
            kdtree.build(target);
            auto kdtree_end = std::chrono::high_resolution_clock::now();
            kdtree_build_time += std::chrono::duration<double, std::milli>(kdtree_end - kdtree_start).count();

            // Find Nearest Neighbors using KD-Tree
            auto nn_start = std::chrono::high_resolution_clock::now();
            matched_target.reserve(source.size());
            for (const auto& s : source) {
                Point3D nearest = kdtree.findNearest(s);
                matched_target.push_back(nearest);
            }
            auto nn_end = std::chrono::high_resolution_clock::now();
            double nn_time = std::chrono::duration<double, std::milli>(nn_end - nn_start).count();
            total_nn_time += nn_time;

        } else {
            // ========== BRUTE-FORCE METHOD ==========
            
            auto nn_start = std::chrono::high_resolution_clock::now();
            for (const auto& s : source) {
                float min_d = 1e10;
                Point3D best_p = target[0];
                for (const auto& t : target) {
                    float d = distSq(s, t);
                    if (d < min_d) { min_d = d; best_p = t; }
                }
                matched_target.push_back(best_p);
            }
            auto nn_end = std::chrono::high_resolution_clock::now();
            double nn_time = std::chrono::duration<double, std::milli>(nn_end - nn_start).count();
            total_nn_time += nn_time;
        }

        // 2. Compute Transform 
        auto transform_start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix4f T = findRigidTransform(source, matched_target);
        auto transform_end = std::chrono::high_resolution_clock::now();
        double transform_time = std::chrono::duration<double, std::milli>(transform_end - transform_start).count();
        total_transform_time += transform_time;

        // 3. Update Source 
        auto apply_start = std::chrono::high_resolution_clock::now();
        for (auto& p : source) {
            Eigen::Vector4f v(p.x, p.y, p.z, 1.0f);
            v = T * v;
            p.x = v.x(); p.y = v.y(); p.z = v.z();
        }
        auto apply_end = std::chrono::high_resolution_clock::now();
        double apply_time = std::chrono::duration<double, std::milli>(apply_end - apply_start).count();
        total_apply_time += apply_time;
    }

    auto program_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(program_end - program_start).count();

    std::cout << "\n=== PERFORMANCE REPORT ===" << std::endl;
    std::cout << "Total execution time:    " << total_time << " ms (" << total_time/1000.0 << " s)" << std::endl;
    std::cout << "Average time per iter:   " << total_time/51.0 << " ms" << std::endl;
    std::cout << std::endl;
    
    if (use_kdtree) {
        std::cout << "KD-Tree build total:     " << kdtree_build_time << " ms (" << (kdtree_build_time/total_time)*100 << "%)" << std::endl;
    }
    std::cout << "Nearest Neighbor total:  " << total_nn_time << " ms (" << (total_nn_time/total_time)*100 << "%)" << std::endl;
    std::cout << "Transform compute total: " << total_transform_time << " ms (" << (total_transform_time/total_time)*100 << "%)" << std::endl;
    std::cout << "Apply transform total:   " << total_apply_time << " ms (" << (total_apply_time/total_time)*100 << "%)" << std::endl;
    std::cout << std::endl;
    
    if (use_kdtree) {
        std::cout << "Avg KD-Tree per iter:    " << kdtree_build_time/51.0 << " ms" << std::endl;
    }
    std::cout << "Avg NN per iter:         " << total_nn_time/51.0 << " ms" << std::endl;
    std::cout << "Avg Transform per iter:  " << total_transform_time/51.0 << " ms" << std::endl;
    std::cout << "Avg Apply per iter:      " << total_apply_time/51.0 << " ms" << std::endl;

    return 0;
}