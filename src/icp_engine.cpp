#include "icp_utils.h"
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string source_file = (argc > 1) ? argv[1] : "bun000.ply";
    std::string target_file = (argc > 2) ? argv[2] : "bun000_target.ply";

    auto source = loadPLY(source_file);
    auto target = loadPLY(target_file);

    if (source.empty() || target.empty()) {
        std::cerr << "Error: Could not load bunny files!" << std::endl;
        return -1;
    }

    if (!fs::exists("frames")) {
        fs::create_directory("frames");
    }

    for (int iter = 0; iter <= 50; ++iter) {
        if (iter % 5 == 0) {
            savePLY("frames/iter_" + std::to_string(iter) + ".ply", source);
            std::cout << "Saved frame " << iter << std::endl;
        }

        // 1. Find Nearest Neighbors (Brute Force)
        std::vector<Point3D> matched_target;
        for (const auto& s : source) {
            float min_d = 1e10;
            Point3D best_p = target[0];
            for (const auto& t : target) {
                float d = distSq(s, t);
                if (d < min_d) { min_d = d; best_p = t; }
            }
            matched_target.push_back(best_p);
        }

        // 2. Compute Transform & Update Source
        Eigen::Matrix4f T = findRigidTransform(source, matched_target);
        for (auto& p : source) {
            Eigen::Vector4f v(p.x, p.y, p.z, 1.0f);
            v = T * v;
            p.x = v.x(); p.y = v.y(); p.z = v.z();
        }
    }
    return 0;
}