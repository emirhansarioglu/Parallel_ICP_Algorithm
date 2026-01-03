#include "icp_utils.h"
#include <Eigen/Geometry>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // 1. Check if a filename was provided
    if (argc < 2) {
        std::cerr << "Usage: generate_data.exe <filename.ply>" << std::endl;
        return -1;
    }

    std::string source_path = argv[1];
    
    // 2. Load the source file
    auto source_points = loadPLY(source_path);
    if (source_points.empty()) {
        std::cerr << "Error: Could not read " << source_path << std::endl;
        return -1;
    }

    // 3. Create the Target filename (e.g., bun_000.ply -> bun_000_target.ply)
    fs::path p(source_path);
    std::string target_path = "data/" + p.stem().string() + "_target.ply";

    // 4. Create a "Hard" Transformation (Target is a rotated/shifted version)
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

    // Rotate 30 degrees around Y and 15 around X to make it challenging
    float angleY = 30.0f * EIGEN_PI / 180.0f;
    float angleX = 15.0f * EIGEN_PI / 180.0f;
    Eigen::Matrix3f R = (Eigen::AngleAxisf(angleY, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(angleX, Eigen::Vector3f::UnitX())).toRotationMatrix();

    transform.block<3,3>(0,0) = R;
    transform(0,3) = 0.3f; // Shift X
    transform(1,3) = 0.1f; // Shift Y

    std::vector<Point3D> target_points = source_points;
    for (auto& pt : target_points) {
        Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0f);
        v = transform * v;
        pt.x = v.x(); pt.y = v.y(); pt.z = v.z();
    }

    // 5. Save the Target file
    savePLY(target_path, target_points);

    std::cout << "Successfully created: " << target_path << " from " << source_path << std::endl;
    std::cout << "Apply these to your ICP: Source=" << source_path << ", Target=" << target_path << std::endl;

    return 0;
}