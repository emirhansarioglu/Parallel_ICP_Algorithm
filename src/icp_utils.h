#ifndef ICP_UTILS_H
#define ICP_UTILS_H

#define NOMINMAX
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <Eigen/Dense>

struct Point3D {
    float x, y, z;
    
    float& operator[](int idx) {
        return (idx == 0) ? x : (idx == 1) ? y : z;
    }
    
    const float& operator[](int idx) const {
        return (idx == 0) ? x : (idx == 1) ? y : z;
    }
};

// Basic PLY Loader (ASCII)
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

// Basic PLY Saver (ASCII)
void savePLY(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream file(filename);
    file << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n";
    file << "property float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto& p : points) file << p.x << " " << p.y << " " << p.z << "\n";
}

// Distance Helper
float distSq(const Point3D& a, const Point3D& b) {
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
}

// SVD Rigid Transformation 
Eigen::Matrix4f findRigidTransform(const std::vector<Point3D>& src, const std::vector<Point3D>& tgt) {
    int N = src.size();
    Eigen::Vector3f center_src(0,0,0), center_tgt(0,0,0);
    for(int i=0; i<N; ++i) {
        center_src += Eigen::Vector3f(src[i].x, src[i].y, src[i].z);
        center_tgt += Eigen::Vector3f(tgt[i].x, tgt[i].y, tgt[i].z);
    }
    center_src /= (float)N;
    center_tgt /= (float)N;

    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for(int i=0; i<N; ++i) {
        Eigen::Vector3f s = Eigen::Vector3f(src[i].x, src[i].y, src[i].z) - center_src;
        Eigen::Vector3f t = Eigen::Vector3f(tgt[i].x, tgt[i].y, tgt[i].z) - center_tgt;
        H += s * t.transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
    
    // Reflection handle
    if (R.determinant() < 0) {
        Eigen::Matrix3f V = svd.matrixV();
        V.col(2) *= -1;
        R = V * svd.matrixU().transpose();
    }

    Eigen::Vector3f t = center_tgt - R * center_src;
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;
}

#endif