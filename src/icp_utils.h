#ifndef ICP_UTILS_H
#define ICP_UTILS_H

#define NOMINMAX
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>

struct Point3D {
    float x, y, z;
    unsigned char r, g, b;  // RGB colors
    
    // Default constructor
    Point3D() : x(0), y(0), z(0), r(255), g(255), b(255) {}
    
    // Constructor without color
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_), r(255), g(255), b(255) {}
    
    // Constructor with color
    Point3D(float x_, float y_, float z_, unsigned char r_, unsigned char g_, unsigned char b_) 
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
    
    // Array access for KD-Tree (spatial coordinates only)
    float& operator[](int idx) {
        return (idx == 0) ? x : (idx == 1) ? y : z;
    }
    
    const float& operator[](int idx) const {
        return (idx == 0) ? x : (idx == 1) ? y : z;
    }
};

// Parse PLY header
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

// Load PLY (supports both ASCII and binary, with or without colors)
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
            } else {
                p.r = p.g = p.b = 255;
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
                } else {
                    p.r = p.g = p.b = 255;
                }
                points.push_back(p);
            }
        }
    }
    
    file.close();
    std::cout << "Loaded " << points.size() << " vertices from " << filename << std::endl;
    return points;
}

// Save PLY in ASCII format
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

// Save PLY in binary format
void savePLYBinary(const std::string& filename, const std::vector<Point3D>& points, bool save_colors = true) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }
    
    // Write header in ASCII
    std::string header = "ply\n";
    header += "format binary_little_endian 1.0\n";
    header += "element vertex " + std::to_string(points.size()) + "\n";
    header += "property float x\n";
    header += "property float y\n";
    header += "property float z\n";
    
    if (save_colors) {
        header += "property uchar diffuse_red\n";
        header += "property uchar diffuse_green\n";
        header += "property uchar diffuse_blue\n";
    }
    
    header += "end_header\n";
    file.write(header.c_str(), header.size());
    
    // Write binary data
    for (const auto& p : points) {
        file.write(reinterpret_cast<const char*>(&p.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&p.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&p.z), sizeof(float));
        
        if (save_colors) {
            file.write(reinterpret_cast<const char*>(&p.r), sizeof(unsigned char));
            file.write(reinterpret_cast<const char*>(&p.g), sizeof(unsigned char));
            file.write(reinterpret_cast<const char*>(&p.b), sizeof(unsigned char));
        }
    }
    
    file.close();
}

// Distance Helper (ignores color)
float distSq(const Point3D& a, const Point3D& b) {
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
}

// SVD Rigid Transformation (only uses spatial coordinates, preserves color)
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