#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

struct Point3D {
    float x, y, z;
    Point3D(float x, float y, float z) : x(x), y(y), z(z) {}
};

void savePLY(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream file(filename);
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";
    
    for (const auto& p : points) {
        file << p.x << " " << p.y << " " << p.z << "\n";
    }
    
    file.close();
    std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
}

// Generate points on a bunny-like shape (simplified)
std::vector<Point3D> generateBunny(int num_points) {
    std::vector<Point3D> points;
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < num_points; i++) {
        // Generate points on/around a sphere with some deformation
        float theta = dis(gen) * 2.0f * M_PI;
        float phi = dis(gen) * M_PI;
        float r = 0.1f + dis(gen) * 0.02f; // Radius with variation
        
        // Add some bunny-like deformation
        float deform = 1.0f + 0.3f * std::sin(3.0f * theta) * std::sin(2.0f * phi);
        r *= deform;
        
        float x = r * std::sin(phi) * std::cos(theta);
        float y = r * std::sin(phi) * std::sin(theta);
        float z = r * std::cos(phi);
        
        points.push_back(Point3D(x, y, z));
    }
    
    return points;
}

// Apply transformation to points
void transform(std::vector<Point3D>& points, 
               float tx, float ty, float tz,
               float angle_degrees) {
    // Rotation around Z axis
    float angle = angle_degrees * M_PI / 180.0f;
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);
    
    for (auto& p : points) {
        // Rotate
        float x_rot = p.x * cos_a - p.y * sin_a;
        float y_rot = p.x * sin_a + p.y * cos_a;
        
        // Translate
        p.x = x_rot + tx;
        p.y = y_rot + ty;
        p.z = p.z + tz;
    }
}

// Add noise to points
void addNoise(std::vector<Point3D>& points, float noise_level) {
    std::random_device rd;
    std::mt19937 gen(123);
    std::normal_distribution<float> dis(0.0f, noise_level);
    
    for (auto& p : points) {
        p.x += dis(gen);
        p.y += dis(gen);
        p.z += dis(gen);
    }
}

int main() {
    std::cout << "Generating test data for ICP...\n\n";
    
    // Generate source point cloud
    int num_points = 1000; // Start small for testing
    std::cout << "Generating " << num_points << " points...\n";
    
    std::vector<Point3D> source = generateBunny(num_points);
    
    // Create target by transforming source
    std::vector<Point3D> target = source; // Copy
    
    // Apply known transformation
    float tx = 0.05f;      // Translation in X
    float ty = 0.03f;      // Translation in Y
    float tz = 0.02f;      // Translation in Z
    float angle = 15.0f;   // Rotation around Z (degrees)
    
    std::cout << "\nApplying transformation:\n";
    std::cout << "  Translation: [" << tx << ", " << ty << ", " << tz << "]\n";
    std::cout << "  Rotation: " << angle << " degrees around Z\n";
    
    transform(target, tx, ty, tz, angle);
    
    // Add small noise to make it realistic
    addNoise(target, 0.001f);
    
    // Save both
    savePLY("bunny_source.ply", source);
    savePLY("bunny_target.ply", target);
    
    std::cout << "\nTest data generated successfully!\n";
    
    return 0;
}