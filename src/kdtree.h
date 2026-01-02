#ifndef KDTREE_H
#define KDTREE_H

#include "icp_utils.h"
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

struct KDNode {
    Point3D point;
    int left;   // Index to left child (-1 if none)
    int right;  // Index to right child (-1 if none)
    int axis;   // Split axis (0=x, 1=y, 2=z)
};

class KDTree {
private:
    std::vector<KDNode> nodes;
    std::vector<Point3D> points;
    
    int buildRecursive(std::vector<int>& indices, int start, int end, int depth) {
        if (start >= end) return -1;
        
        int axis = depth % 3;
        int mid = start + (end - start) / 2;
        
        // Partition around median
        std::nth_element(
            indices.begin() + start,
            indices.begin() + mid,
            indices.begin() + end,
            [&](int a, int b) { return points[a][axis] < points[b][axis]; }
        );
        
        int nodeIdx = nodes.size();
        nodes.push_back(KDNode());
        nodes[nodeIdx].point = points[indices[mid]];
        nodes[nodeIdx].axis = axis;
        nodes[nodeIdx].left = buildRecursive(indices, start, mid, depth + 1);
        nodes[nodeIdx].right = buildRecursive(indices, mid + 1, end, depth + 1);
        
        return nodeIdx;
    }
    
    void searchRecursive(int nodeIdx, const Point3D& query, 
                        Point3D& best, float& bestDist) const {
        if (nodeIdx == -1) return;
        
        const KDNode& node = nodes[nodeIdx];
        
        // Check current node
        float dx = query.x - node.point.x;
        float dy = query.y - node.point.y;
        float dz = query.z - node.point.z;
        float dist = dx*dx + dy*dy + dz*dz;
        
        if (dist < bestDist) {
            bestDist = dist;
            best = node.point;
        }
        
        // Determine which side to search first
        float diff = query[node.axis] - node.point[node.axis];
        int nearIdx = (diff < 0) ? node.left : node.right;
        int farIdx = (diff < 0) ? node.right : node.left;
        
        // Search near side
        searchRecursive(nearIdx, query, best, bestDist);
        
        // Check if we need to search far side
        if (diff * diff < bestDist) {
            searchRecursive(farIdx, query, best, bestDist);
        }
    }
    
public:
    void build(const std::vector<Point3D>& pts) {
        points = pts;
        nodes.clear();
        nodes.reserve(points.size());
        
        std::vector<int> indices(points.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        
        buildRecursive(indices, 0, indices.size(), 0);
    }
    
    Point3D findNearest(const Point3D& query) const {
        Point3D best = nodes[0].point;
        float bestDist = std::numeric_limits<float>::max();
        searchRecursive(0, query, best, bestDist);
        return best;
    }
    
    size_t size() const { return nodes.size(); }
};

#endif