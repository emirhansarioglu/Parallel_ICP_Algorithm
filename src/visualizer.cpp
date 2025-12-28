#include <open3d/Open3D.h>
#include <thread>
#include <chrono>

int main() {
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("ICP Replay", 1024, 768);

    auto target = open3d::io::CreatePointCloudFromFile("bunny_target.ply");
    target->PaintUniformColor({0, 0, 1}); // Blue
    vis.AddGeometry(target);

    auto source = std::make_shared<open3d::geometry::PointCloud>();
    vis.AddGeometry(source);

    for (int i = 0; i <= 50; i += 5) {
        std::string path = "frames/iter_" + std::to_string(i) + ".ply";
        auto frame = open3d::io::CreatePointCloudFromFile(path);
        if (frame->IsEmpty()) continue;

        source->points_ = frame->points_;
        source->PaintUniformColor({1, 0, 0}); // Red
        
        vis.UpdateGeometry(source);
        vis.PollEvents();
        vis.UpdateRender();
        
        std::cout << "Playing Frame: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    vis.Run(); // Keep window open at the end
    vis.DestroyVisualizerWindow();
    return 0;
}