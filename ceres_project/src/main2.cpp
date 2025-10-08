#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <ceres/ceres.h>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

#ifdef OUTPUT_DIR
    const string outputDir = OUTPUT_DIR; // 使用 CMake 定义的输出目录
#else
    const string outputDir = "output"; // 默认输出目录
#endif

struct TrajectoryPoint {
    double t;  
    double y;  
    cv::Point2f pos; // (x, y) 像素位置
};


struct DampedMotionResidual {
    DampedMotionResidual(double t, double x_obs, double y_obs, double x0, double y0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        T g = params[0];      // 重力加速度
        T k = params[1];      // 阻力系数
        T vx0 = params[2];    // 水平初速度
        T vy0 = params[3];    // 垂直初速度

        T dt = T(t_);
        T exp_term = ceres::exp(-k * dt);

        // x(t) = x0 + (vx0 / k) * (1 - exp(-k*dt))
        T x_pred = T(x0_) + (vx0 / k) * (T(1.0) - exp_term);

        // y(t) = y0 + (vy0 + g/k)/k * (1 - exp(-k*dt)) - (g/k)*dt
        T y_pred = T(y0_) + ((vy0 + g / k) / k) * (T(1.0) - exp_term) - (g / k) * dt;

        residuals[0] = x_pred - T(x_obs_);
        residuals[1] = y_pred - T(y_obs_);

        return true;
    }

    // 工厂方法：创建自动微分代价函数
    // 注意：2 个残差，4 个待优化参数（g, k, vx0, vy0）
    static ceres::CostFunction* Create(double t, double x, double y, double x0, double y0) {
        return new ceres::AutoDiffCostFunction<DampedMotionResidual, 2, 4>(
            new DampedMotionResidual(t, x, y, x0, y0)
        );
    }

private:
    double t_;       // 当前时间 t
    double x_obs_;   // 观测到的 x 坐标
    double y_obs_;   // 观测到的 y 坐标
    double x0_;      // 固定初始 x 位置
    double y0_;      // 固定初始 y 位置
};

void FitTrajectory(const std::vector<TrajectoryPoint>& trajectory) {
    if (trajectory.size() < 2) {
        std::cout << "轨迹点不足，无法拟合" << std::endl;
        return;
    }

    std::cout << "\n开始拟合阻尼运动模型（仅优化 g 和 k）..." << std::endl;

    //固定初始位置
    double x0 = trajectory.front().pos.x;
    double y0 = trajectory.front().pos.y;

    // 用前 10 帧做线性回归估计初速度
    double sum_dt = 0.0, sum_dx = 0.0, sum_dy = 0.0;
    for (int i = 1; i < std::min(10, (int)trajectory.size()); ++i) {
        double dt = trajectory[i].t - trajectory[0].t;
        sum_dt += dt;
        sum_dx += trajectory[i].pos.x - trajectory[0].pos.x;
        sum_dy += trajectory[i].pos.y - trajectory[0].pos.y;
    }
    double vx0_guess = sum_dx / sum_dt;
    double vy0_guess = sum_dy / sum_dt;
    
    std::cout << "初始参数估计：" << std::endl;
    std::cout << "x0 = " << x0 << " px, y0 = " << y0 << " px" << std::endl;

    double g_guess = -800;  
    double k_guess = 0.1;  
    double params[4] = {g_guess, k_guess, vx0_guess, vy0_guess};

    //构建优化问题
    ceres::Problem problem;

    for (const auto& pt : trajectory) {
        ceres::CostFunction* cost_function = DampedMotionResidual::Create(
            pt.t, pt.pos.x, pt.pos.y, x0, y0
        );
        problem.AddResidualBlock(cost_function, nullptr, params);
    }

    //添加边界约束
    problem.SetParameterLowerBound(params, 0, -1000.0);   
    problem.SetParameterUpperBound(params, 0, -100.0); 

    problem.SetParameterLowerBound(params, 1, 0.01);  
    problem.SetParameterUpperBound(params, 1, 1.0);   
    
    //求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    double g_fit = params[0];
    double k_fit = params[1];
    double vx0_fit = params[2];
    double vy0_fit = params[3];

    std::cout << "\n拟合完成：" << std::endl;
    std::cout << "拟合重力加速度 g: " << -g_fit << " px/s²" << std::endl;
    std::cout << "拟合阻力系数 k: " << k_fit << " 1/s" << std::endl;
    std::cout << "拟合水平方向初速度：" << vx0_fit << "px/s" <<std::endl;
    std::cout << "拟合竖直方向初速度：" << -vy0_fit << "px/s" <<std::endl;
    std::cout << "Cost (初始 → 最终): " << summary.initial_cost << " → " << summary.final_cost << std::endl;

}

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string videoPath = "../resources/TASK03/video_h264.mp4";  
    cv::VideoCapture cap(videoPath);
    
    // 获取视频属性
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "FPS: " << fps << std::endl;
    cv::Mat frame, gray, blurred;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    
    int frameIndex = 0;
    std::vector<TrajectoryPoint> trajectory;
    while (cap.read(frame)) { 
        //这里使用滤波加灰度图对每一帧的图像进行处理
        if (frame.empty()) {
        std::cout << "读取到空帧，退出" << std::endl;
        break;
        }
        
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5,5), 1.0);

        //这里进行hough圆检测
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(
            blurred,            // 输入图像
            circles,            // 输出：检测到的圆
            cv::HOUGH_GRADIENT, // 方法
            2,                  // dp: 分辨率比例
            100,                 // minDist: 圆心最小距离
            40,                // canny 高阈值
            10,                 // 中心检测阈值（越小越敏感）
            1,                 // 最小半径（像素）
            300                 // 最大半径
        );

        cv::Point2f ballCenter;

        //接下来取最大圆作为小球
        if (!circles.empty()) {
            // 选择半径最大的圆
            auto largest = std::max_element(circles.begin(), circles.end(),
                [](const cv::Vec3f& a, const cv::Vec3f& b) {
                    return a[2] < b[2]; // 比较半径
                });
            float x = (*largest)[0];
            float y = (*largest)[1];

            ballCenter = cv::Point2f(x, y);

            double t = frameIndex / fps;  // 时间（秒）
            trajectory.push_back({t, y, ballCenter});  // 存储时间 t 和 y 坐标

        frameIndex ++;
        
        char key = waitKey(30);  // 33 FPS
        if (key == 'q' || key == 27) break; 
    }
}

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    if (!trajectory.empty()) {
        FitTrajectory(trajectory);
    } else {
        std::cout << "未检测到任何轨迹点，无法拟合" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cap.release();

    std::cout << "\n共检测到 " << trajectory.size() << "个轨迹点" << std::endl;
    std::cout << "函数运行时间: " << duration.count() << " 毫秒" << std::endl;

    return 0;
}
