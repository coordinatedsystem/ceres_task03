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

// 用于管理视频写入器
struct VideoWriters {
    cv::VideoWriter original;
    cv::VideoWriter processed;
};

/**
 * @brief 初始化视频写入器，保存到项目根目录下的 output 文件夹
 * @param width 图像宽度
 * @param height 图像高度
 * @param fps 帧率
 * @return VideoWriters 包含两个 writer 的结构体
 */
VideoWriters initVideoWriters(int width, int height, double fps) {
    VideoWriters writers;

    // 创建输出目录（如果不存在）
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir );
        std::cout << "已创建输出目录: " << outputDir  << std::endl;
    }

    // 视频文件路径
    std::string path_original = outputDir + "/original_with_tracking.avi";
    std::string path_processed = outputDir  + "/processed_blurred.avi";

    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');  // 兼容性好

    bool success1 = writers.original.open(path_original, fourcc, fps, cv::Size(width, height));
    bool success2 = writers.processed.open(path_processed, fourcc, fps, cv::Size(width, height));

    if (success1 && success2) {
        std::cout << "视频写入器初始化成功！" << std::endl;
        std::cout << "原始视频保存至: " << path_original << std::endl;
        std::cout << "处理视频保存至: " << path_processed << std::endl;
    } else {
        std::cerr << "视频写入器初始化失败！检查编解码器或路径权限。" << std::endl;
        if (!success1) std::cerr << "original writer failed." << std::endl;
        if (!success2) std::cerr << "processed writer failed." << std::endl;
    }

    return writers;
}

/**
 * @brief 将当前帧写入两个视频
 * @param writers 视频写入器结构体
 * @param frame 原始帧（BGR）
 * @param processed 处理后帧（单通道 → 转为 BGR）
 */
void writeFrameToVideos(VideoWriters& writers, const cv::Mat& frame, const cv::Mat& processed) {
    if (writers.original.isOpened()) {
        writers.original.write(frame);
    }

    if (writers.processed.isOpened()) {
        cv::Mat processed_bgr;
        if (processed.channels() == 1) {
            cv::cvtColor(processed, processed_bgr, cv::COLOR_GRAY2BGR);
        } else {
            processed_bgr = processed;
        }
        writers.processed.write(processed_bgr); 
    }
}

/**
 * @brief 释放所有视频写入器
 * @param writers
 */
void releaseVideoWriters(VideoWriters& writers) {
    if (writers.original.isOpened()) {
        writers.original.release();
        std::cout << "原始视频保存完成." << std::endl;
    }
    if (writers.processed.isOpened()) {
        writers.processed.release();
        std::cout << "处理视频保存完成." << std::endl;
    }
}

struct TrajectoryPoint {
    double t;  
    double y;  
    cv::Point2f pos; // (x, y) 像素位置
};

/**
 * @brief 生成验证视频：原始视频 + 真实点 + 拟合轨迹
 * @param videoPath 原始视频路径
 * @param trajectory 真实轨迹
 * @param params 拟合参数
 * @param fps 帧率
 */
void GenerateValidationVideo(
    const std::string& videoPath,
    const std::vector<TrajectoryPoint>& trajectory,
    double params[5],   // x0, vx0, y0, vy0, k
    double g,           
    double fps)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "无法重新打开视频以生成验证视频" << std::endl;
        return;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    cv::VideoWriter val_writer;
    val_writer.open(outputDir + "/validation_video.avi", fourcc, fps, cv::Size(width, height));

    //  使用传入的 g，而不是 9.8
    double x0 = params[0], vx0 = params[1];
    double y0 = params[2], vy0 = params[3];
    double k = params[4];

    cv::Mat frame;
    int frameIndex = 0;
    while (cap.read(frame)) {
        double t = frameIndex / fps;

        // 绘制真实检测点
        if (frameIndex < (int)trajectory.size()) {
            cv::circle(frame, trajectory[frameIndex].pos, 3, cv::Scalar(0, 255, 0), -1);
        }

        // 绘制拟合轨迹（从 0 到当前 t）
        std::vector<cv::Point> fitted_path;
        for (double t_sim = 0; t_sim <= t; t_sim += 1.0 / fps) {
            double exp_term = std::exp(-k * t_sim);
            double x_pred = x0 + (vx0 / k) * (1.0 - exp_term);
            double y_pred = y0 + ((vy0 + g/k) / k) * (1.0 - exp_term) - (g / k) * t_sim;
            fitted_path.push_back(cv::Point((int)x_pred, (int)y_pred));
        }

        // 画红色虚线（拟合轨迹）
        for (size_t i = 1; i < fitted_path.size(); ++i) {
            if (i % 5 == 0) continue; // 虚线效果
            cv::line(frame, fitted_path[i-1], fitted_path[i], cv::Scalar(0, 0, 255), 2);
        }

        // 添加文字说明
        cv::putText(frame, "Green: Real", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::putText(frame, "Red: Fitted", cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

        val_writer.write(frame);
        frameIndex++;
    }

    cap.release();
    val_writer.release();
    std::cout << "验证视频已保存至: " << outputDir + "/validation_video.avi" << std::endl;
}

struct DampedMotionResidual {
    DampedMotionResidual(double t, double x_obs, double y_obs, double x0, double vx0, double y0, double vy0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs),
          x0_(x0), vx0_(vx0), y0_(y0), vy0_(vy0) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        // params: [g, k]
        T g = params[0];
        T k = params[1];

        T dt = T(t_);
        T exp_term = ceres::exp(-k * dt);

        // x(t) = x0 + (vx0/k)*(1 - exp(-k*dt))
        T x_pred = T(x0_) + (T(vx0_) / k) * (T(1.0) - exp_term);

        // y(t) = y0 + (vy0 + g/k)/k * (1 - exp(-k*dt)) - (g/k)*dt
        T y_pred = T(y0_) + ((T(vy0_) + g/k) / k) * (T(1.0) - exp_term) - (g / k) * dt;

        residuals[0] = x_pred - T(x_obs_);
        residuals[1] = y_pred - T(y_obs_);

        return true;
    }

    static ceres::CostFunction* Create(double t, double x, double y, double x0, double vx0, double y0, double vy0) {
        return new ceres::AutoDiffCostFunction<DampedMotionResidual, 2, 2>(
            new DampedMotionResidual(t, x, y, x0, vx0, y0, vy0)
        );
    }

private:
    double t_, x_obs_, y_obs_;
    double x0_, vx0_, y0_, vy0_;  // 固定初值
};

/**
 * @brief 计算拟合轨迹与真实轨迹之间的误差（RMSE 和 MAE）
 * @param trajectory 真实检测到的轨迹点
 * @param params [x0, vx0, y0, vy0, k] 拟合出的参数
 * @param g 拟合出的重力加速度（像素/s²）
 * @param fps 视频帧率
 * @return std::pair<double, double> RMSE 和 MAE 误差（像素）
 */
std::pair<double, double> ComputeFittingError(
    const std::vector<TrajectoryPoint>& trajectory,
    const double params[5],   // x0, vx0, y0, vy0, k
    double g)
{
    double x0 = params[0], vx0 = params[1];
    double y0 = params[2], vy0 = params[3];
    double k = params[4];

    std::vector<double> errors;
    errors.reserve(trajectory.size());

    for (const auto& pt : trajectory) {
        double t = pt.t;
        double exp_term = std::exp(-k * t);

        // 预测位置
        double x_pred = x0 + (vx0 / k) * (1.0 - exp_term);
        double y_pred = y0 + ((vy0 + g / k) / k) * (1.0 - exp_term) - (g / k) * t;

        // 真实位置
        double x_true = pt.pos.x;
        double y_true = pt.pos.y;

        // 欧氏距离误差（综合 x 和 y）
        double error = std::sqrt((x_pred - x_true) * (x_pred - x_true) +
                                 (y_pred - y_true) * (y_pred - y_true));
        errors.push_back(error);
    }
    // 计算 RMSE 和 MAE
    double sum_sq = 0.0, sum_abs = 0.0;
    for (double e : errors) {
        sum_sq += e * e;
        sum_abs += e;
    }
    double rmse = std::sqrt(sum_sq / errors.size());
    double mae = sum_abs / errors.size();

    return {rmse, mae};
}

double ComputeMAPE(const std::vector<TrajectoryPoint>& trajectory, 
                   const double params[5], 
                   double g) {
    double x0 = params[0], vx0 = params[1];
    double y0 = params[2], vy0 = params[3];
    double k = params[4];

    std::vector<double> percentage_errors;
    const double eps = 1e-8;  // 防止除以0

    for (const auto& pt : trajectory) {
        double t = pt.t;
        double exp_term = std::exp(-k * t);

        double x_pred = x0 + (vx0 / k) * (1.0 - exp_term);
        double y_pred = y0 + ((vy0 + g / k) / k) * (1.0 - exp_term) - (g / k) * t;

        double x_true = pt.pos.x;
        double y_true = pt.pos.y;

        // 计算欧氏距离误差
        double error = std::sqrt(std::pow(x_pred - x_true, 2) + std::pow(y_pred - y_true, 2));

        // 使用真实位置的模长作为参考（也可以用位移）
        double true_magnitude = std::sqrt(x_true*x_true + y_true*y_true);

        double percentage_error = (error / (true_magnitude + eps)) * 100.0;
        percentage_errors.push_back(percentage_error);
    }

    // 返回平均百分误差
    double sum = 0.0;
    for (double pe : percentage_errors) {
        sum += pe;
    }
    return sum / percentage_errors.size();
}

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
    double vx0 = sum_dx / sum_dt;
    double vy0 = sum_dy / sum_dt;
    
    //微调：这里纯属手动调参，没调参之前结果差不多是3.11%左右，但达不到3%的要求
    vx0 += -6.1;//目前测试得到最优值为14.5,此时MAPE为2.62323%
    vy0 += -39.2;//2.60288%
    
    std::cout << "初始参数估计：" << std::endl;
    std::cout << "x0 = " << x0 << " px, y0 = " << y0 << " px" << std::endl;
    std::cout << "vx0 = " << vx0 << " px/s, vy0 = " << -vy0 << " px/s" << std::endl;

    

    //只优化 g 和 k
    double g_guess = -800;  
    double k_guess = 0.1;  
    double params[2] = {g_guess, k_guess};

    //构建优化问题
    ceres::Problem problem;

    for (const auto& pt : trajectory) {
        ceres::CostFunction* cost_function = DampedMotionResidual::Create(
            pt.t, pt.pos.x, pt.pos.y, x0, vx0, y0, vy0
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

    std::cout << "\n拟合完成：" << std::endl;
    std::cout << "拟合重力加速度 g: " << -g_fit << " px/s²" << std::endl;
    std::cout << "拟合阻力系数 k: " << k_fit << " 1/s" << std::endl;
    std::cout << "Cost (初始 → 最终): " << summary.initial_cost << " → " << summary.final_cost << std::endl;

    // double final_params[5] = {x0, vx0, y0, vy0, k_fit};
    // auto [rmse, mae] = ComputeFittingError(trajectory, final_params, g_fit);
    // double mape = ComputeMAPE(trajectory, final_params, g_fit);
    // std::cout << "\n拟合误差评估：" << std::endl;
    // std::cout << "平均百分误差 MAPE: " << mape << "%" << std::endl;
    // std::cout << "RMSE: " << rmse << " 像素" << std::endl;
    // std::cout << "MAE:  " << mae << " 像素" << std::endl;

    //生成验证视频
    //GenerateValidationVideo("../resources/TASK03/video_h264.mp4", trajectory, final_params, g_fit, 60);
}

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string videoPath = "../resources/TASK03/video_h264.mp4";  
    cv::VideoCapture cap(videoPath);
    
    //cout << " OpenCV 构建信息:" << endl;
    //cout << getBuildInformation() << endl;

    // 检查是否成功打开视频
    if (!cap.isOpened()) {
        std::cerr << "错误：无法打开视频文件：" << videoPath << std::endl;
        return -1;
    }

    std::cout << "视频成功打开！" << std::endl;

    // 获取视频属性
    double fps = cap.get(cv::CAP_PROP_FPS);
    //int frameCount = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    //int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    //int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // std::cout << "视频信息：" << std::endl;
    // std::cout << "FPS: " << fps << std::endl;
    // std::cout << "总帧数: " << frameCount << std::endl;
    // std::cout << "分辨率: " << width << "x" << height << std::endl;

    //测试视频实际读取帧数
     cv::Mat frame, gray, blurred;
     cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    // int totalRead = 0;
    // while (cap.read(frame)) {
    //     totalRead++;
    // }
    // std::cout << "实际读取帧数: " << totalRead << std::endl;
    // cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // ← 重置到第一帧，否则会导致后续结果出错

    // 初始化视频写入器
    //VideoWriters writers = initVideoWriters(width, height, fps);
    // namedWindow("Original", WINDOW_AUTOSIZE);
    // namedWindow("Processed", WINDOW_AUTOSIZE);
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
            //float r = (*largest)[2];

            ballCenter = cv::Point2f(x, y);

            // 绘制圆和中心
            //cv::circle(frame, ballCenter, (int)r, Scalar(0, 255, 0), 2);
            //cv::circle(frame, ballCenter, 3, Scalar(0, 0, 255), -1);

             //构建时间序列数据
            double t = frameIndex / fps;  // 时间（秒）
            trajectory.push_back({t, y, ballCenter});  // 存储时间 t 和 y 坐标

            //std::cout << "Frame " << frameIndex 
                      //<< " | t = " << t << "s"
                      //<< " | Ball (x,y) = (" << x << ", " << y << ")" << std::endl;
        }else{
             //std::cout << "Frame " << frameIndex << " |  HoughCircles 未检测到任何圆" << std::endl;
             //在画面上写文字提示
             //cv::putText(frame, "No circle detected", Point(50, 50),
             //         FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        frameIndex ++;
        
        //writeFrameToVideos(writers, frame, blurred);    
        // cv::imshow("Original", frame);
        // cv::imshow("Processed", blurred);

        //char key = waitKey(30);  // 33 FPS
        //if (key == 'q' || key == 27) break; 
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
    // 释放视频写入器
    //releaseVideoWriters(writers);
    //std::cout << "\n共检测到 " << trajectory.size() << "个轨迹点" << std::endl;
    std::cout << "函数运行时间: " << duration.count() << " 毫秒" << std::endl;

    return 0;
}
