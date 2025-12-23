//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/viewer.h"
#include "myslam/feature.h"
#include "myslam/frame.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam {
// 定义命名空间 myslam，用于封装整个SLAM系统的类

Viewer::Viewer() {
    // Viewer类的构造函数
    viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
    // 创建一个新线程，绑定到ThreadLoop成员函数
    // this指针作为参数传递给绑定的成员函数
    // 线程将在后台运行，负责可视化界面的渲染和更新
}

void Viewer::Close() {
    // 关闭可视化器
    viewer_running_ = false;  // 设置运行标志为false，通知线程退出
    viewer_thread_.join();    // 等待线程结束
}

void Viewer::AddCurrentFrame(Frame::Ptr current_frame) {
    // 添加当前帧到可视化器
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    // 获取互斥锁，保护共享数据的线程安全
    current_frame_ = current_frame;  // 更新当前帧指针
}

void Viewer::UpdateMap() {
    // 更新地图数据
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    // 获取互斥锁
    assert(map_ != nullptr);  // 确保地图指针不为空
    active_keyframes_ = map_->GetActiveKeyFrames();  // 从地图获取活跃关键帧
    active_landmarks_ = map_->GetActiveMapPoints();  // 从地图获取活跃地图点
    map_updated_ = true;  // 设置地图已更新标志
}

void Viewer::ThreadLoop() {
    // 可视化线程的主循环函数
    pangolin::CreateWindowAndBind("MySLAM", 1024, 768);
    // 创建Pangolin窗口，标题"MySLAM"，分辨率1024x768
    
    glEnable(GL_DEPTH_TEST);  // 启用深度测试，用于3D渲染
    glEnable(GL_BLEND);       // 启用混合，用于透明效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // 设置混合函数：源颜色*源alpha + 目标颜色*(1-源alpha)
    
    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        // 创建投影矩阵：窗口宽高，焦距(fx,fy)，光心(cx,cy)，近平面0.1，远平面1000
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));
        // 设置相机初始位姿：从(0,-5,-10)看向(0,0,0)，上方向为(0,-1,0)

    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            // 设置显示边界：上下左右边界(0-1)，宽高比
            .SetHandler(new pangolin::Handler3D(vis_camera));
            // 设置3D交互处理器，绑定到相机状态

    const float blue[3] = {0, 0, 1};   // 蓝色，用于某些绘制
    const float green[3] = {0, 1, 0};  // 绿色，用于绘制当前帧

    while (!pangolin::ShouldQuit() && viewer_running_) {
        // 主循环：当窗口未关闭且viewer_running_为true时继续
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // 清除颜色缓冲和深度缓冲
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        // 设置清除颜色为白色
        vis_display.Activate(vis_camera);
        // 激活显示视图和相机状态

        std::unique_lock<std::mutex> lock(viewer_data_mutex_);
        // 获取互斥锁，保护共享数据
        if (current_frame_) {
            // 如果当前帧存在
            DrawFrame(current_frame_, green);  // 绘制当前帧
            FollowCurrentFrame(vis_camera);    // 相机跟随当前帧

            cv::Mat img = PlotFrameImage();    // 生成帧图像
            cv::imshow("image", img);          // 显示图像
            cv::waitKey(1);                    // 等待1ms，刷新OpenCV窗口
        }

        if (map_) {
            // 如果地图存在
            DrawMapPoints();  // 绘制地图点
        }

        pangolin::FinishFrame();  // 结束当前帧的渲染
        usleep(5000);  // 休眠5ms，控制循环频率
    }

    LOG(INFO) << "Stop viewer";  // 记录日志：可视化器停止
}

cv::Mat Viewer::PlotFrameImage() {
    // 生成当前帧的图像用于显示
    cv::Mat img_out;
    cv::cvtColor(current_frame_->left_img_, img_out, CV_GRAY2BGR);
    // 将灰度图转换为BGR彩色图
    
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // 遍历左图的所有特征点
        if (current_frame_->features_left_[i]->map_point_.lock()) {
            // 如果特征点有对应的地图点（使用weak_ptr，需要lock检查）
            auto feat = current_frame_->features_left_[i];  // 获取特征点
            cv::circle(img_out, feat->position_.pt, 2, cv::Scalar(0, 250, 0), 2);
            // 在地图点位置绘制绿色圆点，半径2像素，线宽2
        }
    }
    return img_out;  // 返回处理后的图像
}

void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    // 设置相机跟随当前帧
    SE3 Twc = current_frame_->Pose().inverse();
    // 获取当前帧的位姿Twc（世界到相机），并求逆得到Tcw（相机到世界）
    pangolin::OpenGlMatrix m(Twc.matrix());
    // 将SE3转换为Pangolin的OpenGL矩阵
    vis_camera.Follow(m, true);
    // 设置相机跟随该矩阵，true表示保持相对位置
}

void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
    // 绘制一个关键帧（相机坐标系）
    SE3 Twc = frame->Pose().inverse();
    // 获取相机在世界坐标系中的位姿
    
    // 定义相机可视化参数
    const float sz = 1.0;          // 缩放因子
    const int line_width = 2.0;    // 线宽
    const float fx = 400;          // 相机焦距x
    const float fy = 400;          // 相机焦距y
    const float cx = 512;          // 相机光心x
    const float cy = 384;          // 相机光心y
    const float width = 1080;      // 图像宽度
    const float height = 768;      // 图像高度

    glPushMatrix();  // 保存当前变换矩阵

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    // 将SE3矩阵转换为float类型的4x4矩阵
    glMultMatrixf((GLfloat*)m.data());
    // 应用变换矩阵，将后续绘制变换到相机坐标系

    // 设置绘制颜色
    if (color == nullptr) {
        glColor3f(1, 0, 0);  // 默认红色
    } else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(line_width);  // 设置线宽
    glBegin(GL_LINES);        // 开始绘制线条
    
    // 绘制相机视锥体（金字塔形状）
    // 从相机中心(0,0,0)到图像四个角在归一化平面上的投影
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    
    // 绘制图像平面的边框
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();        // 结束绘制
    glPopMatrix();  // 恢复之前的变换矩阵
}

void Viewer::DrawMapPoints() {
    // 绘制地图点和关键帧
    const float red[3] = {1.0, 0, 0};  // 红色，用于绘制关键帧
    
    // 绘制所有活跃关键帧
    for (auto& kf : active_keyframes_) {
        DrawFrame(kf.second, red);  // 绘制关键帧为红色相机框
    }

    // 绘制所有活跃地图点
    glPointSize(2);  // 设置点大小
    glBegin(GL_POINTS);  // 开始绘制点
    for (auto& landmark : active_landmarks_) {
        auto pos = landmark.second->Pos();  // 获取地图点的3D位置
        glColor3f(red[0], red[1], red[2]);  // 设置颜色为红色
        glVertex3d(pos[0], pos[1], pos[2]); // 绘制点
    }
    glEnd();  // 结束绘制
}

}  // namespace myslam
