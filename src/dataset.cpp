#include "myslam/dataset.h"
#include "myslam/frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

namespace myslam {

Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

// 初始化数据集，读取相机标定参数并创建相机对象
bool Dataset::Init() {
    // 读取相机内参和外参
    // 打开标定文件，文件路径为 dataset_path_ + "/calib.txt"
    ifstream fin(dataset_path_ + "/calib.txt");
    // 检查文件是否成功打开
    if (!fin) {
        // 如果文件打开失败，记录错误日志并返回false
        LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    // 循环读取4个相机的标定数据
    for (int i = 0; i < 4; ++i) {
        char camera_name[3];
        // 读取相机名称（3个字符）
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        // 读取投影矩阵的12个数据（3x4矩阵）
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        
        // 从投影矩阵提取内参矩阵K（3x3）
        // 投影矩阵P = K[R|t]，这里假设R是单位矩阵
        Mat33 K;
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];
        
        // 提取平移向量t（投影矩阵的最后一列）
        Vec3 t;
        t << projection_data[3], projection_data[7], projection_data[11];
        
        // 将t从像素坐标系转换到相机坐标系：t_cam = K^{-1} * t_pixel
        t = K.inverse() * t;
        
        // 将内参矩阵K缩放0.5倍（可能是由于图像下采样）
        K = K * 0.5;
        
        // 创建相机对象
        // 参数：fx, fy, cx, cy, 基线长度, 相机位姿
        // 基线长度通过t的模长计算（双目相机基线）
        // 位姿设置为旋转为单位矩阵，平移为t
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), SE3(SO3(), t)));
        // 将相机对象添加到相机列表中
        cameras_.push_back(new_camera);
        // 记录相机外参信息
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    // 关闭标定文件
    fin.close();
    
    // 初始化当前图像索引为0（从第一帧开始）
    current_image_index_ = 0;
    // 初始化成功，返回true
    return true;
}

// 获取下一帧图像数据
Frame::Ptr Dataset::NextFrame() {
    // 定义图像文件名格式：数据集路径/image_相机编号/6位数字序号.png
    boost::format fmt("%s/image_%d/%06d.png");
    cv::Mat image_left, image_right;
    
    // 读取左右目图像
    // 左目图像：相机编号为0，当前图像索引
    image_left =
        cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);  // 以灰度图方式读取
    // 右目图像：相机编号为1，当前图像索引
    image_right =
        cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    // 检查图像是否成功读取
    if (image_left.data == nullptr || image_right.data == nullptr) {
        // 如果图像读取失败，记录警告日志并返回空指针
        LOG(WARNING) << "cannot find images at index " << current_image_index_;
        return nullptr;
    }

    // 创建缩放后的图像容器
    cv::Mat image_left_resized, image_right_resized;
    // 将图像尺寸缩放为原来的一半（宽高各乘以0.5）
    // 使用最近邻插值法保持图像清晰度
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);

    // 创建新的帧对象
    auto new_frame = Frame::CreateFrame();
    // 将缩放后的左右目图像赋值给帧对象
    new_frame->left_img_ = image_left_resized;
    new_frame->right_img_ = image_right_resized;
    
    // 图像索引自增，为读取下一帧做准备
    current_image_index_++;
    
    // 返回帧对象
    return new_frame;
}

}  // namespace myslam