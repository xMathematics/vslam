//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

// 前端跟踪函数：估计当前帧的位姿，并决定是否插入关键帧
bool Frontend::Track() {
    // 如果存在上一帧，使用相对运动来预测当前帧的位姿
    // relative_motion_ 是上一帧到当前帧的位姿变换
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    // 跟踪上一帧的特征点，返回成功跟踪的特征点数量
    int num_track_last = TrackLastFrame();
    
    // 基于跟踪的特征点估计当前帧的位姿，并返回内点数量
    tracking_inliers_ = EstimateCurrentPose();

    // 根据内点数量判断跟踪状态
    if (tracking_inliers_ > num_features_tracking_) {
        // 跟踪良好：内点数量超过良好阈值
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // 跟踪较差：内点数量超过较差阈值但未达到良好阈值
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // 跟踪丢失：内点数量过低
        status_ = FrontendStatus::LOST;
    }

    // 尝试插入关键帧（根据内点数量决定是否真正插入）
    InsertKeyframe();
    
    // 更新相对运动：计算当前帧到上一帧的位姿变换
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    // 如果可视化器存在，将当前帧添加到可视化器中
    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    
    return true;
}

// 插入关键帧函数：根据当前跟踪质量决定是否插入关键帧
bool Frontend::InsertKeyframe() {
    // 如果内点数量足够多，不需要插入关键帧
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        return false;
    }
    
    // 将当前帧标记为关键帧
    current_frame_->SetKeyFrame();
    
    // 将关键帧插入到地图中
    map_->InsertKeyFrame(current_frame_);

    // 记录关键帧信息
    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    // 为关键帧中的特征点建立与地图点的观测关系
    SetObservationsForKeyFrame();
    
    // 在左图中检测新的特征点
    DetectFeatures();
    
    // 在右图中跟踪特征点（立体匹配）
    FindFeaturesInRight();
    
    // 对新的特征点进行三角化，生成新的地图点
    TriangulateNewPoints();
    
    // 通知后端更新地图（因为有关键帧插入）
    backend_->UpdateMap();

    // 如果可视化器存在，更新地图显示
    if (viewer_) viewer_->UpdateMap();

    return true;
}

// 为关键帧建立观测关系：将关键帧中的特征点与地图点关联
void Frontend::SetObservationsForKeyFrame() {
    // 遍历当前帧左图的所有特征点
    for (auto &feat : current_frame_->features_left_) {
        // 获取特征点关联的地图点（weak_ptr需要lock()获取shared_ptr）
        auto mp = feat->map_point_.lock();
        
        // 如果地图点存在，将该特征点添加为地图点的观测
        if (mp) mp->AddObservation(feat);
    }
}

// 三角化新的地图点：利用左右目的匹配特征点生成三维地图点
int Frontend::TriangulateNewPoints() {
    // 获取左右相机的位姿（从相机坐标系到世界坐标系）
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    
    // 获取当前帧的世界坐标系到相机坐标系的变换（Twc的逆是Tcw，但这里注释说是Twc）
    // 注意：通常Pose()返回的是Tcw（世界到相机），所以inverse()得到的是Twc（相机到世界）
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    
    // 统计成功三角化的点数
    int cnt_triangulated_pts = 0;
    
    // 遍历左图的所有特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // 条件：左图特征点没有关联地图点（expired()检查weak_ptr是否失效），且右图对应特征点存在
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            
            // 将左右图的像素坐标转换到相机归一化平面（去除内参影响）
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            
            // 初始化三维点坐标
            Vec3 pworld = Vec3::Zero();

            // 调用三角化函数，并检查三角化是否成功且深度为正
            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                // 创建新的地图点
                auto new_map_point = MapPoint::CreateNewMappoint();
                
                // 将三角化得到的点从当前帧相机坐标系转换到世界坐标系
                pworld = current_pose_Twc * pworld;
                
                // 设置地图点的三维坐标
                new_map_point->SetPos(pworld);
                
                // 将左右图的特征点添加为该地图点的观测
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]);

                // 将地图点关联到左右图的特征点
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                
                // 将地图点插入地图
                map_->InsertMapPoint(new_map_point);
                
                // 计数增加
                cnt_triangulated_pts++;
            }
        }
    }
    // 记录新生成的地图点数量
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

// 估计当前帧的位姿：使用g2o优化，基于已有关联地图点的特征点
int Frontend::EstimateCurrentPose() {
    // 设置g2o优化器
    // 定义块求解器：优化变量维度为6（位姿），误差维度为3（地图点位置）
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    // 定义线性求解器类型（稠密矩阵）
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    
    // 创建优化算法：Levenberg-Marquardt
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    
    // 创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 添加顶点：当前帧的位姿
    VertexPose *vertex_pose = new VertexPose();  // 相机位姿顶点
    vertex_pose->setId(0);  // 设置顶点ID
    vertex_pose->setEstimate(current_frame_->Pose());  // 设置初始估计值
    optimizer.addVertex(vertex_pose);  // 将顶点添加到优化器

    // 获取相机内参矩阵
    Mat33 K = camera_left_->K();

    // 添加边
    int index = 1;  // 边的ID从1开始（0是顶点）
    std::vector<EdgeProjectionPoseOnly *> edges;  // 保存边的指针
    std::vector<Feature::Ptr> features;  // 保存对应的特征点
    
    // 遍历左图的所有特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // 获取特征点关联的地图点
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        // 如果地图点存在，则添加重投影误差边
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            
            // 创建边：地图点的三维位置和相机内参作为参数
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);  // 连接到位姿顶点
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));  // 设置观测值（像素坐标）
            edge->setInformation(Eigen::Matrix2d::Identity());  // 信息矩阵设为单位矩阵
            edge->setRobustKernel(new g2o::RobustKernelHuber);  // 设置鲁棒核函数，抑制异常值
            
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // 开始优化：使用卡方检验剔除异常值
    const double chi2_th = 5.991;  // 卡方检验阈值（对应95%置信度，自由度2）
    int cnt_outlier = 0;
    
    // 迭代4次优化，每次优化后根据误差剔除异常值
    for (int iteration = 0; iteration < 4; ++iteration) {
        // 设置初始估计值
        vertex_pose->setEstimate(current_frame_->Pose());
        // 初始化优化
        optimizer.initializeOptimization();
        // 执行10次优化迭代
        optimizer.optimize(10);
        cnt_outlier = 0;

        // 统计异常值数量
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            // 如果特征点已经是异常值，则计算误差（用于后续判断是否恢复为内点）
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            // 如果边的卡方值大于阈值，标记为异常值
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);  // 设置为level 1，下次优化时不再使用
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);  // 设置为level 0，下次优化时继续使用
            };

            // 在第3次迭代时移除鲁棒核函数，进行更精确的优化
            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    // 记录内点和异常值的数量
    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    
    // 将优化后的位姿赋值给当前帧
    current_frame_->SetPose(vertex_pose->estimate());

    // 输出当前帧的位姿矩阵
    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    // 处理异常值：断开异常值特征点与地图点的关联，并重置异常值标记
    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();  // 重置关联的地图点
            feat->is_outlier_ = false;  // 重置异常值标记，未来可能重新使用
        }
    }
    
    // 返回内点数量
    return features.size() - cnt_outlier;
}

// 跟踪上一帧的特征点：使用光流法在当前帧中跟踪上一帧的特征点
int Frontend::TrackLastFrame() {
    // 准备LK光流法需要的点集：上一帧的特征点和当前帧的初始估计点
    std::vector<cv::Point2f> kps_last, kps_current;
    
    // 遍历上一帧左图的所有特征点
    for (auto &kp : last_frame_->features_left_) {
        // 如果特征点关联有有效的地图点
        if (kp->map_point_.lock()) {
            // 获取地图点
            auto mp = kp->map_point_.lock();
            // 将地图点投影到当前帧的像素坐标系
            auto px = camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            // 上一帧的特征点位置
            kps_last.push_back(kp->position_.pt);
            // 当前帧的初始估计位置（通过地图点投影得到）
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // 如果没有关联地图点，则假设特征点在两帧之间位置不变（无运动先验）
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    // 光流跟踪状态向量和误差矩阵
    std::vector<uchar> status;
    Mat error;
    
    // 使用金字塔LK光流法进行特征点跟踪
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_,  // 上一帧左图
        current_frame_->left_img_,  // 当前帧左图
        kps_last,  // 上一帧特征点位置
        kps_current,  // 当前帧特征点位置（输入为初始估计，输出为跟踪结果）
        status,  // 跟踪状态，1表示成功，0表示失败
        error,  // 跟踪误差
        cv::Size(11, 11),  // 搜索窗口大小
        3,  // 金字塔层数
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),  // 停止条件：最大迭代30次或误差<0.01
        cv::OPTFLOW_USE_INITIAL_FLOW  // 使用初始估计值
    );

    // 统计成功跟踪的点数
    int num_good_pts = 0;

    // 处理跟踪结果
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {  // 如果跟踪成功
            // 创建关键点对象，大小为7像素
            cv::KeyPoint kp(kps_current[i], 7);
            // 创建特征点对象，与当前帧关联
            Feature::Ptr feature(new Feature(current_frame_, kp));
            // 继承上一帧特征点的地图点关联
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            // 将特征点添加到当前帧
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    // 记录成功跟踪的特征点数量
    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

// 双目初始化：检测特征点并进行立体匹配，构建初始地图
bool Frontend::StereoInit() {
    // 在左图中检测特征点
    int num_features_left = DetectFeatures();
    // 在右图中跟踪左图的特征点（立体匹配）
    int num_coor_features = FindFeaturesInRight();
    
    // 如果成功匹配的特征点数量不足，初始化失败
    if (num_coor_features < num_features_init_) {
        return false;
    }

    // 构建初始地图（三角化生成初始地图点）
    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        // 设置前端状态为跟踪良好
        status_ = FrontendStatus::TRACKING_GOOD;
        // 如果可视化器存在，更新显示
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

// 检测特征点：在当前帧左图中检测新的特征点
int Frontend::DetectFeatures() {
    // 创建掩膜，初始值为255（全白，表示所有区域都可以检测特征点）
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    
    // 遍历当前帧已有的特征点，在掩膜上将特征点周围的区域设为0（黑色）
    // 避免在已有特征点附近重复检测
    for (auto &feat : current_frame_->features_left_) {
        // 以特征点为中心，创建20x20像素的矩形区域（上下左右各10像素）
        cv::rectangle(mask, 
                     feat->position_.pt - cv::Point2f(10, 10),  // 矩形左上角
                     feat->position_.pt + cv::Point2f(10, 10),  // 矩形右下角
                     0,  // 填充颜色为0（黑色）
                     CV_FILLED);  // 填充整个矩形
    }

    // 存储检测到的关键点
    std::vector<cv::KeyPoint> keypoints;
    // 使用GFTT（Good Features to Track）检测器检测特征点
    // 参数mask指定了哪些区域不检测特征点
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    
    // 统计新检测到的特征点数量
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        // 为每个关键点创建特征点对象，并添加到当前帧
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    // 记录新检测的特征点数量
    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

// 在右图中寻找左图特征点的对应点：使用光流法进行立体匹配
int Frontend::FindFeaturesInRight() {
    // 使用LK光流法在右图中估计左图特征点的位置
    std::vector<cv::Point2f> kps_left, kps_right;
    
    // 遍历当前帧左图的所有特征点
    for (auto &kp : current_frame_->features_left_) {
        // 左图特征点的位置
        kps_left.push_back(kp->position_.pt);
        
        // 获取特征点关联的地图点
        auto mp = kp->map_point_.lock();
        if (mp) {
            // 如果有关联地图点，则将地图点投影到右图作为光流的初始估计
            auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // 如果没有地图点，则使用左图的相同位置作为初始估计（假设无水平位移）
            kps_right.push_back(kp->position_.pt);
        }
    }

    // 光流跟踪状态向量和误差矩阵
    std::vector<uchar> status;
    Mat error;
    
    // 使用金字塔LK光流法从左图到右图进行特征点跟踪
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_,  // 左图
        current_frame_->right_img_,  // 右图
        kps_left,  // 左图特征点位置
        kps_right,  // 右图特征点位置（输入为初始估计，输出为跟踪结果）
        status,  // 跟踪状态，1表示成功，0表示失败
        error,  // 跟踪误差
        cv::Size(11, 11),  // 搜索窗口大小
        3,  // 金字塔层数
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),  // 停止条件
        cv::OPTFLOW_USE_INITIAL_FLOW  // 使用初始估计值
    );

    // 统计成功匹配的点数
    int num_good_pts = 0;
    
    // 处理匹配结果
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {  // 如果匹配成功
            // 创建关键点对象，大小为7像素
            cv::KeyPoint kp(kps_right[i], 7);
            // 创建特征点对象，与当前帧关联
            Feature::Ptr feat(new Feature(current_frame_, kp));
            // 标记该特征点不在左图上（在右图上）
            feat->is_on_left_image_ = false;
            // 将特征点添加到当前帧的右图特征点列表
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            // 如果匹配失败，对应位置设为nullptr
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    
    // 记录在右图中成功匹配的特征点数量
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

// 构建初始地图：通过三角化左右目匹配点生成初始地图点
bool Frontend::BuildInitMap() {
    // 获取左右相机的位姿（从相机坐标系到世界坐标系）
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    
    // 统计成功创建的初始地图点数量
    size_t cnt_init_landmarks = 0;
    
    // 遍历左图所有特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // 如果右图没有对应的特征点，跳过
        if (current_frame_->features_right_[i] == nullptr) continue;
        
        // 创建地图点：通过三角化
        // 将左右图的像素坐标转换到相机归一化平面
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        
        // 初始化三维点坐标
        Vec3 pworld = Vec3::Zero();

        // 调用三角化函数，检查三角化是否成功且深度为正
        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            // 创建新的地图点
            auto new_map_point = MapPoint::CreateNewMappoint();
            // 设置地图点的三维坐标（这里pworld已经在相机坐标系中）
            new_map_point->SetPos(pworld);
            // 将左右图的特征点添加为该地图点的观测
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            // 将地图点关联到左右图的特征点
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            // 计数增加
            cnt_init_landmarks++;
            // 将地图点插入地图
            map_->InsertMapPoint(new_map_point);
        }
    }
    
    // 将当前帧设置为关键帧
    current_frame_->SetKeyFrame();
    // 将关键帧插入地图
    map_->InsertKeyFrame(current_frame_);
    // 通知后端更新地图（因为有关键帧插入）
    backend_->UpdateMap();

    // 记录初始地图创建信息
    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

// 重置前端：目前未实现具体功能
bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam