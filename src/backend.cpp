//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

// Backend类的构造函数实现
Backend::Backend() {
    // 将backend_running_原子变量设置为true，表示后端线程正在运行
    backend_running_.store(true);
    
    // 创建并启动后端线程，绑定BackendLoop成员函数作为线程执行体
    // std::bind用于将成员函数与对象实例绑定，this指向当前Backend对象
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

// 触发地图更新的函数
void Backend::UpdateMap() {
    // 创建互斥锁的独占锁，保护共享数据的访问
    // data_mutex_是Backend类的成员变量，用于同步数据访问
    std::unique_lock<std::mutex> lock(data_mutex_);
    
    // 通知一个等待在map_update_条件变量上的线程（即后端线程）
    // 这通常是前端有新的关键帧或地图点需要优化时调用
    map_update_.notify_one();
}

// 停止后端线程的函数
void Backend::Stop() {
    // 将运行标志设置为false，通知后端线程退出循环
    backend_running_.store(false);
    
    // 再次通知条件变量，确保后端线程能够及时退出等待状态
    map_update_.notify_one();
    
    // 等待后端线程执行完成（线程join），确保线程安全退出
    backend_thread_.join();
}

// 后端线程的主循环函数
void Backend::BackendLoop() {
    // 当backend_running_为true时持续循环执行优化任务
    while (backend_running_.load()) {
        // 创建互斥锁的独占锁，保护共享数据的访问
        // 注意：这个锁会在wait调用时暂时释放，被notify时重新获取
        std::unique_lock<std::mutex> lock(data_mutex_);
        
        // 等待条件变量的通知（由UpdateMap()或Stop()触发）
        // wait调用时会暂时释放锁，被唤醒后重新获取锁
        // 防止了忙等待，提高了CPU效率
        map_update_.wait(lock);
        
        /// 后端优化处理开始
        // 从地图中获取所有激活的关键帧（需要优化的关键帧）
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        
        // 从地图中获取所有激活的地图点（需要优化的地图点）
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        
        // 调用优化函数，对激活的关键帧和地图点进行优化
        // 这通常是指捆集调整（Bundle Adjustment, BA）优化
        Optimize(active_kfs, active_landmarks);
    }
}

// Backend类中的优化函数，用于对关键帧和路标点进行图优化
void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &landmarks) {
    // 1. 设置g2o优化器
    // 定义块求解器：优化变量维度为6(位姿:旋转3+平移3)和3(路标点:3D坐标)
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    // 使用CSparse作为线性求解器
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    // 创建Levenberg-Marquardt优化算法
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    // 创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);  // 设置优化算法

    // 2. 添加关键帧位姿顶点（使用SE(3)位姿顶点）
    // 存储关键帧ID到顶点指针的映射
    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;  // 记录最大关键帧ID，用于路标点顶点ID偏移
    // 遍历所有关键帧
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;  // 获取关键帧对象
        VertexPose *vertex_pose = new VertexPose();  // 创建位姿顶点
        vertex_pose->setId(kf->keyframe_id_);  // 设置顶点ID为关键帧ID
        vertex_pose->setEstimate(kf->Pose());  // 设置顶点初始值为当前位姿
        optimizer.addVertex(vertex_pose);  // 将顶点添加到优化器
        
        // 更新最大关键帧ID
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        // 存储顶点指针到map中，便于后续查找
        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // 3. 添加路标点顶点（使用3D点顶点）
    // 存储路标点ID到顶点指针的映射
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    // 4. 获取相机参数
    // 相机内参矩阵
    Mat33 K = cam_left_->K();
    // 左右相机之间的外参变换
    SE3 left_ext = cam_left_->pose();   // 左相机外参（通常为单位变换）
    SE3 right_ext = cam_right_->pose(); // 右相机外参

    // 5. 添加边（观测约束）
    int index = 1;  // 边ID计数器
    double chi2_th = 5.991;  // 鲁棒核函数阈值（对应95%置信度的卡方分布）
    // 存储边与特征的对应关系，用于后续异常值处理
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    // 遍历所有路标点
    for (auto &landmark : landmarks) {
        // 跳过异常路标点
        if (landmark.second->is_outlier_) continue;
        
        unsigned long landmark_id = landmark.second->id_;  // 路标点ID
        auto observations = landmark.second->GetObs();  // 获取该路标点的所有观测
        
        // 遍历路标点的所有观测
        for (auto &obs : observations) {
            if (obs.lock() == nullptr) continue;  // 如果观测已失效，跳过
            
            auto feat = obs.lock();  // 获取特征点
            // 如果特征点为异常或所属关键帧已失效，跳过
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

            auto frame = feat->frame_.lock();  // 获取观测到该路标点的关键帧
            
            // 根据特征点在左/右图像，创建不同的边
            EdgeProjection *edge = nullptr;
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);  // 左目观测边
            } else {
                edge = new EdgeProjection(K, right_ext); // 右目观测边
            }

            // 如果路标点顶点尚未添加到优化器中，则创建并添加
            if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;  // 创建3D点顶点
                v->setEstimate(landmark.second->Pos());  // 设置初始值为路标点位置
                // 设置顶点ID：为了避免与位姿顶点ID冲突，加上最大关键帧ID+1
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);  // 设置边缘化，提升计算效率
                vertices_landmarks.insert({landmark_id, v});  // 存储顶点指针
                optimizer.addVertex(v);  // 添加到优化器
            }

            // 添加边：连接位姿顶点和路标点顶点
            // 确保两个顶点都已存在于优化器中
            if (vertices.find(frame->keyframe_id_) != vertices.end() && 
                vertices_landmarks.find(landmark_id) != vertices_landmarks.end()) {
                    edge->setId(index);  // 设置边ID
                    edge->setVertex(0, vertices.at(frame->keyframe_id_));    // 第一个顶点：位姿
                    edge->setVertex(1, vertices_landmarks.at(landmark_id));  // 第二个顶点：路标点
                    edge->setMeasurement(toVec2(feat->position_.pt));  // 测量值：特征点像素坐标
                    edge->setInformation(Mat22::Identity());  // 信息矩阵设为单位矩阵
                    
                    // 设置鲁棒核函数（Huber核函数），减少异常值影响
                    auto rk = new g2o::RobustKernelHuber();
                    rk->setDelta(chi2_th);  // 设置鲁棒核函数阈值
                    edge->setRobustKernel(rk);
                    
                    edges_and_features.insert({edge, feat});  // 存储边与特征的对应关系
                    optimizer.addEdge(edge);  // 将边添加到优化器
                    index++;  // 边ID递增
                }
            else 
                delete edge;  // 如果无法添加边，则删除
        }
    }

    // 6. 执行优化
    optimizer.initializeOptimization();  // 初始化优化
    optimizer.optimize(10);  // 执行10次优化迭代

    // 7. 异常值检测与处理（自适应阈值调整）
    int cnt_outlier = 0, cnt_inlier = 0;  // 异常值和正常值计数器
    int iteration = 0;  // 迭代计数器
    while (iteration < 5) {  // 最多迭代5次调整阈值
        cnt_outlier = 0;
        cnt_inlier = 0;
        
        // 统计当前阈值下的正常值和异常值数量
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {  // 卡方值超过阈值视为异常
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        
        // 计算正常值比例
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {  // 如果正常值比例超过50%，停止调整
            break;
        } else {  // 否则，增大阈值（放宽异常值判断标准）
            chi2_th *= 2;
            iteration++;
        }
    }

    // 8. 标记异常观测
    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {  // 卡方值超过阈值
            ef.second->is_outlier_ = true;  // 标记特征点为异常
            // 从路标点的观测中移除该异常观测
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;  // 否则标记为正常
        }
    }

    // 输出异常值和正常值数量
    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // 9. 更新优化后的变量
    // 更新关键帧位姿
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    // 更新路标点位置
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}
}  // namespace myslam