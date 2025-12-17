//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

// 定义命名空间 myslam，用于组织SLAM相关代码
namespace myslam {

/**
 * 使用SVD进行线性三角测量
 * @param poses     相机位姿列表（从世界坐标系到相机坐标系）
 * @param points    归一化平面上的点坐标（每个相机的观测）
 * @param pt_world  三角化得到的3D世界坐标点（输出参数）
 * @return true 如果三角测量成功（解质量好）
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
    // 创建系数矩阵A，大小为 2n × 4，其中n是相机数量
    // 每对匹配点提供2个约束方程
    MatXX A(2 * poses.size(), 4);
    // 创建右侧向量b，设置为零（齐次线性方程组Ax=0）
    VecX b(2 * poses.size());
    b.setZero();  // 将b向量所有元素设为0
    
    // 遍历每个相机位姿和对应的点
    for (size_t i = 0; i < poses.size(); ++i) {
        // 获取相机投影矩阵P = K[R|t]，这里假设K已归一化，所以直接使用3x4矩阵
        // matrix3x4()返回从SE3变换得到的3x4投影矩阵
        Mat34 m = poses[i].matrix3x4();
        
        // 构建约束方程：
        // 对于每个观测点(u,v,1)，有约束：x × (P * X) = 0
        // 其中x是归一化图像坐标，P是投影矩阵，X是3D点（齐次坐标）
        
        // 第一个约束方程：u * (P的第3行) - (P的第1行) = 0
        // block<1,4>表示取1行4列的子块，从(2*i, 0)位置开始
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        
        // 第二个约束方程：v * (P的第3行) - (P的第2行) = 0
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    
    // 使用SVD分解求解齐次线性方程组Ax=0
    // bdcSvd是Eigen库的分治SVD算法
    // ComputeThinU和ComputeThinV表示计算精简的U和V矩阵
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    // SVD分解后，最小奇异值对应的V的列就是解x
    // V的最后一列（col(3)）对应最小奇异值的右奇异向量
    // 需要归一化使得齐次坐标的最后一维为1：x = V(:,4)/V(4,4)
    // 然后取前三维作为3D坐标
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();
    
    // 检查解的质量：通过比较最后两个奇异值的比值
    // singularValues()[3]是最小奇异值，singularValues()[2]是次小奇异值
    // 如果最小奇异值远小于次小奇异值（比值<1e-2），说明解是唯一的且质量好
    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量不好，放弃
        return true;  // 注意：这里返回true表示函数成功执行，但实际是解质量不好
        // 可能函数设计是返回false表示失败，这里可能有逻辑问题
    }
    return false;  // 解质量好，返回false？这看起来逻辑反了
}

// 转换函数：将OpenCV的Point2f转换为Eigen的Vec2
inline Vec2 toVec2(const cv::Point2f p) { 
    return Vec2(p.x, p.y); 
}

}  // 命名空间结束  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
