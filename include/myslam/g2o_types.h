//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_G2O_TYPES_H
#define MYSLAM_G2O_TYPES_H

#include "myslam/common_include.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace myslam {
// 位姿顶点类，表示相机的位置和姿态
class VertexPose : public g2o::BaseVertex<6, SE3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 确保Eigen类型的内存对齐

    // 重置顶点估计值为初始值（单位变换矩阵）
    virtual void setToOriginImpl() override { _estimate = SE3(); }

    // 更新顶点估计值的实现，使用李代数更新
    virtual void oplusImpl(const double *update) override {
        Vec6 update_eigen;  // 6维向量，表示se(3)李代数
        update_eigen << update[0], update[1], update[2], update[3], update[4],
            update[5];
        // 使用指数映射将李代数转换为SE3，然后左乘到当前估计值上
        _estimate = SE3::exp(update_eigen) * _estimate;
    }

    // 读入顶点数据（这里没有实现，直接返回true）
    virtual bool read(std::istream &in) override { return true; }

    // 写出顶点数据（这里没有实现，直接返回true）
    virtual bool write(std::ostream &out) const override { return true; }
};

// 路标顶点类，表示3D空间点
class VertexXYZ : public g2o::BaseVertex<3, Vec3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 确保Eigen类型的内存对齐
    
    // 重置顶点估计值为零向量
    virtual void setToOriginImpl() override { _estimate = Vec3::Zero(); }

    // 更新顶点估计值的实现，直接加法更新
    virtual void oplusImpl(const double *update) override {
        _estimate[0] += update[0];  // x坐标增加
        _estimate[1] += update[1];  // y坐标增加
        _estimate[2] += update[2];  // z坐标增加
    }

    // 读入顶点数据（这里没有实现，直接返回true）
    virtual bool read(std::istream &in) override { return true; }

    // 写出顶点数据（这里没有实现，直接返回true）
    virtual bool write(std::ostream &out) const override { return true; }
};

// 仅估计位姿的一元边，用于仅位姿优化（Pose-Only BA）
class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 确保Eigen类型的内存对齐

    // 构造函数：传入3D点坐标和相机内参矩阵
    EdgeProjectionPoseOnly(const Vec3 &pos, const Mat33 &K)
        : _pos3d(pos), _K(K) {}

    // 计算边的误差
    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        SE3 T = v->estimate();  // 获取相机位姿估计值
        Vec3 pos_pixel = _K * (T * _pos3d);  // 将3D点转换到相机坐标系，再投影到像素平面
        pos_pixel /= pos_pixel[2];  // 归一化（除以深度z）
        // 误差 = 测量值（实际观测到的像素坐标） - 预测值
        _error = _measurement - pos_pixel.head<2>();
    }

    // 计算雅可比矩阵（关于位姿顶点的导数）
    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        SE3 T = v->estimate();  // 获取当前位姿估计
        Vec3 pos_cam = T * _pos3d;  // 3D点在相机坐标系下的坐标
        double fx = _K(0, 0);  // 相机内参fx
        double fy = _K(1, 1);  // 相机内参fy
        double X = pos_cam[0];  // 相机坐标系下X坐标
        double Y = pos_cam[1];  // 相机坐标系下Y坐标
        double Z = pos_cam[2];  // 相机坐标系下Z坐标（深度）
        double Zinv = 1.0 / (Z + 1e-18);  // 1/Z，避免除零
        double Zinv2 = Zinv * Zinv;  // 1/Z^2
        // 雅可比矩阵计算（2x6矩阵，重投影误差对李代数的导数）
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;
    }

    // 读入边数据（这里没有实现，直接返回true）
    virtual bool read(std::istream &in) override { return true; }

    // 写出边数据（这里没有实现，直接返回true）
    virtual bool write(std::ostream &out) const override { return true; }

   private:
    Vec3 _pos3d;   // 3D空间点坐标（世界坐标系）
    Mat33 _K;      // 相机内参矩阵
};

// 同时优化位姿和路标的二元边，用于完整BA
class EdgeProjection
    : public g2o::BaseBinaryEdge<2, Vec2, VertexPose, VertexXYZ> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 确保Eigen类型的内存对齐

    // 构造函数：传入相机内参和外参（相机到车体的变换）
    EdgeProjection(const Mat33 &K, const SE3 &cam_ext) : _K(K) {
        _cam_ext = cam_ext;
    }

    // 计算边的误差
    virtual void computeError() override {
        const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);  // 位姿顶点
        const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);    // 路标顶点
        SE3 T = v0->estimate();  // 车体位姿（车体到世界）
        // 完整的投影过程：世界点 -> 车体坐标系 -> 相机坐标系 -> 像素坐标系
        Vec3 pos_pixel = _K * (_cam_ext * (T * v1->estimate()));
        pos_pixel /= pos_pixel[2];  // 归一化（除以深度z）
        // 误差 = 测量值 - 预测值
        _error = _measurement - pos_pixel.head<2>();
    }

    // 计算雅可比矩阵（对位姿和路标的导数）
    virtual void linearizeOplus() override {
        const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);  // 位姿顶点
        const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);    // 路标顶点
        SE3 T = v0->estimate();  // 车体位姿
        Vec3 pw = v1->estimate();  // 世界坐标系下的3D点
        Vec3 pos_cam = _cam_ext * T * pw;  // 相机坐标系下的点
        double fx = _K(0, 0);  // 相机内参fx
        double fy = _K(1, 1);  // 相机内参fy
        double X = pos_cam[0];  // 相机坐标系下X坐标
        double Y = pos_cam[1];  // 相机坐标系下Y坐标
        double Z = pos_cam[2];  // 相机坐标系下Z坐标（深度）
        double Zinv = 1.0 / (Z + 1e-18);  // 1/Z，避免除零
        double Zinv2 = Zinv * Zinv;  // 1/Z^2
        // 雅可比矩阵关于位姿的部分（2x6）
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;

        // 雅可比矩阵关于路标点的部分（2x3）
        // 通过链式法则计算：∂e/∂pw = (∂e/∂pc) * (∂pc/∂pw)
        _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) *
                           _cam_ext.rotationMatrix() * T.rotationMatrix();
    }

    // 读入边数据（这里没有实现，直接返回true）
    virtual bool read(std::istream &in) override { return true; }

    // 写出边数据（这里没有实现，直接返回true）
    virtual bool write(std::ostream &out) const override { return true; }

   private:
    Mat33 _K;       // 相机内参矩阵
    SE3 _cam_ext;   // 相机外参（相机到车体的变换）
};

}  // namespace myslam

#endif  // MYSLAM_G2O_TYPES_H
