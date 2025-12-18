#include "myslam/camera.h"

namespace myslam {

Camera::Camera() {
}

// 将世界坐标系下的3D点转换到相机坐标系
// 参数：
//   p_w: 世界坐标系下的3D点（齐次坐标或普通3D坐标）
//   T_c_w: 从世界坐标系到相机坐标系的变换矩阵
// 返回值：相机坐标系下的3D点
Vec3 Camera::world2camera(const Vec3 &p_w, const SE3 &T_c_w) {
    // 变换顺序：先将点从世界坐标变换到双目相机参考坐标系（通过T_c_w），
    // 然后变换到当前相机坐标系（通过pose_）
    // 注意：这里假设pose_表示从双目参考坐标系到当前相机坐标系的变换
    return pose_ * T_c_w * p_w;
}

// 将相机坐标系下的3D点转换到世界坐标系
// 参数：
//   p_c: 相机坐标系下的3D点
//   T_c_w: 从世界坐标系到相机坐标系的变换矩阵
// 返回值：世界坐标系下的3D点
Vec3 Camera::camera2world(const Vec3 &p_c, const SE3 &T_c_w) {
    // 变换顺序：先将点从当前相机坐标系变换到双目参考坐标系（通过pose_inv_），
    // 然后变换到世界坐标系（通过T_c_w.inverse()）
    // 这里使用了预计算的pose_逆矩阵以提高效率
    return T_c_w.inverse() * pose_inv_ * p_c;
}

// 将相机坐标系下的3D点投影到像素坐标系
// 参数：
//   p_c: 相机坐标系下的3D点，假设p_c(2,0)是深度/Z值
// 返回值：像素坐标系下的2D点(u,v)
Vec2 Camera::camera2pixel(const Vec3 &p_c) {
    // 使用针孔相机模型公式：
    // u = fx * (X/Z) + cx
    // v = fy * (Y/Z) + cy
    // 其中p_c(0,0)=X, p_c(1,0)=Y, p_c(2,0)=Z
    return Vec2(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,  // 计算像素横坐标u
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_   // 计算像素纵坐标v
    );
}

// 将像素坐标和深度值反投影到相机坐标系
// 参数：
//   p_p: 像素坐标系下的2D点(u,v)
//   depth: 深度值（到相机光心的距离），默认值为1
// 返回值：相机坐标系下的3D点
Vec3 Camera::pixel2camera(const Vec2 &p_p, double depth) {
    // 使用针孔相机模型反投影公式：
    // X = (u - cx) * depth / fx
    // Y = (v - cy) * depth / fy
    // Z = depth
    return Vec3(
            (p_p(0, 0) - cx_) * depth / fx_,  // 计算相机坐标系下的X坐标
            (p_p(1, 0) - cy_) * depth / fy_,  // 计算相机坐标系下的Y坐标
            depth                              // Z坐标等于深度值
    );
}

// 将世界坐标系下的3D点投影到像素坐标系
// 参数：
//   p_w: 世界坐标系下的3D点
//   T_c_w: 从世界坐标系到相机坐标系的变换矩阵
// 返回值：像素坐标系下的2D点
Vec2 Camera::world2pixel(const Vec3 &p_w, const SE3 &T_c_w) {
    // 组合变换：先通过world2camera将点转换到相机坐标系，
    // 再通过camera2pixel投影到像素坐标系
    return camera2pixel(world2camera(p_w, T_c_w));
}

// 将像素坐标和深度值转换到世界坐标系
// 参数：
//   p_p: 像素坐标系下的2D点
//   T_c_w: 从世界坐标系到相机坐标系的变换矩阵
//   depth: 深度值
// 返回值：世界坐标系下的3D点
Vec3 Camera::pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth) {
    // 组合变换：先通过pixel2camera反投影到相机坐标系，
    // 再通过camera2world转换到世界坐标系
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

}
