/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/frame.h"

namespace myslam {

Frame::Frame(long id, double time_stamp, const SE3 &pose, const Mat &left, const Mat &right)
        : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {}

// 创建并返回一个新的 Frame 智能指针对象
Frame::Ptr Frame::CreateFrame() {
    // 静态变量，用于为每个新创建的帧生成唯一ID（生命周期贯穿整个程序）
    static long factory_id = 0;
    
    // 创建新的 Frame 对象并通过智能指针管理
    Frame::Ptr new_frame(new Frame);
    
    // 为当前帧分配唯一ID，然后递增工厂计数器
    new_frame->id_ = factory_id++;
    
    // 返回新创建的帧对象
    return new_frame;
}

// 将当前帧标记为关键帧并分配关键帧ID
void Frame::SetKeyFrame() {
    // 静态变量，用于为每个关键帧生成唯一ID（生命周期贯穿整个程序）
    static long keyframe_factory_id = 0;
    
    // 设置当前帧的关键帧标志为 true
    is_keyframe_ = true;
    
    // 为当前帧分配唯一的关键帧ID，然后递增关键帧计数器
    keyframe_id_ = keyframe_factory_id++;
}


}
