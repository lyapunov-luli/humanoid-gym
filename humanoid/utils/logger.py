# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import os
import sys
import pandas as pd
from datetime import datetime
from openpyxl import load_workbook
from scipy.signal import find_peaks

class Logger:
    def __init__(self, dt):
        # self.state_log = defaultdict(list)
        # self.rew_log = defaultdict(list)
        self.dt = dt
        self.j_num = 6
        self.num_episodes = 0
        self.plot_process = None
        sys.path.append("/home/ldy/humanoid-gym/humanoid/state_log")
        self.base_save_addr = os.path.join(sys.path[-1], "play")
        # self.save_addr = '/home/aius/Reinforcement_Learning_Project/humanoid-gym/sim2real_pic/real_data'  # 图表保存地址
        # self.excel_path = '/sim2real_pic/2_13_推力训练/real_data.xlsx'

        # 创建带时间戳的子目录
        save_dir = create_timestamped_dir(self.base_save_addr)
        # 构建完整保存路径
        excel_filename = "real_data.xlsx"
        self.excel_path = os.path.join(save_dir, excel_filename)
        self.save_addr = os.path.join(save_dir, 'real_data')  # 图表保存地址

        self.joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
            'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
            'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        ]

        self.data={
            'target_pos': [],
            'dof_pos': [],
            'dof_vel': [],
            'dof_torque': [],
            'command': [],
            'base_lin_vel': [],
            'base_euler_xyz': [],
            'contact_forces_z': [],
            'foot_height':[],
        }

    def log_state(self, key, value):
        self.data[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def save_data(self):
        save_data_to_excel(self.data, self.excel_path)

    def save_plot(self, plot_name):
        plt.gcf().set_size_inches(16, 9)  # 获取当前figure并设置大小
        plt.savefig(self.save_addr + f'{plot_name}.png', dpi=100)  # 保存当前figure
        plt.close()  # 关闭当前figure释放内存

    def plot(self):
        # self.plot_target_pos(save_name = '_1_target_pos')
        self.plot_dof_pos(save_name = '_2_dof_pos')
        self.plot_dof_vel(save_name = '_3_dof_vel')
        self.plot_dof_torque(save_name = '_4_dof_torque')
        # self.plot_command(save_name = '_5_command_x')
        self.plot_base_lin_vel(save_name = '_6_base_lin_vel')
        self.plot_base_euler_ang(save_name = '_7_euler_ang')
        self.plot_contact_force(save_name = '_8_contact_forces')
        self.plot_foot_height(save_name = '_9_feet_height')

    def plot_dof_pos(self, save_name=None):
        pos = self.data['dof_pos']
        nested_out = np.array(transpose_list_of_lists(pos))
        plt.figure()
        plt.suptitle("Joint Positions")
        for idx, p in enumerate(nested_out):
            plt.subplot(2, self.j_num, idx + 1)
            plt.plot(p, label=f"Joint {idx + 1}")  # 添加 label

            # 添加图例
            plt.legend()

            # 添加标题和标签
            if idx < self.j_num:
                title, i = self.gen_joint_title(idx)
                plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel("Position")

            # 显示网格
            plt.grid(True)

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        joint_pos_data = {'Time Step': range(len(pos))}
        for idx, p in enumerate(nested_out):
            joint_pos_data[self.joint_names[idx]] = p
        save_data_to_excel({'Joint Positions': joint_pos_data}, self.excel_path)

    def plot_dof_vel(self, save_name=None):
        vel = self.data['dof_vel']
        nested_out = np.array(transpose_list_of_lists(vel))
        plt.figure()
        plt.suptitle("Joint Velocities")
        for idx, p in enumerate(nested_out):
            plt.subplot(2, self.j_num, idx + 1)
            plt.plot(p, label=f"Joint {idx + 1}")  # 添加 label

            # 添加图例
            plt.legend()

            # 添加标题和标签
            if idx < self.j_num:
                title, i = self.gen_joint_title(idx)
                plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel("Velocities")

            # 显示网格
            plt.grid(True)

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        joint_vel_data = {'Time Step': range(len(vel))}
        for idx, p in enumerate(nested_out):
            joint_vel_data[self.joint_names[idx]] = p
        save_data_to_excel({'Joint Velocities': joint_vel_data}, self.excel_path)

    def plot_dof_torque(self, save_name=None):
        """
        Plot joint torque data.
        """
        torque = self.data['dof_torque']
        nested_out = np.array(transpose_list_of_lists(torque))
        plt.figure()
        plt.suptitle("Joint Torque", fontsize=16)

        for idx, t in enumerate(nested_out):
            plt.subplot(2, self.j_num, idx + 1)
            plt.plot(t, label=f"Joint {idx + 1}")  # 绘制曲线并添加 label

            # 添加图例
            plt.legend()

            # 添加标题和标签
            if idx < self.j_num:
                title, i = self.gen_joint_title(idx)
                plt.title(title)
            plt.xlabel('Time Step')
            plt.ylabel("Torque")

            # 显示网格
            plt.grid(True)

        if save_name is not None:
            self.save_plot(save_name)

        # 保存数据到 Excel
        joint_torque_data = {'Time Step': range(len(torque))}
        for idx, t in enumerate(nested_out):
            joint_torque_data[self.joint_names[idx]] = t
        save_data_to_excel({'Joint Torques': joint_torque_data}, self.excel_path)
    
    def plot_base_lin_vel(self, save_name=None):

        base_lin_vel = self.data['base_lin_vel']
        base_lin_vel_tr = np.array(transpose_list_of_lists(base_lin_vel))  # 转换为 (3, N) 的数组

        # 颜色分配
        colors = {
            'x': 'blue',
            'y': 'green',
            'z': 'red'
        }

        # 调用模板函数
        self.plot_base_info_template(
            data=base_lin_vel_tr,
            title="Base Linear Velocities",
            ylabel="Velocity (m/s)",
            colors=colors,
            labels=['x', 'y', 'z'],
            save_name=save_name
        )

        # 保存数据到 Excel
        base_lin_vel_data = {
            'Time Step': range(len(base_lin_vel)),
            'X Linear Velocity': base_lin_vel_tr[0],
            'Y Linear Velocity': base_lin_vel_tr[1],
            'Z Linear Velocity': base_lin_vel_tr[2]
        }
        save_data_to_excel({'Base Linear Velocity': base_lin_vel_data}, self.excel_path)

    def plot_base_euler_ang(self, save_name=None):
        # 假设 self.data['eu_ang'] 是一个列表的列表，表示 roll, pitch, yaw 数据
        euler_ang = self.data['base_euler_xyz']
        euler_ang_tr = np.array(transpose_list_of_lists(euler_ang))  # 转换为 (3, N) 的数组

        # 颜色分配
        colors = {
            'roll': 'blue',
            'pitch': 'green',
            'yaw': 'red'
        }

        # 调用模板函数
        self.plot_base_info_template(
            data=euler_ang_tr,
            title="Base Euler Angles",
            ylabel="Angle (radians)",
            colors=colors,
            labels=['roll', 'pitch', 'yaw'],
            save_name=save_name
        )
        # 保存数据到 Excel
        base_ang_data = {
            'Time Step': range(len(euler_ang)),
            'Roll': euler_ang_tr[0],
            'Pitch': euler_ang_tr[1],
            'Yaw': euler_ang_tr[2]
        }
        save_data_to_excel({'Base Euler Angles': base_ang_data}, self.excel_path)
    
    def plot_contact_force(self, save_name=None):
        
        contact_force = self.data['contact_forces_z']
        contact_force_tr = np.array(transpose_list_of_lists(contact_force))  # 转换为 (2, N) 的数组

        # 颜色分配
        colors = {
            'left': 'blue',
            'right': 'green'
        }

        # 调用模板函数
        self.plot_base_info_template_2(
            data=contact_force_tr,
            title="Contact Forces",
            ylabel="newton",
            colors=colors,
            labels=['left', 'right'],
            save_name=save_name
        )
        # 保存数据到 Excel
        contact_force_data = {
            'Time Step': range(len(contact_force)),
            'left': contact_force_tr[0],
            'right': contact_force_tr[1],
        }
        save_data_to_excel({'Contact Forces': contact_force_data}, self.excel_path)

    def plot_foot_height(self,save_name=None):
        foot_height = self.data['foot_height']
        foot_height_tr = np.array(transpose_list_of_lists(foot_height))  # 转换为 (2, N) 的数组

        # 颜色分配
        colors = {
            'left': 'blue',
            'right': 'green'
        }

        # 调用模板函数
        self.plot_base_info_template_2(
            data=foot_height_tr,
            title="foot height",
            ylabel="m",
            colors=colors,
            labels=['left', 'right'],
            save_name=save_name
        )
        # 保存数据到 Excel
        foot_height_data = {
            'Time Step': range(len(foot_height)),
            'left': foot_height_tr[0],
            'right': foot_height_tr[1],
        }
        save_data_to_excel({'Foot Height': foot_height_data}, self.excel_path)

    def plot_base_info_template(self, data, title, ylabel, colors, labels, save_name=None):
        """
        通用绘图模板函数。

        参数:
            data: 形状为 (3, N) 的 NumPy 数组，表示要绘制的数据。
            title: 图像的主标题。
            ylabel: y 轴的标签。
            colors: 颜色字典，键为数据名称，值为颜色。
            labels: 数据名称列表，顺序与 data 的行对应。
            save_name: 保存图片的文件名。如果为 None，则不保存图片。
        """
        # 创建图像窗口，大小为 (10, 8)
        plt.figure(figsize=(10, 8))
        plt.suptitle(title)

        # 绘制子图
        for i in range(3):
            plt.subplot(3, 1, i + 1)  # 3 行 1 列，第 i+1 个子图
            plt.plot(data[i], label=labels[i], color=colors[labels[i]])
            plt.title(labels[i].capitalize())
            plt.xlabel('Time Step')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid()

        # 调整子图间距
        plt.tight_layout()

        # 如果提供了 save_name，则保存图片
        if save_name is not None:
            self.save_plot(save_name)

    def plot_base_info_template_2(self, data, title, ylabel, colors, labels, save_name=None):
        """
        通用绘图模板函数。

        参数:
            data: 形状为 (2, N) 的 NumPy 数组，表示要绘制的数据。
            title: 图像的主标题。
            ylabel: y 轴的标签。
            colors: 颜色字典，键为数据名称，值为颜色。
            labels: 数据名称列表，顺序与 data 的行对应。
            save_name: 保存图片的文件名。如果为 None，则不保存图片。
        """
        # 创建图像窗口，大小为 (10, 8)
        plt.figure(figsize=(10, 8))
        plt.suptitle(title)

        # 绘制子图
        for i in range(2):
            plt.subplot(2, 1, i + 1)  # 2 行 1 列，第 i+1 个子图
            plt.plot(data[i], label=labels[i], color=colors[labels[i]])
            plt.title(labels[i].capitalize())
            plt.xlabel('Time Step')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid()

        # 调整子图间距
        plt.tight_layout()

        # 如果提供了 save_name，则保存图片
        if save_name is not None:
            self.save_plot(save_name)
    
    def gen_joint_title(self, i):
        if i == 0:
            title = 'Hip_pitch'
        elif i == 1:
            title = 'Hip_roll'
        elif i == 2:
            title = 'Hip_yaw'
        elif i == 3:
            title = 'Knee'
        elif i == 4:
            title = 'Ankle_pitch'
        elif i == 5:
            title = 'Ankle_roll'
        return title, i + 1
    


def exponential_moving_average(velocity_data, alpha):
    """
    对速度数据应用指数移动平均滤波器。

    :param velocity_data: 二维 numpy 数组，每一行代表一个时间点的速度向量。
    :param alpha: 平滑因子，介于0和1之间。
    :return: 平滑后的速度数据。
    """
    # 初始化EMA数据数组，与输入数据形状相同
    smoothed_data = np.zeros_like(velocity_data)
    # 初始化第一个值
    smoothed_data[0] = velocity_data[0]

    # 遍历速度数据，并应用EMA滤波器
    for t in range(1, len(velocity_data)):
        smoothed_data[t] = alpha * velocity_data[t] + (1 - alpha) * smoothed_data[t - 1]

    return smoothed_data


def transpose_list_of_lists(nested_list):
    # 使用zip()函数和列表推导式来进行转置
    return [list(item) for item in zip(*nested_list)]


def plot_data_with_limits(upper_limit, lower_limit, data_list):
    # 绘制数据
    plt.plot(data_list)

    # 画出上限和下限的线
    plt.axhline(y=upper_limit, color='r', linestyle='--', label='Upper Limit')
    plt.axhline(y=lower_limit, color='g', linestyle='--', label='Lower Limit')

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Data Plot with Limits")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 显示网格
    plt.grid(True)


def plot_with_limits(data, lower_limit, upper_limit, title):
    """
    绘制带有上下限的图表。
    :param data: 数据列表。
    :param lower_limit: 下界。
    :param upper_limit: 上界。
    :param title: 图表的标题。
    """
    x = list(range(len(data)))
    plt.plot(x, data, label='Data')
    plt.fill_between(x, lower_limit, upper_limit, color='grey', alpha=0.3, label='Limits')
    plt.title(title)
    plt.legend()
    plt.grid(True)


def plot_with_tracking(data, track_value, title):
    """
    绘制带跟踪值的图表。
    :param data: 数据列表。
    :param track_value: 被跟踪的值。
    :param title: 图表的标题。
    """
    x = list(range(len(data)))
    plt.plot(x, data, label='Data')
    plt.axhline(y=track_value, color='r', linestyle='--', label=f'Track Value: {track_value}')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)


def plot_smooth_transition_areas(data, label_name):
    # 找到局部极大值即平台结束点
    peaks, _ = find_peaks(data)

    # 找到局部极小值即平台开始点（通过找到极大值点在-y上）
    troughs, _ = find_peaks(-data)

    # 确保首尾都被考虑进来
    if troughs[0] > peaks[0]:
        troughs = np.insert(troughs, 0, 0)
    if troughs[-1] < peaks[-1]:
        troughs = np.append(troughs, len(data) - 1)

    # 根据找到的极值索引进行绘图
    x_values = np.arange(len(data))
    # plt.figure(figsize=(14, 7))
    plt.plot(x_values, data, label=label_name)

    # 画出每个平台之间的高度差异
    for start, end in zip(troughs, peaks):
        height_diff = data[end] - data[start]
        # plt.plot([start, end], [y_values[start], y_values[end]], 'ro-')  # 'ro-'表示红色实心点和实线
        if height_diff > 0.05:
            mid_point = (start + end) / 2
            plt.text(mid_point, (data[start] + data[end]) / 2, f'{height_diff:.3f}',
                     ha='center', va='center', color='blue', fontsize=10,
                     bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.2'))

    plt.xlabel('Time Step')
    plt.ylabel('X Value')
    plt.title('Foothold observation')
    plt.legend()
    plt.grid(True)
def save_data_to_excel(data_dict, save_path):
    """增强版Excel保存函数"""
    try:
        # 处理目录创建
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # 文件存在性检查
        file_exists = os.path.exists(save_path)
        mode = 'a' if file_exists else 'w'

        # 使用上下文管理器处理Excel写入
        with pd.ExcelWriter(
                save_path,
                engine='openpyxl',
                mode=mode,
                if_sheet_exists='replace' if file_exists else None
        ) as writer:
            for sheet_name, data in data_dict.items():
                # 自动转换数据类型
                df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
                df.to_excel(writer, sheet_name=str(sheet_name)[:30], index=False)  # 截断超过30字符的表名
            print(f"数据成功保存至：{os.path.abspath(save_path)}")

    except PermissionError:
        print(f"错误：文件 {save_path} 被其他程序占用")
    except Exception as e:
        print(f"保存失败，错误类型：{type(e).__name__}，详细信息：{str(e)}")

def create_timestamped_dir(base_dir):
    """在指定基础目录下创建时间戳子目录"""
    # 生成时间戳格式：YYYY-MM-DD-HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # 拼接完整路径
    full_path = os.path.join(base_dir, timestamp)

    # 创建目录（自动创建父目录）
    os.makedirs(full_path, exist_ok=True)
    return full_path
def g1_whole_rearrange_array(arr):
    """
    调整数组的顺序，按照 [0:6], [12:16], [6:12], [16:20] 进行重组。

    参数:
    arr: 输入的 NumPy 数组，长度为 20。

    返回:
    rearranged_arr: 重组后的 NumPy 数组
    """
    # 检查输入数组的长度是否为 20
    if len(arr) != 20:
        raise ValueError("输入数组的长度必须为 20")

    # 按照 [0:6], [12:16], [6:12], [16:20] 的顺序重组
    rearranged_arr = np.concatenate([arr[0:6], arr[12:16], arr[6:12], arr[16:20]])

    return rearranged_arr