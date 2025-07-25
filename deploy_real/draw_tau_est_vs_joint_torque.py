import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = "/home/ldy/humanoid-gym/humanoid/sim2real/sim2real_pic/2025-03-24-16-39-07/real_data.xlsx"  #"C:\\Users\\16190\\Downloads\\2025-03-17-22-07-54_real_data.xlsx"  #"E:\\Project\\LearningHumanoidWalking-main-run_jump\\utils\\real_data.xlsx"
df_q = pd.read_excel(file_path, sheet_name='Joint Positions', index_col=0)
df_dq = pd.read_excel(file_path, sheet_name='Joint Velocities', index_col=0)
df_tau = pd.read_excel(file_path, sheet_name='Joint Torques', index_col=0)
df_target_q = pd.read_excel(file_path, sheet_name='Joint Target q', index_col=0)

# 定义关节顺序（必须与Excel列顺序完全一致）
joint_columns = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint','right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint','right_ankle_roll_joint'
]

# 定义PD参数（顺序与joint_columns完全一致）
# kps = [200, 200, 200, 300, 40, 40, 100, 100, 40, 100,
#        200, 200, 200, 300, 40, 40, 100, 100, 40, 100]
# kds = [2, 2, 2, 4, 2, 2, 2, 2, 1, 2,
#        2, 2, 2, 4, 2, 2, 2, 2, 1, 2]
kps = [100, 100, 100, 150, 40, 40,
       100, 100, 100, 150, 40, 40,]
kds = [2, 2, 2, 4, 2, 2,
       2, 2, 2, 4, 2, 2,]
tau_limit = [88, 139, 88, 139, 50, 50,
             88, 139, 88, 139, 50, 50,]

# 创建PD参数字典
kp_dict = {joint: val for joint, val in zip(joint_columns, kps)}
kd_dict = {joint: val for joint, val in zip(joint_columns, kds)}
tau_limit_dict = {joint: val for joint, val in zip(joint_columns, tau_limit)}

# 验证列顺序
assert list(df_q.columns) == joint_columns, "列顺序不匹配！"

# 初始化结果存储
calculated_tau = pd.DataFrame(index=df_q.index, columns=joint_columns)
absolute_error = pd.DataFrame(index=df_q.index, columns=joint_columns)
relative_error = pd.DataFrame(index=df_q.index, columns=joint_columns)


def cal_clamped_pos(q, dq, target_pos, kp, kd, tau_limit):
    """
    优化后的向量化PD力矩钳位函数
    :param q: 当前关节位置 [n_joints]
    :param dq: 当前关节速度 [n_joints]
    :param target_pos: 目标位置 [n_joints]
    :param kp: 比例增益 [n_joints]
    :param kd: 微分增益 [n_joints]
    :param tau_limit: 力矩限制 [n_joints]
    :return: 调整后的安全位置 [n_joints]
    """
    # 计算预测力矩
    predicted_tau = kp * (target_pos - q) + kd * (-dq)

    # 安全阈值计算
    safe_tau = 0.9 * tau_limit * np.sign(predicted_tau)

    # 计算允许的最大位置调整量
    max_adjust = (safe_tau - kd * (-dq)) / kp

    # 构建调整后的目标位置
    return np.where(
        np.abs(predicted_tau) <= 0.9 * tau_limit,
        target_pos,
        q + max_adjust
    )


# 执行PD计算
for joint in joint_columns:
    df_target_q[joint] = cal_clamped_pos(q=df_q[joint],
                                         dq=df_dq[joint],
                                         target_pos=df_target_q[joint],
                                         kp=kp_dict[joint],
                                         kd=kd_dict[joint],
                                         tau_limit=tau_limit_dict[joint])
    # 提取参数
    kp = kp_dict[joint]
    kd = kd_dict[joint]

    # 计算误差
    pos_error = df_target_q[joint] - df_q[joint]
    vel_error = 0 - df_dq[joint]

    # 计算理论tau
    calculated_tau[joint] = kp * pos_error + kd * vel_error

    # 计算误差
    absolute_error[joint] = np.abs(calculated_tau[joint] - df_tau[joint])
    relative_error[joint] = absolute_error[joint] / (df_tau[joint].abs() + 1e-6)  # 避免除零

# ++++ 新增部分开始：创建对比用DataFrame ++++
# 将实际tau和计算tau合并，列名添加后缀
actual_tau = df_tau.add_suffix('_Actual')
calculated_tau_suffixed = calculated_tau.add_suffix('_Calculated')

# 按关节名配对列
paired_columns = []
for joint in joint_columns:
    paired_columns.append(f"{joint}_Actual")
    paired_columns.append(f"{joint}_Calculated")

# 合并数据时保持成对顺序
combined_tau = pd.concat(
    [actual_tau, calculated_tau_suffixed],
    axis=1
).reindex(columns=paired_columns)

# 新增对比误差计算
comparison_df = pd.concat([
    actual_tau.stack().rename('Actual_Tau'),
    calculated_tau_suffixed.stack().rename('Calculated_Tau')
], axis=1)
comparison_df['Delta_Tau'] = comparison_df['Calculated_Tau'] - comparison_df['Actual_Tau']
# ++++ 新增部分结束 ++++

# 生成统计报告
error_stats = pd.DataFrame({
    'Max Error': absolute_error.max(),
    'Mean Error': absolute_error.mean(),
    'RMS Error': np.sqrt((absolute_error ** 2).mean()),
    'Error > 10Nm': (absolute_error > 10).sum()
})

# 添加列索引转Excel列名的工具函数
def xl_colname(col_idx):
    """将列索引转换为Excel列名（0-based）"""
    letters = []
    while col_idx >= 0:
        letters.append(chr(col_idx % 26 + ord('A')))
        col_idx = col_idx // 26 - 1
    return ''.join(reversed(letters))


# 保存结果
with pd.ExcelWriter('pd_analysis_results.xlsx', engine='xlsxwriter') as writer:  # 明确指定引擎
    calculated_tau.to_excel(writer, sheet_name='Calculated Tau')
    absolute_error.to_excel(writer, sheet_name='Absolute Error')
    relative_error.to_excel(writer, sheet_name='Relative Error')
    error_stats.to_excel(writer, sheet_name='Error Statistics')
    combined_tau.to_excel(writer, sheet_name='Torque Comparison', index=True)

    # ============== 新增代码开始 ==============
    # 获取workbook和worksheet对象
    workbook = writer.book
    worksheet = writer.sheets['Torque Comparison']

    # 设置自动筛选（跳过时间步列）
    first_data_col = 1  # 时间步列是第0列，数据从第1列开始
    last_data_col = len(paired_columns)  # 原列数需要+1因为包含时间步列
    worksheet.autofilter(0, first_data_col, 0, last_data_col)

    # 定义高亮格式
    format_highlight = workbook.add_format({'bg_color': '#FFFF00'})

    # 修正后的列索引计算（考虑时间步列）
    for pair_idx in range(len(joint_columns)):
        # 实际数据列位置：1,3,5...（第0列是时间步）
        actual_col = 1 + pair_idx * 2
        # 计算数据列位置：2,4,6...
        calc_col = actual_col + 1

        # 转换列索引为Excel列名
        actual_col_name = xl_colname(actual_col)
        calc_col_name = xl_colname(calc_col)

        # 设置条件格式
        worksheet.conditional_format(
            1, calc_col,  # 从第2行开始
            len(df_tau), calc_col,
            {
                'type': 'formula',
                'criteria': f'=ABS({actual_col_name}2 - {calc_col_name}2) > 10',
                'format': format_highlight
            }
        )

    # 保存对比数据
    comparison_df.reset_index().to_excel(
        writer,
        sheet_name='Comparison Data',
        index=False
    )

# 可视化关键关节
plot_joints = ['left_knee_joint', 'right_ankle_pitch_joint', 'left_hip_pitch_joint']

plt.figure(figsize=(15, 9))
for i, joint in enumerate(plot_joints, 1):
    plt.subplot(3, 1, i)
    plt.plot(df_tau[joint], label='Actual Tau')
    plt.plot(calculated_tau[joint], '--', label='Calculated Tau')
    # ++++ 新增误差曲线 ++++
    plt.plot(np.abs(calculated_tau[joint] - df_tau[joint]),
             alpha=0.5, label='Absolute Error')
    plt.title(f"{joint} Torque Comparison (KP={kp_dict[joint]}, KD={kd_dict[joint]})")
    plt.ylabel('Torque (Nm)')
    plt.legend()
plt.tight_layout()
plt.savefig('tau_comparison.png')
plt.show()

print("分析完成！结果已保存至 pd_analysis_results.xlsx 和 tau_comparison.png")