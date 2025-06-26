import os
import wfdb
import numpy as np

#====之后可以改写成函数的====

# 原始数据目录
data_dir = r"E:\读研\科研相关\LXJ_learn\Reproduce\datasets\training\ptb-xl\g5"

# 保存片段的目录
save_dir = r"E:\pycharmprojects\HECG\data\ecg_test1_200"
os.makedirs(save_dir, exist_ok=True)

if os.path.exists(save_dir):
    print(f"目录已创建或已存在: {save_dir}")
else:
    print(f"目录创建失败: {save_dir}")

segment_count = 117950  # 用于命名输出文件

# 读取前100个样本
for i in range(4719, 5000):
    #将变量 i 的值格式化为一个宽度为 5 个字符的字符串，如果 i 的值不足 5 位，则在前面补
    file_id = f"HR{i:05d}"  # HR04719 ~ HR04999
    mat_path = os.path.join(data_dir, file_id)


    record = wfdb.rdrecord(mat_path)
    signal = record.p_signal  # (5000, 12)

    if signal.shape != (5000, 12):
        print(f"跳过 {file_id}，shape 不对：{signal.shape}")
        continue

    # 切成25段 (200, 12)
    for j in range(25):
        segment = signal[j*200:(j+1)*200, :]
        filename = f"segment_{(segment_count+1):04d}.npy"
        filepath = os.path.join(save_dir, filename)
        np.save(filepath, segment)
        segment_count += 1


print(f"共保存了 {segment_count-117950} 个 (200,12) 的 .npy 文件到 {save_dir}")


#====只读单一片段====
# file_id = "HR11000"
# mat_path = os.path.join(data_dir, file_id)
#
# record = wfdb.rdrecord(mat_path)
# signal = record.p_signal  # (5000, 12)
# filename = "test5000.npy"
# filepath = os.path.join(save_dir, filename)
#
# #检查信号是否有效
# if signal is not None:
#     filename = f"test5000.npy"
#     filepath = os.path.join(save_dir, filename)
#     np.save(filepath, signal)
#     print("File saved successfully:", filepath)
# else:
#     print("Signal is None, skipping save.")