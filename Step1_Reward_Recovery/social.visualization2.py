import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import pickle
import os
import pandas as pd

# ==========================================
# 1. 核心模型 (与您提供的代码保持一致)
# ==========================================
class SocialPotentialNet(nn.Module):
    def __init__(self, A, B, rx=6.0, ry=2.0):
        super().__init__()
        self.A = A
        self.B = B
        self.rx = rx
        self.ry = ry
        self.lat_weight = self.rx / self.ry

        # 归一化参数 (必须与 Step1/NGSIM_env/envs/env.py 保持一致)
        # index 0/4 -> [0, 25] (横向)
        # index 1/5 -> [0, 500] (纵向)
        self.norm_x = 25.0   # Lateral Width
        self.norm_y = 500.0  # Longitudinal Length

    def forward(self, states):
        # 自动处理 Tensor 或 Numpy
        if torch.is_tensor(states):
            states = states.detach().cpu().numpy()

        # 提取相对距离 (Batch, State_Dim)
        # env.py 定义: index 4=fv_lat_norm, index 5=fv_long_norm
        dx_norm = states[:, 4]
        dy_norm = states[:, 5]

        # 还原物理距离 (米)
        dx = np.abs(dx_norm * self.norm_x)
        dy = np.abs(dy_norm * self.norm_y)

        # Helbing 公式计算
        d_eff = np.sqrt(dx ** 2 + (dy * self.lat_weight) ** 2 + 1e-6)
        exponent = (self.rx - d_eff) / self.B
        exponent = np.clip(exponent, -10.0, 10.0)

        U = self.A * np.exp(exponent)

        # 这里的 Clamp 只是为了防止绘图时极大值破坏分布，训练代码里也有
        U = np.clip(U, 0.0, 10.0)
        return U

# ==========================================
# 2. 加载与处理数据
# ==========================================
def load_real_expert_data(filepath):
    """加载 Step1 生成的 expert_data.pkl"""
    search_paths = [
        filepath,
        '/mnt/h/A_CODE/Diff-LC/Step1_Reward_Recovery/data/expert_data.pkl',
        '/mnt/h/A_CODE/Diff-LC/Step1_Reward_Recovery/data/sample_data_with_speed.pkl',
        '/mnt/h/A_CODE/Diff-LC/Step1_Reward_Recovery/data/sample_data.pkl',
    ]
    target_path = None
    for p in search_paths:
        if p and os.path.exists(p):
            target_path = p
            break
    if target_path is None:
        raise FileNotFoundError(f"未找到可用的数据文件，请先生成 expert_data.pkl 或 sample_data_with_speed.pkl。")

    print(f"正在加载专家/轨迹数据: {target_path} ...")
    with open(target_path, 'rb') as f:
        data = pickle.load(f)

    # 情况 1：标准 expert_data，包含 'state'
    if isinstance(data, dict) and 'state' in data:
        states = data['state']
        if torch.is_tensor(states):
            states = states.numpy()
        print(f"成功加载 {len(states)} 条专家轨迹数据。")
        return states

    # 情况 2：sample_data_with_speed / sample_data 风格
    def scene_to_states(scene_df: pd.DataFrame):
        """将单个场景的轨迹表转为最简 16 维状态（仅 dx/dy 归一化，其余置零）"""
        # 优先使用列命名约定：NLV -> (y, x), FV -> (y_y, x_y)
        if {'y', 'x', 'y_y', 'x_y'}.issubset(scene_df.columns):
            dx = (scene_df['x_y'] - scene_df['x']).abs() / 25.0
            dy = (scene_df['y_y'] - scene_df['y']).abs() / 500.0
        elif {'y', 'x', 'y_x', 'x_x'}.issubset(scene_df.columns):
            dx = (scene_df['x_x'] - scene_df['x']).abs() / 25.0
            dy = (scene_df['y_x'] - scene_df['y']).abs() / 500.0
        else:
            raise ValueError("无法从轨迹表推断前车/本车位置列，请检查数据列命名。")
        n = len(scene_df)
        states = np.zeros((n, 16), dtype=np.float32)
        states[:, 4] = dx.values
        states[:, 5] = dy.values
        return states

    # dict 形式：键包含 lcv/fv/nlv/olv 列表
    if isinstance(data, dict) and all(k in data for k in ('lcv', 'fv', 'nlv')):
        states_list = []
        for lcv, fv, nlv in zip(data['lcv'], data['fv'], data['nlv']):
            # 构建 DataFrame 便于列运算
            scene_df = pd.DataFrame({
                'y':   nlv['y'], 'x':   nlv['x'],
                'y_y': fv['y'],  'x_y': fv['x'],
            })
            states_list.append(scene_to_states(scene_df))
        states = np.concatenate(states_list, axis=0)
        print(f"从 sample_data 风格构造了 {len(states)} 条状态。")
        return states

    # list 形式：每个元素是 DataFrame 场景
    if isinstance(data, list):
        states_list = []
        for scene in data:
            if isinstance(scene, pd.DataFrame):
                states_list.append(scene_to_states(scene))
            else:
                raise ValueError("列表元素不是 DataFrame，无法解析。")
        states = np.concatenate(states_list, axis=0)
        print(f"从列表场景构造了 {len(states)} 条状态。")
        return states

    raise ValueError("数据格式未识别，无法加载状态。")

# ==========================================
# 3. 核心验证逻辑
# ==========================================
def verify_parameters():
    # 1. 准备数据
    try:
        states_expert = load_real_expert_data('expert_data.pkl')
    except Exception as e:
        print(e)
        return

    # 2. 构造“危险对照组” (Synthetic Danger)
    # 我们复制一份专家数据，但强制把所有车辆的“纵向距离”改为 5 米，“横向”改为 0 米
    # 模拟紧贴前车的情况
    states_danger = states_expert.copy()
    # states_danger[:, 4] = 0.5 / 25.0   # 横向距离 0m
    # states_danger[:, 5] = 5.0 / 500.0  # 纵向距离 5m
    lat_target = 0.5 / 25.0
    lon_target = 5.0 / 500.0
    lat_noise = np.random.normal(0, 0.05 / 25.0, size=len(states_danger))  # ±0.05m 级别抖动
    lon_noise = np.random.normal(0, 0.5 / 500.0, size=len(states_danger))  # ±0.5m 级别抖动
    states_danger[:, 4] = np.clip(lat_target + lat_noise, 0.0, None)
    states_danger[:, 5] = np.clip(lon_target + lon_noise, 0.0, None)


    # 3. 定义待测参数组 (Sensitivity Analysis)
    # 固定 A=1.0，扫描不同 B
    scenarios = [
        {"A": 1.0, "B": 4.0, "label": "B=4"},
        {"A": 1.0, "B": 6.0, "label": "B=6"},
        {"A": 1.0, "B": 8.0, "label": "B=8"},
        {"A": 1.0, "B": 12.0, "label": "B=12"},
    ]

    # 4. 绘图配置
    plt.figure(figsize=(16, 10))
    plt.subplots_adjust(hspace=0.4)

    for i, sc in enumerate(scenarios):
        A, B, label = sc['A'], sc['B'], sc['label']
        model = SocialPotentialNet(A, B)

        # 计算势能
        U_expert = model.forward(states_expert)
        U_danger = model.forward(states_danger)

        # 统计指标
        mean_exp = np.mean(U_expert)
        mean_dng = np.mean(U_danger)
        snr = mean_dng / (mean_exp + 1e-6) # 信噪比

        # 绘制子图
        ax = plt.subplot(2, 2, i+1)

        # 绘制专家分布 (绿色)
        sns.kdeplot(U_expert, fill=True, color='green', label='Expert (Safe)', ax=ax, clip=(0, 5))
        # 绘制危险分布 (红色)
        sns.kdeplot(U_danger, fill=True, color='red', label='Danger (5m)', ax=ax, clip=(0, 5))

        # 标注统计信息
        ax.set_title(f"{label}\nSignal-to-Noise Ratio: {snr:.1f}", fontweight='bold')
        ax.set_xlabel("Potential Energy U")
        if i % 2 == 0: ax.set_ylabel("Density")

        # 辅助线
        ax.axvline(x=mean_exp, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=mean_dng, color='red', linestyle='--', alpha=0.5)
        ax.text(mean_dng, ax.get_ylim()[1]*0.8, f" Signal: {mean_dng:.2f}", color='red')

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Parameter Verification on Real NGSIM Data (Fixed A=1.0, Varying B)", fontsize=16, y=0.98)
    plt.show()
    # plt.savefig('param_verification_ngsim.png') # 保存图片

if __name__ == '__main__':
    verify_parameters()
