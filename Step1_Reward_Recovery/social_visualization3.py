import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import matplotlib.colors as mcolors
import os
import pickle
import pandas as pd
import seaborn as sns


# ==========================================
# 1. 核心模型 (已修正几何形状)
# ==========================================
class SocialPotentialNet(nn.Module):
    def __init__(self, A, B, rx=6.0, ry=2.0):
        super().__init__()
        self.A = A
        self.B = B
        self.rx = rx
        self.ry = ry
        self.lat_weight = self.rx / self.ry

        # 归一化参数 (适配 NGSIMEnv)
        self.norm_x = 25.0
        self.norm_y = 500.0

    def helbing_potential(self, dx_meters, dy_meters):
        """用于绘图 (输入物理距离)"""
        dx = torch.tensor(dx_meters)
        dy = torch.tensor(dy_meters)

        # 【关键修正】权重乘在 dx 上，或者 dy 不乘权重
        # 物理意义：横向 1 米的危险度 = 纵向 3 米
        # 这样势场才是纵向延伸的“长椭圆”，解决“近视”问题
        d_eff = torch.sqrt((dx * self.lat_weight) ** 2 + dy ** 2 + 1e-6)

        exponent = (self.rx - d_eff) / self.B
        exponent = torch.clamp(exponent, min=-10.0, max=10.0)
        U = self.A * torch.exp(exponent)
        return U.numpy()

    def forward_batch(self, states):
        """用于统计验证 (输入归一化状态)"""
        if torch.is_tensor(states):
            states = states.detach().cpu().numpy()

        # 提取相对距离 (Batch, 16) -> index 4, 5
        dx_norm = states[:, 4]
        dy_norm = states[:, 5]

        dx = np.abs(dx_norm * self.norm_x)
        dy = np.abs(dy_norm * self.norm_y)

        # Numpy 版计算
        d_eff = np.sqrt((dx * self.lat_weight) ** 2 + dy ** 2 + 1e-6)
        exponent = (self.rx - d_eff) / self.B
        exponent = np.clip(exponent, -10.0, 10.0)
        U = self.A * np.exp(exponent)
        return U


# ==========================================
# 2. 数据加载函数
# ==========================================
def load_real_data():
    """加载真实专家数据用于统计验证"""
    paths = [
        'expert_data.pkl',
        'Step1_Reward_Recovery/expert_data.pkl',
        '/mnt/h/A_CODE/Diff-LC/Step1_Reward_Recovery/expert_data.pkl'
    ]

    states = None
    for p in paths:
        if os.path.exists(p):
            print(f"Loading real data from: {p}")
            with open(p, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'state' in data:
                states = data['state']
                if torch.is_tensor(states): states = states.numpy()
            break

    if states is None:
        print("Warning: Real data not found. Creating dummy data for demo.")
        # 造一些假数据防止报错 (仅用于演示流程)
        states = np.zeros((1000, 16))
        states[:, 5] = np.random.normal(30 / 500.0, 5 / 500.0, 1000)  # 30m 跟车

    return states


# ==========================================
# 3. 绘图辅助函数 (保持原风格)
# ==========================================
def draw_lanes(ax, y_min, y_max):
    lane_width = 3.75
    boundaries = [-lane_width * 1.5, -lane_width * 0.5, lane_width * 0.5, lane_width * 1.5]
    for x_pos in boundaries:
        ax.vlines(x_pos, y_min, y_max, colors='gray', linestyles=(0, (5, 10)), linewidths=0.8, alpha=0.4, zorder=1)


def plot_single_interaction(A, B, dist_label, dy, param_title, file_name, vmax=None, stats_text=""):
    fig, ax = plt.subplots(figsize=(6, 8))

    x = np.linspace(-12, 12, 300)
    y = np.linspace(-10, 35, 300)  # 覆盖纵向 35m，避免色块留白
    X, Y = np.meshgrid(x, y)

    model = SocialPotentialNet(A=A, B=B)

    # 计算双车叠加
    pos1 = (0, 0)
    pos2 = (3.75, dy)
    Z_total = model.helbing_potential(X - pos1[0], Y - pos1[1]) + \
              model.helbing_potential(X - pos2[0], Y - pos2[1])

    # 绘图
    cmap = plt.cm.RdYlBu_r
    contour = ax.contourf(X, Y, Z_total, levels=100, cmap=cmap, vmin=0, vmax=vmax, extend='max', zorder=0)

    # 等势线
    levels = [0.1, 0.5, 1.0, 2.0]
    levels = [l for l in levels if l < vmax]
    if len(levels) > 0:
        lines = ax.contour(X, Y, Z_total, levels=levels, colors='k', linewidths=0.6, alpha=0.5, zorder=2)
        ax.clabel(lines, inline=True, fontsize=8, fmt='%.1f', colors='k')

    # 车辆
    ax.scatter([pos1[0]], [pos1[1]], c='white', s=200, marker='^', edgecolors='black', linewidth=1.5, label='Ego',
               zorder=10)
    ax.scatter([pos2[0]], [pos2[1]], c='cyan', s=200, marker='^', edgecolors='black', linewidth=1.5, label='Target',
               zorder=10)

    draw_lanes(ax, -10, 35)

    # 标题 (包含验证统计)
    full_title = f"{param_title}\n{dist_label}\n{stats_text}"
    ax.set_title(full_title, fontsize=12, pad=12, fontweight='bold')

    ax.set_xlabel('Lateral (m)', fontsize=12)
    ax.set_ylabel('Longitudinal (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 35)
    ax.grid(False)

    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Potential U', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    # plt.savefig(f"{file_name}.png", dpi=300)
    plt.show()


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    # 1. 加载真实数据进行验证
    real_states = load_real_data()

    # 构造危险数据 (5m 跟车)
    danger_states = real_states.copy()
    danger_states[:, 4] = 0.5 / 25.0
    danger_states[:, 5] = 5.0 / 500.0

    scenarios = [
        {"A": 0.5, "B": 2.0, "title": "A=0.5 B=2.0 (Medium)", "suffix": "a0.5"},
        {"A": 1.0, "B": 2.0, "title": "A=1.0 B=2.0 (Strong)", "suffix": "a1.0"},
        {"A": 2.0, "B": 2.0, "title": "A=2.0 B=2.0 (Extreme)", "suffix": "a2.0"},
        {"A": 0.5, "B": 4.0, "title": "A=0.5 B=4.0 (Medium)", "suffix": "a0.5"},
        {"A": 1.0, "B": 4.0, "title": "A=1.0 B=4.0 (Strong)", "suffix": "a1.0"},
        {"A": 2.0, "B": 4.0, "title": "A=2.0 B=4.0 (Extreme)", "suffix": "a2.0"},
        {"A": 0.5, "B": 8.0, "title": "A=0.5 B=8.0 (Medium)", "suffix": "a0.5"},
        {"A": 1.0, "B": 8.0, "title": "A=1.0 B=8.0 (Strong)", "suffix": "a1.0"},
        {"A": 2.0, "B": 8.0, "title": "A=2.0 B=8.0 (Extreme)", "suffix": "a2.0"},
        {"A": 0.5, "B": 10.0, "title": "A=0.5 B=10.0 (Medium)", "suffix": "a0.5"},
        {"A": 1.0, "B": 10.0, "title": "A=1.0 B=10.0 (Strong)", "suffix": "a1.0"},
        {"A": 2.0, "B": 10.0, "title": "A=2.0 B=10.0 (Extreme)", "suffix": "a2.0"},
        {"A": 0.5, "B": 12.0, "title": "A=0.5 B=12.0 (Medium)", "suffix": "a0.5"},
        {"A": 1.0, "B": 12.0, "title": "A=1.0 B=12.0 (Strong)", "suffix": "a1.0"},
        {"A": 2.0, "B": 12.0, "title": "A=2.0 B=12.0 (Extreme)", "suffix": "a2.0"},
    ]

    # 我们只画 Close (8m) 和 Medium (15m) 的图来对比
    distances = [
        {"label": "Close (8m)", "dy": 8.0, "suffix": "close"},
        {"label": "Medium (15m)", "dy": 15.0, "suffix": "medium"},
        {"label": "Far (30m)", "dy": 30.0, "suffix": "far"}
    ]

    for sc in scenarios:
        A = sc["A"]
        B = sc["B"]
        title = sc["title"]

        # --- 统计验证 ---
        model = SocialPotentialNet(A, B)
        u_expert = model.forward_batch(real_states)
        u_danger = model.forward_batch(danger_states)

        mean_exp = np.mean(u_expert)
        mean_dng = np.mean(u_danger)
        snr = mean_dng / (mean_exp + 1e-6)

        stats_str = f"[Verify] Safe: {mean_exp:.2f} | Danger(5m): {mean_dng:.2f} | SNR: {snr:.1f}"
        print(f"Scenario {title}: {stats_str}")

        # 确定色标 (统一对比)
        vmax = 2.0 if A <= 1.0 else 4.0

        for dist in distances:
            file_name = f"verify_{sc['suffix']}_{dist['suffix']}"
            plot_single_interaction(
                A=A, B=B,
                dist_label=dist["label"], dy=dist["dy"],
                param_title=title,
                file_name=file_name,
                vmax=vmax,
                stats_text=stats_str
            )
