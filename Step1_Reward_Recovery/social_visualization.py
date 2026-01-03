import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import matplotlib.colors as mcolors


# ==========================================
# 1. 核心模型 (保持不变)
# ==========================================
class SocialPotentialNet(nn.Module):
    def __init__(self, A, B, rx=6.0, ry=2.0):
        super().__init__()
        self.A = A
        self.B = B
        self.rx = rx
        self.ry = ry
        self.lat_weight = self.rx / self.ry

    def helbing_potential(self, dx_meters, dy_meters):
        dx = torch.tensor(dx_meters)
        dy = torch.tensor(dy_meters)
        d_eff = torch.sqrt(dx ** 2 + (dy * self.lat_weight) ** 2 + 1e-6)

        exponent = (self.rx - d_eff) / self.B
        exponent = torch.clamp(exponent, min=-10.0, max=10.0)

        U = self.A * torch.exp(exponent)
        return U.numpy()


# ==========================================
# 2. 绘图辅助函数
# ==========================================
def draw_lanes(ax, y_min, y_max):
    """绘制标准车道线"""
    lane_width = 3.75
    # 绘制边界: 左(-3.75), 中(0), 右(+3.75)
    boundaries = [-lane_width * 1.5, -lane_width * 0.5, lane_width * 0.5, lane_width * 1.5]
    for x_pos in boundaries:
        ax.vlines(x_pos, y_min, y_max, colors='gray', linestyles=(0, (5, 10)), linewidths=0.8, alpha=0.4, zorder=1)


def plot_single_interaction(A, B, dist_label, dy, param_title, file_name, vmax=None):
    """
    生成并显示/保存单张图片
    """
    # 设置画布: 竖向长图适应道路
    fig, ax = plt.subplots(figsize=(6, 8))

    # 网格生成
    x = np.linspace(-12, 12, 300)
    y = np.linspace(-30, 30, 300)
    X, Y = np.meshgrid(x, y)

    # 实例化模型
    model = SocialPotentialNet(A=A, B=B)

    # --- 计算双车势能叠加 ---
    pos1 = (0, 0)  # Ego
    pos2 = (3.75, dy)  # Target (右侧前方)

    Z1 = model.helbing_potential(X - pos1[0], Y - pos1[1])
    Z2 = model.helbing_potential(X - pos2[0], Y - pos2[1])
    Z_total = Z1 + Z2  # 线性叠加

    # --- 绘制热力图 ---
    # 使用 RdYlBu_r (蓝=安全 -> 黄 -> 红=危险)
    cmap = plt.cm.RdYlBu_r
    contour = ax.contourf(X, Y, Z_total, levels=100, cmap=cmap, vmin=0, vmax=vmax, extend='max', zorder=0)

    # --- 绘制等势线 ---
    levels = [0.1, 0.5, 1.0, 2.0]
    levels = [l for l in levels if l < vmax]  # 过滤过高的线

    if len(levels) > 0:
        lines = ax.contour(X, Y, Z_total, levels=levels, colors='k', linewidths=0.6, alpha=0.5, zorder=2)
        ax.clabel(lines, inline=True, fontsize=8, fmt='%.1f', colors='k')

    # --- 绘制车辆 ---
    ax.scatter([pos1[0]], [pos1[1]], c='white', s=200, marker='^', edgecolors='black', linewidth=1.5, label='Ego',
               zorder=10)
    ax.scatter([pos2[0]], [pos2[1]], c='cyan', s=200, marker='^', edgecolors='black', linewidth=1.5, label='Target',
               zorder=10)

    # --- 装饰 ---
    draw_lanes(ax, -30, 30)

    # 标题包含参数和距离信息
    full_title = f"{param_title}\nInteraction Distance: {dist_label}"
    ax.set_title(full_title, fontsize=14, pad=12, fontweight='bold')

    ax.set_xlabel('Lateral Distance (m)', fontsize=12)
    ax.set_ylabel('Longitudinal Distance (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 35)  # 聚焦前方
    ax.grid(False)

    # 添加色条
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Interaction Potential $U_{total}$', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()

    # 如果想保存图片，请取消下面这行的注释
    # plt.savefig(f"{file_name}.png", dpi=300, bbox_inches='tight')

    plt.show()


# ==========================================
# 3. 主程序：参数敏感性分析循环
# ==========================================
if __name__ == '__main__':
    # 定义 3 组参数配置
    scenarios = [
        {
            "A": 0.5, "B": 1.0,
            "title": "Scenario 1: Steep (A=0.5, B=1)",
            "suffix": "steep",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 0.5, "B": 2.0,
            "title": "Scenario 2: Smooth (A=0.5, B=2)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
     {
            "A": 1, "B": 3.0,
            "title": "Scenario 1: Steep (A=1, B=3)",
            "suffix": "steep",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 1, "B": 4.0,
            "title": "Scenario 2: Smooth (A=1, B=4)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 1, "B": 5.0,
            "title": "Scenario 2: Smooth (A=1, B=5)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 1, "B": 6.0,
            "title": "Scenario 2: Smooth (A=1, B=6)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 1, "B": 7.0,
            "title": "Scenario 2: Smooth (A=1, B=7)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 1, "B": 8.0,
            "title": "Scenario 2: Smooth (A=1, B=8)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 1, "B": 9.0,
            "title": "Scenario 2: Smooth (A=1, B=9)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
        {
            "A": 1, "B": 10.0,
            "title": "Scenario 2: Smooth (A=1, B=10)",
            "suffix": "smooth",
            "vmax_override": None  # 自动计算
        },
    ]

    # 定义 3 种交互距离
    distances = [
        {"label": "Close (8m)", "dy": 8.0, "suffix": "close"},
        {"label": "Medium (15m)", "dy": 15.0, "suffix": "medium"},
        {"label": "Far (25m)", "dy": 25.0, "suffix": "far"}
    ]

    # 双重循环：遍历参数 -> 遍历距离
    for sc in scenarios:
        A = sc["A"]
        B = sc["B"]
        title = sc["title"]

        # --- 步骤 1: 确定当前参数组的统一色标 (vmax) ---
        # 为了保证 近/中/远 三张图颜色可对比，必须使用相同的 vmax
        if sc["vmax_override"] is not None:
            vmax = sc["vmax_override"]
        else:
            # 如果没指定，就用 Close 场景的 99% 分位数自动计算
            # 这样能保证最亮的地方不会爆掉，也不会太暗
            x_temp = np.linspace(-12, 12, 100)
            y_temp = np.linspace(-30, 30, 100)
            X_temp, Y_temp = np.meshgrid(x_temp, y_temp)
            model_temp = SocialPotentialNet(A=A, B=B)
            Z_temp = model_temp.helbing_potential(X_temp, Y_temp) + \
                     model_temp.helbing_potential(X_temp - 3.75, Y_temp - 8.0)
            vmax = np.percentile(Z_temp, 99)
            if vmax > 5.0: vmax = 5.0  # 封顶

        print(f"Generating sequence for {title} (vmax={vmax:.2f})...")

        # --- 步骤 2: 一张一张生成图片 ---
        for dist in distances:
            file_name = f"param_{sc['suffix']}_dist_{dist['suffix']}"
            print(f"  -> Plotting {dist['label']}...")

            plot_single_interaction(
                A=A,
                B=B,
                dist_label=dist["label"],
                dy=dist["dy"],
                param_title=title,
                file_name=file_name,
                vmax=vmax
            )