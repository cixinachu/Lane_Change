import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import os
import pickle

# 引入环境 (确保你在 Step1_Reward_Recovery 目录下运行)
try:
    from NGSIM_env.envs.env import NGSIMEnv
    print("成功加载 NGSIMEnv 环境！")
except ImportError:
    print("错误：请确保脚本在 Step1_Reward_Recovery 目录下，并且该目录包含 NGSIM_env 文件夹。")
    exit()

# ==========================================
# 1. 定义待测势场模型 (与训练代码一致)
# ==========================================
class SocialPotentialNet(nn.Module):
    def __init__(self, A, B, rx=6.0, ry=2.0):
        super().__init__()
        self.A = A
        self.B = B
        self.rx = rx
        self.ry = ry
        self.lat_weight = self.rx / self.ry 
        
        # 这里的归一化参数必须与 env.py 中的一致！
        # env.py: index 0/4 -> [0, 25] (横向), index 1/5 -> [0, 500] (纵向)
        self.norm_x = 25.0   # Lateral Width
        self.norm_y = 500.0  # Longitudinal Length

    def forward(self, states):
        # 提取相对位置 (env.py 中定义 state 结构: [ego, fv_rel, nlv_rel, olv_rel])
        # fv_rel 在 index 4, 5
        # nlv_rel 在 index 8, 9 (如果有的话，这里只演示 FV)
        
        # 注意：env.py 里 fv_veh = lcv_veh - fv_veh
        # 这意味着 state 里存的是 "Ego - Other"，即相对距离
        
        if torch.is_tensor(states):
            states = states.cpu().numpy()
            
        # 提取 FV 的相对距离 (batch, state_dim) -> 取第 4, 5 列
        # 确保输入是 2D 数组
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
            
        dx_norm = states[:, 4] # 横向差异
        dy_norm = states[:, 5] # 纵向差异

        # 还原物理距离 (米)
        # 注意方向：势能只关心距离大小，不关心前后，所以取绝对值
        dx = np.abs(dx_norm * self.norm_x)
        dy = np.abs(dy_norm * self.norm_y)

        # Helbing 公式
        d_eff = np.sqrt(dx ** 2 + (dy * self.lat_weight) ** 2 + 1e-6)
        exponent = (self.rx - d_eff) / self.B
        exponent = np.clip(exponent, -10.0, 10.0)
        U = self.A * np.exp(exponent)
        
        # 截断保护
        U = np.clip(U, 0.0, 10.0)
        return U, dx, dy

# ==========================================
# 2. 核心实验流程
# ==========================================
def run_experiment():
    # --- A. 加载真实专家数据 ---
    print("\n正在生成/加载 NGSIM 专家数据...")
    env = NGSIMEnv(scene='us-101')
    
    # 如果已有 expert_data.pkl 直接读取，否则生成一点
    if os.path.exists('expert_data.pkl'):
        with open('expert_data.pkl', 'rb') as f:
            expert_data = pickle.load(f)
        states = expert_data['state'] # Tensor
        print(f"加载了 {len(states)} 条专家轨迹点。")
    else:
        print("未找到 expert_data.pkl，正在通过环境生成少量数据...")
        # 跑 10 个场景收集一点数据
        states_list = []
        for i in range(10):
            s = env.reset(scene_id=i)
            states_list.append(s)
            # 简单跑几步
            for _ in range(50):
                # 动作不重要，我们只看状态
                # 这里假设 expert 数据已经包含在 reset 里生成的轨迹中
                # 实际 env.step 需要 action，我们直接用环境内部的轨迹数据
                pass 
                # (为简化，这里直接建议使用已有的 pkl，如果没有建议先运行 main.py 生成)
        print("请先运行 main.py 生成 expert_data.pkl 以获得更准确的测试结果！")
        return

    # 转换为 Numpy
    if torch.is_tensor(states):
        states = states.numpy()

    # --- B. 对比实验 ---
    params = [
        (0.1, 8.0, "Old (A=0.1)"),
        (1.0, 8.0, "New (A=1.0)")
    ]

    plt.figure(figsize=(14, 6))

    for idx, (A, B, label) in enumerate(params):
        model = SocialPotentialNet(A, B)
        
        # 1. 计算专家状态下的势能 (False Positive Test)
        # 期望：大部分应该接近 0，因为专家是安全的
        U_expert, dx_exp, dy_exp = model.forward(states)
        
        # 2. 构造危险状态 (True Positive Test)
        # 强制把所有专家的纵向距离改为 5 米 (极其危险)
        states_danger = states.copy()
        # 归一化后的 5米 = 5 / 500 = 0.01
        states_danger[:, 5] = 5.0 / 500.0 
        U_danger, _, _ = model.forward(states_danger)

        # --- 打印统计数据 ---
        print(f"\n[{label}] 统计结果:")
        print(f"  专家平均势能 (干扰): {np.mean(U_expert):.4f} (期望越低越好)")
        print(f"  危险平均势能 (信号): {np.mean(U_danger):.4f} (期望适中，如 >0.5)")
        print(f"  信噪比 (Signal/Noise): {np.mean(U_danger) / (np.mean(U_expert)+1e-6):.2f}")

        # --- 绘图 ---
        plt.subplot(1, 2, idx+1)
        sns.kdeplot(U_expert, fill=True, color='green', label='Expert (Safe)')
        sns.kdeplot(U_danger, fill=True, color='red', label='Danger (5m)')
        plt.title(f"Potential Distribution: {label}\n(A={A}, B={B})")
        plt.xlabel("Potential Energy U")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 画一条线表示可能的判别器阈值
        plt.axvline(x=0.5, color='gray', linestyle='--', label='Threshold')

    plt.tight_layout()
    plt.savefig('experiment_prior_result.png')
    print("\n结果图已保存为 experiment_prior_result.png")
    plt.show()

if __name__ == '__main__':
    run_experiment()