import torch
from torch import nn
import torch.nn.functional as F


# --- 1. 基础组件 ---
def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


# --- 2. 原始 AIRL 判别器 (用于标准模式) ---
class AIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones.unsqueeze(-1)) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator output: D = exp(f) / (exp(f) + pi)
        exp_f = torch.exp(self.f(states, dones, next_states))
        return (exp_f / (exp_f + torch.exp(log_pis.unsqueeze(-1))))

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return (torch.log(logits + 1e-3) - torch.log((1 - logits) + 1e-3))


# --- 3. 新增：共享社会势场网络 (SocialPotential) ---
class SocialPotentialNet(nn.Module):
    def __init__(self, state_shape, hidden_units=[64, 64], hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()
        layers = []
        # 输入通常是联合状态 (Joint State)
        units = state_shape
        for next_units in hidden_units:
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        layers.append(nn.Linear(units, 1))  # 输出标量势能 Phi
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return self.net(states)


# --- 4. 新增：支持势场分解的改进判别器 (SocialAIRLDiscrim) ---
class SocialAIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma, shared_phi_net,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.gamma = gamma
        self.shared_phi_net = shared_phi_net  # 外部传入的共享网络实例

        # 1. 私有偏好网络 (epsilon^i)
        self.private_g = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )

        # 2. 势能函数 V (用于 Advantage 估计)
        self.h = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        # 3. 可学习的权重 alpha (控制对社会势场的依赖程度)
        self.alpha = nn.Parameter(torch.ones(1) * 1.0)

    def get_reward(self, states):
        # R_total = alpha * Phi(s) + epsilon(s)
        phi = self.shared_phi_net(states)
        epsilon = self.private_g(states)
        return self.alpha * phi + epsilon, phi, epsilon

    def f(self, states, dones, next_states):
        # AIRL 的核心：f(s,s') = g(s) + gamma * h(s') - h(s)
        # 这里 g(s) 是复合奖励
        rs, _, _ = self.get_reward(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        # f 实际上近似于优势函数 A(s, a)
        return rs + self.gamma * (1 - dones.unsqueeze(-1)) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator output: D = exp(f) / (exp(f) + pi)
        exp_f = torch.exp(self.f(states, dones, next_states))
        return (exp_f / (exp_f + torch.exp(log_pis.unsqueeze(-1))))

    def calculate_reward(self, states, dones, log_pis, next_states):
        # 用于 PPO 更新的伪奖励
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return (torch.log(logits + 1e-3) - torch.log((1 - logits) + 1e-3))