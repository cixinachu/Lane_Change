import torch
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # [新增] 引入 TensorBoard
from ppo_model.PPO import PPO, RolloutBuffer
from NGSIM_env.envs.env import NGSIMEnv
import numpy as np
import warnings
import os
from datetime import datetime

# 从 model.airl 导入模型类
from airl import AIRLDiscrim, SocialAIRLDiscrim, SocialPotentialNet

warnings.filterwarnings('ignore')


def run():
    # ================= 配置区域 (Configuration) =================
    # [开关] True: 启用共享社会势场 (Improved); False: 原始 AIRL (Original)
    USE_SOCIAL_POTENTIAL = True

    # 正则项权重
    LAMBDA_SOC = 0.1  # 社会一致性权重 (Social Consistency)
    LAMBDA_EQ = 0.01  # 博弈均衡权重 (Equilibrium / Best-Response)

    # TensorBoard 日志目录设置
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = "SocialAIRL" if USE_SOCIAL_POTENTIAL else "OriginalAIRL"
    base_log_dir = os.path.join("runs", "log","reward_recovery_new")
    run_name = f"main2_{exp_name}_{time_str}"
    log_dir = os.path.join(base_log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)  # [新增] 初始化 Writer
    print(f">>> TensorBoard Logging to: {log_dir}")
    # ==========================================================

    env = NGSIMEnv(scene='us-101')
    env.generate_experts()

    buffer_exp = RolloutBuffer()
    buffer_exp.add_exp(path='expert_data.pkl')

    # Generator (PPO) 初始化
    model_lcv = PPO(state_dim=16, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                    has_continuous_action_space=True, action_std_init=0.2)
    model_fv = PPO(state_dim=16, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                   has_continuous_action_space=True, action_std_init=0.2)

    # ================= 判别器 (Discriminator) 初始化分支 =================
    social_phi_net = None

    if USE_SOCIAL_POTENTIAL:
        print(">>> Training Mode: Social Potential AIRL (Shared Framework)")
        # 1. 初始化共享社会势场网络
        social_phi_net = SocialPotentialNet(state_shape=16).to('cuda')

        # 2. 初始化改进版判别器
        disc_lcv = SocialAIRLDiscrim(state_shape=16, gamma=0.99, shared_phi_net=social_phi_net).to('cuda')
        disc_fv = SocialAIRLDiscrim(state_shape=16, gamma=0.99, shared_phi_net=social_phi_net).to('cuda')

        # 3. 优化器
        optim_disc = Adam([
            {'params': disc_lcv.parameters()},
            {'params': disc_fv.parameters()},
            {'params': social_phi_net.parameters()}
        ], lr=3e-4)

    else:
        print(">>> Training Mode: Original MA-AIRL (Independent Framework)")
        # 1. 初始化原始判别器
        disc_lcv = AIRLDiscrim(state_shape=16, gamma=0.99).to('cuda')
        disc_fv = AIRLDiscrim(state_shape=16, gamma=0.99).to('cuda')

        # 2. 优化器
        optim_disc = Adam([
            {'params': disc_lcv.parameters()},
            {'params': disc_fv.parameters()}
        ], lr=3e-4)
    # ===================================================================

    disc_criterion = nn.BCELoss()

    time_step = 0
    i_episode = 0
    scene_id = 0

    # 训练循环
    while i_episode <= 110000:
        state = env.reset(scene_id=scene_id)
        current_ep_reward = 0

        # --- 采样阶段 (Generator Sampling) ---
        while True:
            action_lcv = model_lcv.select_action(state) * 5
            action_fv = model_fv.select_action(state) * 5
            action = np.concatenate([action_lcv, action_fv])
            state, reward, done, _ = env.step(action)

            model_lcv.buffer.rewards.append(reward)
            model_lcv.buffer.is_terminals.append(done)
            model_lcv.buffer.next_states.append(torch.from_numpy(state.reshape(1, -1)).to('cuda'))

            model_fv.buffer.rewards.append(reward)
            model_fv.buffer.is_terminals.append(done)
            model_fv.buffer.next_states.append(torch.from_numpy(state.reshape(1, -1)).to('cuda'))

            time_step += 1
            current_ep_reward += reward

            if time_step % int(1e5) == 0:
                model_lcv.decay_action_std(0.05, 0.1)
                model_fv.decay_action_std(0.05, 0.1)

            if done:
                break

        # [新增] 记录 Generator 的环境奖励
        writer.add_scalar('Generator/Episode_Reward', current_ep_reward, i_episode)

        # --- AIRL 更新阶段 (Discriminator Update) ---
        if i_episode % 150 == 0:
            # 用于记录本轮更新的平均 Loss
            epoch_loss_total = []
            epoch_loss_base = []
            epoch_loss_soc = []
            epoch_loss_eq = []

            for _ in range(10):
                # 1. 采样
                states_lcv, actions_lcv, _, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.sample(64)
                states_fv, actions_fv, _, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.sample(64)
                states_exp, actions_exp, _, dones_exp, next_states_exp = buffer_exp.sample(64, if_exp=True)

                # 类型转换
                states_exp = states_exp.float()
                actions_exp = actions_exp.float() / 5
                next_states_exp = next_states_exp.float()
                states_lcv = states_lcv.float()
                states_fv = states_fv.float()
                next_states_lcv = next_states_lcv.float()
                next_states_fv = next_states_fv.float()
                dones_lcv = dones_lcv.int().to('cuda')
                dones_fv = dones_fv.int().to('cuda')
                dones_exp = dones_exp.int().to('cuda')

                with torch.no_grad():
                    log_pis_exp_lcv, _, _ = model_lcv.policy.evaluate(states_exp, actions_exp[:, :2])
                    log_pis_exp_fv, _, _ = model_fv.policy.evaluate(states_exp, actions_exp[:, 2:])

                # 2. 计算 Discriminator 输出
                prob_pi_lcv = disc_lcv(states_lcv, dones_lcv, log_pis_lcv, next_states_lcv)
                prob_exp_lcv = disc_lcv(states_exp, dones_exp.squeeze(-1), log_pis_exp_lcv, next_states_exp)

                prob_pi_fv = disc_fv(states_fv, dones_fv, log_pis_fv, next_states_fv)
                prob_exp_fv = disc_fv(states_exp, dones_exp.squeeze(-1), log_pis_exp_fv, next_states_exp)

                # 3. 基础 Loss
                loss_base = disc_criterion(prob_exp_lcv, torch.ones_like(prob_exp_lcv)) + \
                            disc_criterion(prob_pi_lcv, torch.zeros_like(prob_pi_lcv)) + \
                            disc_criterion(prob_exp_fv, torch.ones_like(prob_exp_fv)) + \
                            disc_criterion(prob_pi_fv, torch.zeros_like(prob_pi_fv))

                # 4. 额外正则项 Loss
                loss_soc = torch.tensor(0.0).to('cuda')
                loss_eq = torch.tensor(0.0).to('cuda')

                if USE_SOCIAL_POTENTIAL:
                    # A. 社会一致性
                    _, _, eps_lcv_exp = disc_lcv.get_reward(states_exp)
                    _, _, eps_fv_exp = disc_fv.get_reward(states_exp)
                    loss_soc = torch.mean(eps_lcv_exp ** 2) + torch.mean(eps_fv_exp ** 2)

                    # B. 博弈均衡
                    adv_lcv_exp = disc_lcv.f(states_exp, dones_exp.squeeze(-1), next_states_exp)
                    adv_fv_exp = disc_fv.f(states_exp, dones_exp.squeeze(-1), next_states_exp)
                    loss_eq = torch.mean(adv_lcv_exp ** 2) + torch.mean(adv_fv_exp ** 2)

                    # 总 Loss
                    loss_disc = loss_base + (LAMBDA_SOC * loss_soc) + (LAMBDA_EQ * loss_eq)
                else:
                    loss_disc = loss_base

                optim_disc.zero_grad()
                loss_disc.backward()
                optim_disc.step()

                # 收集 Loss 数据
                epoch_loss_total.append(loss_disc.item())
                epoch_loss_base.append(loss_base.item())
                if USE_SOCIAL_POTENTIAL:
                    epoch_loss_soc.append(loss_soc.item())
                    epoch_loss_eq.append(loss_eq.item())

            # [新增] 记录 Discriminator Loss 到 TensorBoard (取平均)
            writer.add_scalar('Discriminator/Total_Loss', np.mean(epoch_loss_total), i_episode)
            writer.add_scalar('Discriminator/Base_Loss', np.mean(epoch_loss_base), i_episode)
            if USE_SOCIAL_POTENTIAL:
                writer.add_scalar('Discriminator/Social_Loss', np.mean(epoch_loss_soc), i_episode)
                writer.add_scalar('Discriminator/Equilibrium_Loss', np.mean(epoch_loss_eq), i_episode)

            # --- PPO 更新阶段 ---
            states_lcv, actions_lcv, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.get()
            states_fv, actions_fv, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.get()
            next_states_lcv = next_states_lcv.float()
            next_states_fv = next_states_fv.float()
            dones_lcv = dones_lcv.int().to('cuda')
            dones_fv = dones_fv.int().to('cuda')

            rewards_lcv = disc_lcv.calculate_reward(states_lcv, dones_lcv, log_pis_lcv, next_states_lcv)
            rewards_fv = disc_fv.calculate_reward(states_fv, dones_fv, log_pis_fv, next_states_fv)

            # [新增] 记录恢复的奖励值 (Mean Recovered Reward)
            # 这是一个关键指标：如果学习正常，这个值应该趋于稳定，且不应该无限变大或变小
            writer.add_scalar('Recovered_Reward/LCV_Mean', rewards_lcv.mean().item(), i_episode)
            writer.add_scalar('Recovered_Reward/FV_Mean', rewards_fv.mean().item(), i_episode)

            model_lcv.buffer.rewards = []
            model_fv.buffer.rewards = []
            for sub_r in rewards_lcv:
                model_lcv.buffer.rewards.append(sub_r)
            for sub_r in rewards_fv:
                model_fv.buffer.rewards.append(sub_r)

            print(f'Episode {i_episode}, Disc Loss: {np.mean(epoch_loss_total):.4f}')
            model_lcv.update()
            model_fv.update()

        i_episode += 1
        scene_id += 1

        if scene_id > 150:
            scene_id = 0
            # ================= 模型保存 =================
            torch.save(disc_lcv, 'disc_lcv.pt')
            torch.save(disc_fv, 'disc_fv.pt')

            if USE_SOCIAL_POTENTIAL and social_phi_net is not None:
                torch.save(social_phi_net, 'social_phi.pt')
                print("Saved social_phi.pt")

            torch.save(model_lcv, 'model_lcv.pt')
            torch.save(model_fv, 'model_fv.pt')
            print(f"Models saved at episode {i_episode}")
            # ============================================

    # 训练结束关闭 Writer
    writer.close()
    env.close()


if __name__ == '__main__':
    run()
