import torch
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ppo_model.PPO import PPO, RolloutBuffer
from NGSIM_env.envs.env import NGSIMEnv
import numpy as np
import warnings
import os
from datetime import datetime
from collections import deque

# 从 model.airl 导入模型类
from airl import AIRLDiscrim, SocialAIRLDiscrim, SocialPotentialNet

warnings.filterwarnings('ignore')

def run():
    # 保证日志路径与脚本位置绑定（不受启动目录影响）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # ================= 配置区域 (Configuration) =================
    # [开关] True: 启用共享社会势场 (Improved); False: 原始 AIRL (Original)
    USE_SOCIAL_POTENTIAL = True
    # [开关] 历史帧堆叠（考虑历史特征）；False 保持单帧输入
    USE_HISTORY = False
    HISTORY_LEN = 4  # 仅在 USE_HISTORY=True 时生效

    # 正则项权重
    LAMBDA_SOC = 0.5   # 社会一致性权重 (Social Consistency) — 增大势场约束
    LAMBDA_EQ = 0.01   # 博弈均衡权重 (Equilibrium / Best-Response) — 再次减小 EQ 影响

    # TensorBoard 日志目录设置
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = "SocialAIRL" if USE_SOCIAL_POTENTIAL else "OriginalAIRL"
    base_log_dir = os.path.join(script_dir, "runs", "log","reward_recovery_new")
    run_name = f"main2_{exp_name}_{time_str}"
    log_dir = os.path.join(base_log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)  # [新增] 初始化 Writer
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    expert_data_path = os.path.join(log_dir, "expert_data.pkl")
    os.makedirs(checkpoints_dir, exist_ok=True)  # 统一保存模型/缓冲的目录
    print(f">>> TensorBoard Logging to: {log_dir}")
    # ==========================================================

    env = NGSIMEnv(scene='us-101')
    env.generate_experts(save_path=expert_data_path)

    # 根据是否使用历史帧调整状态维度
    STATE_DIM = 16 * (HISTORY_LEN if USE_HISTORY else 1)

    # 历史帧工具：初始化/堆叠观测，便于在开关启用时考虑历史
    def init_history(obs):
        hist = deque(maxlen=HISTORY_LEN)
        for _ in range(HISTORY_LEN):
            hist.append(obs.copy())
        return hist

    def stack_history(hist):
        if not USE_HISTORY:
            return hist[-1]
        return np.concatenate(list(hist), axis=0)

    buffer_exp = RolloutBuffer()
    try:
        buffer_exp.add_exp(path=expert_data_path, use_history=USE_HISTORY, history_len=HISTORY_LEN)  # 对专家数据也做可选堆叠
    except TypeError:
        # 兼容旧版 RolloutBuffer（无 use_history 参数）
        buffer_exp.add_exp(path=expert_data_path)

    # Generator (PPO) 初始化
    model_lcv = PPO(state_dim=STATE_DIM, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                    has_continuous_action_space=True, action_std_init=0.2)
    model_fv = PPO(state_dim=STATE_DIM, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                   has_continuous_action_space=True, action_std_init=0.2)

    # ================= 判别器 (Discriminator) 初始化分支 =================
    social_phi_net = None

    if USE_SOCIAL_POTENTIAL:
        print(">>> Training Mode: Social Potential AIRL (Shared Framework)")
        # 1. 初始化共享社会势场网络
        social_phi_net = SocialPotentialNet(state_shape=STATE_DIM).to('cuda')

        # 2. 初始化改进版判别器
        disc_lcv = SocialAIRLDiscrim(state_shape=STATE_DIM, gamma=0.99, shared_phi_net=social_phi_net).to('cuda')
        disc_fv = SocialAIRLDiscrim(state_shape=STATE_DIM, gamma=0.99, shared_phi_net=social_phi_net).to('cuda')

        # 3. 优化器
        # 原实现将共享的 social_phi_net 同时出现在 disc_lcv/ disc_fv.parameters() 中，再单独添加，导致重复参数进入优化器。
        # optim_disc = Adam([
        #     {'params': disc_lcv.parameters()},
        #     {'params': disc_fv.parameters()},
        #     {'params': social_phi_net.parameters()}
        # ], lr=3e-4)

        # 拆分参数组并显式列出，确保每个参数只出现一次，避免 ValueError: parameters appear in more than one group。
        optim_disc = Adam([
            {'params': disc_lcv.private_g.parameters()},  # LCV 私有奖励网络
            {'params': disc_lcv.h.parameters()},          # LCV 值函数网络
            {'params': [disc_lcv.alpha]},                 # LCV 势场权重
            {'params': disc_fv.private_g.parameters()},   # FV 私有奖励网络
            {'params': disc_fv.h.parameters()},           # FV 值函数网络
            {'params': [disc_fv.alpha]},                  # FV 势场权重
            {'params': social_phi_net.parameters()}       # 共享社会势场，仅加入一次
        ], lr=3e-4)

    else:
        print(">>> Training Mode: Original MA-AIRL (Independent Framework)")
        # 1. 初始化原始判别器
        disc_lcv = AIRLDiscrim(state_shape=STATE_DIM, gamma=0.99).to('cuda')
        disc_fv = AIRLDiscrim(state_shape=STATE_DIM, gamma=0.99).to('cuda')

        # 2. 优化器
        optim_disc = Adam([
            {'params': disc_lcv.parameters()},
            {'params': disc_fv.parameters()}
        ], lr=3e-4)
    # ===================================================================
    # --- GPU 设备检查（一次性打印，确认是否在用 GPU） ---
    print(f">>> torch.cuda.is_available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f">>> GPU[0]: {torch.cuda.get_device_name(0)}")

    def log_device(name, module):
        try:
            dev = next(module.parameters()).device
        except StopIteration:
            dev = "no-params"
        print(f">>> Device[{name}]: {dev}")

    log_device("ppo_lcv_policy", model_lcv.policy)
    log_device("ppo_fv_policy", model_fv.policy)
    log_device("disc_lcv", disc_lcv)
    log_device("disc_fv", disc_fv)
    if USE_SOCIAL_POTENTIAL and social_phi_net is not None:
        log_device("social_phi_net", social_phi_net)
    # ===================================================================

    disc_criterion = nn.BCELoss()

    time_step = 0
    i_episode = 0
    scene_id = 0

    # 训练循环
    while i_episode <= 110000:
        obs = env.reset(scene_id=scene_id)
        hist_buf = init_history(obs)  # 历史缓冲（若关闭历史，则内部只重复当前帧）
        state = stack_history(hist_buf)
        current_ep_reward = 0

        # --- 采样阶段 (Generator Sampling) ---
        while True:
            action_lcv = model_lcv.select_action(state) * 5
            action_fv = model_fv.select_action(state) * 5
            action = np.concatenate([action_lcv, action_fv])
            next_obs, reward, done, _ = env.step(action)

            # 更新历史并获得下一个堆叠状态（若关闭历史则退化为单帧）
            hist_buf.append(next_obs)
            next_state = stack_history(hist_buf)

            model_lcv.buffer.rewards.append(reward)
            model_lcv.buffer.is_terminals.append(done)
            model_lcv.buffer.next_states.append(torch.from_numpy(next_state.reshape(1, -1)).to('cuda'))

            model_fv.buffer.rewards.append(reward)
            model_fv.buffer.is_terminals.append(done)
            model_fv.buffer.next_states.append(torch.from_numpy(next_state.reshape(1, -1)).to('cuda'))

            time_step += 1
            current_ep_reward += reward
            state = next_state  # 推进到下一个时间步（包含历史的状态表示）

            if time_step % int(1e5) == 0:
                model_lcv.decay_action_std(0.05, 0.1)
                model_fv.decay_action_std(0.05, 0.1)
                # 新增：将 action_std 下降过程同步到 TensorBoard，便于与命令行输出对齐
                writer.add_scalar('actor/action_std_lcv', model_lcv.action_std, time_step)
                writer.add_scalar('actor/action_std_fv', model_fv.action_std, time_step)
                # 当降至最小标准差时再打印一次，方便检索日志
                if model_lcv.action_std == 0.1 or model_fv.action_std == 0.1:
                    print(f"[ActionStd] Reached min_action_std=0.1 | LCV: {model_lcv.action_std}, FV: {model_fv.action_std}")

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
            acc_exp_lcv = []  # 新增：专家判别准确率（LCV）
            acc_exp_fv = []   # 新增：专家判别准确率（FV）
            acc_pi_lcv = []   # 新增：策略判别准确率（LCV）
            acc_pi_fv = []    # 新增：策略判别准确率（FV）

            batch_size = 64
            for _ in range(10):
                # 1. 采样
                states_lcv, actions_lcv, _, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.sample(batch_size)
                states_fv, actions_fv, _, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.sample(batch_size)
                states_exp, actions_exp, _, dones_exp, next_states_exp = buffer_exp.sample(batch_size, if_exp=True)

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

                # 数值检查工具，避免 NaN/Inf 进入 BCE 造成 CUDA assert
                def check_probs(name, prob):
                    if torch.isnan(prob).any() or torch.isinf(prob).any():
                        raise RuntimeError(f"{name} contains NaN/Inf, min={prob.min().item()}, max={prob.max().item()}")

                def has_invalid(name, tensor):
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        print(f"[Skip batch] {name} has NaN/Inf")
                        return True
                    return False

                # 若输入/动作存在 NaN/Inf，跳过本次迭代
                if any([
                    has_invalid("states_lcv", states_lcv),
                    has_invalid("states_fv", states_fv),
                    has_invalid("states_exp", states_exp),
                    has_invalid("next_states_lcv", next_states_lcv),
                    has_invalid("next_states_fv", next_states_fv),
                    has_invalid("next_states_exp", next_states_exp),
                ]):
                    continue

                with torch.no_grad():
                    log_pis_exp_lcv, _, _ = model_lcv.policy.evaluate(states_exp, actions_exp[:, :2])
                    log_pis_exp_fv, _, _ = model_fv.policy.evaluate(states_exp, actions_exp[:, 2:])
                    # 检查 log_pi 是否异常
                    if has_invalid("log_pis_exp_lcv", log_pis_exp_lcv) or has_invalid("log_pis_exp_fv", log_pis_exp_fv):
                        continue

                # 2. 计算 Discriminator 输出
                prob_pi_lcv = disc_lcv(states_lcv, dones_lcv, log_pis_lcv, next_states_lcv)
                prob_exp_lcv = disc_lcv(states_exp, dones_exp.squeeze(-1), log_pis_exp_lcv, next_states_exp)

                prob_pi_fv = disc_fv(states_fv, dones_fv, log_pis_fv, next_states_fv)
                prob_exp_fv = disc_fv(states_exp, dones_exp.squeeze(-1), log_pis_exp_fv, next_states_exp)

                # 数值稳定：避免极端 0/1 造成 BCE 梯度消失
                prob_pi_lcv = torch.clamp(prob_pi_lcv, 0.05, 0.95)
                prob_exp_lcv = torch.clamp(prob_exp_lcv, 0.05, 0.95)
                prob_pi_fv = torch.clamp(prob_pi_fv, 0.05, 0.95)
                prob_exp_fv = torch.clamp(prob_exp_fv, 0.05, 0.95)

                check_probs("prob_exp_lcv", prob_exp_lcv)
                check_probs("prob_pi_lcv", prob_pi_lcv)
                check_probs("prob_exp_fv", prob_exp_fv)
                check_probs("prob_pi_fv", prob_pi_fv)
                # BCE 要求输入在 [0,1]，为数值稳定做截断
                # prob_exp_lcv = torch.clamp(prob_exp_lcv, 1e-6, 1 - 1e-6)
                # prob_pi_lcv = torch.clamp(prob_pi_lcv, 1e-6, 1 - 1e-6)
                # prob_exp_fv = torch.clamp(prob_exp_fv, 1e-6, 1 - 1e-6)
                # prob_pi_fv = torch.clamp(prob_pi_fv, 1e-6, 1 - 1e-6)

                # 判别准确率计算（与 main.py 保持一致的定义）
                expert_acc_lcv = ((prob_exp_lcv > 0.5).float()).mean().detach().cpu()
                expert_acc_fv = ((prob_exp_fv > 0.5).float()).mean().detach().cpu()
                learner_acc_lcv = ((prob_pi_lcv < 0.5).float()).mean().detach().cpu()
                learner_acc_fv = ((prob_pi_fv < 0.5).float()).mean().detach().cpu()

                # 3. 基础 Loss
                # Label smoothing 避免判别器饱和：专家 0.9，策略 0.1
                expert_target = torch.full_like(prob_exp_lcv, 0.9, device=prob_exp_lcv.device)
                agent_target = torch.full_like(prob_pi_lcv, 0.1, device=prob_pi_lcv.device)

                #Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
                expert_loss = disc_criterion(prob_exp_lcv, expert_target) + \
                              disc_criterion(prob_exp_fv, expert_target)
                agent_loss = disc_criterion(prob_pi_lcv, agent_target) + \
                             disc_criterion(prob_pi_fv, agent_target)
                # [修改后] 加入 0.1 的平滑
            #     smooth_val = 0.1
            #     real_label = 1.0 - smooth_val  # 0.9
            #     fake_label = 0.0 + smooth_val  # 0.1

            #     expert_loss = disc_criterion(prob_exp_lcv, torch.full_like(prob_exp_lcv, real_label)) + \
            #   disc_criterion(prob_exp_fv, torch.full_like(prob_exp_fv, real_label))
            #     agent_loss = disc_criterion(prob_pi_lcv, torch.full_like(prob_pi_lcv, fake_label)) + \
            #  disc_criterion(prob_pi_fv, torch.full_like(prob_pi_fv, fake_label))

                loss_base = (expert_loss + agent_loss)

                # 4. 额外正则项 Loss
                loss_soc = torch.tensor(0.0).to('cuda')
                loss_eq = torch.tensor(0.0).to('cuda')

                if USE_SOCIAL_POTENTIAL:
                    # A. 社会一致性
                    _, _, eps_lcv_exp = disc_lcv.get_reward(states_exp)
                    _, _, eps_fv_exp = disc_fv.get_reward(states_exp)
                    loss_soc = torch.mean(eps_lcv_exp ** 2) + torch.mean(eps_fv_exp ** 2)

                    # B. 博弈均衡
                    adv_lcv_exp = torch.clamp(disc_lcv.f(states_exp, dones_exp.squeeze(-1), next_states_exp), -5.0, 5.0)
                    adv_fv_exp = torch.clamp(disc_fv.f(states_exp, dones_exp.squeeze(-1), next_states_exp), -5.0, 5.0)
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
                # 收集判别准确率数据，便于后续打印/记录
                acc_exp_lcv.append(expert_acc_lcv)
                acc_exp_fv.append(expert_acc_fv)
                acc_pi_lcv.append(learner_acc_lcv)
                acc_pi_fv.append(learner_acc_fv)

            # [新增] 记录 Discriminator Loss 到 TensorBoard (取平均)
            writer.add_scalar('Discriminator/Total_Loss', np.mean(epoch_loss_total), i_episode)
            writer.add_scalar('Discriminator/Base_Loss', np.mean(epoch_loss_base), i_episode)
            if USE_SOCIAL_POTENTIAL:
                writer.add_scalar('Discriminator/Social_Loss', np.mean(epoch_loss_soc), i_episode)
                writer.add_scalar('Discriminator/Equilibrium_Loss', np.mean(epoch_loss_eq), i_episode)

            # --- PPO 更新阶段 ---
            states_lcv, actions_lcv, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.get()
            states_fv, actions_fv, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.get()
            torch.save(model_lcv.buffer, os.path.join(checkpoints_dir, 'lcv_buffer.pt'))
            torch.save(model_fv.buffer, os.path.join(checkpoints_dir, 'fv_buffer.pt'))
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

            # 将判别器奖励均值压缩输出，避免异常大值
            rewards_lcv = torch.clamp(rewards_lcv, -10.0, 10.0)
            rewards_fv = torch.clamp(rewards_fv, -10.0, 10.0)

            model_lcv.buffer.rewards = []
            model_fv.buffer.rewards = []
            for sub_r in rewards_lcv:
                model_lcv.buffer.rewards.append(sub_r)
            for sub_r in rewards_fv:
                model_fv.buffer.rewards.append(sub_r)

            # 计算均值指标（与 main.py 对齐）
            disc_loss_mean = float(np.mean(epoch_loss_total)) if len(epoch_loss_total) > 0 else 0.0
            acc_exp_lcv_mean = float(np.mean(acc_exp_lcv)) if len(acc_exp_lcv) > 0 else 0.0
            acc_pi_lcv_mean = float(np.mean(acc_pi_lcv)) if len(acc_pi_lcv) > 0 else 0.0
            acc_exp_fv_mean = float(np.mean(acc_exp_fv)) if len(acc_exp_fv) > 0 else 0.0
            acc_pi_fv_mean = float(np.mean(acc_pi_fv)) if len(acc_pi_fv) > 0 else 0.0

            # print(f'Episode {i_episode}, Disc Loss: {np.mean(epoch_loss_total):.4f}')  # 原输出已被下方对齐 main.py 的打印替代
            # 终端输出，格式与 main.py 一致
            print('Epoch disc loss {:.4}, acc exp lcv {:.4}, acc pi lcv {:.4},acc exp fv {:.4}, acc pi fv {:.4}'
                  .format(disc_loss_mean, acc_exp_lcv_mean, acc_pi_lcv_mean, acc_exp_fv_mean, acc_pi_fv_mean))

            # TensorBoard 记录对齐 main.py 的指标
            writer.add_scalar('disc/loss', disc_loss_mean, i_episode)
            writer.add_scalar('disc/acc_exp_lcv', acc_exp_lcv_mean, i_episode)
            writer.add_scalar('disc/acc_pi_lcv', acc_pi_lcv_mean, i_episode)
            writer.add_scalar('disc/acc_exp_fv', acc_exp_fv_mean, i_episode)
            writer.add_scalar('disc/acc_pi_fv', acc_pi_fv_mean, i_episode)

            # PPO 更新
            model_lcv.update()
            model_fv.update()

            # 输出和记录恢复奖励均值（与 main.py 对齐）
            print('Episode {}, reward {:.4}, disc r lcv {:.4}, disc r fv {:.4}'
                  .format(i_episode, current_ep_reward, rewards_lcv.mean(), rewards_fv.mean()))
            writer.add_scalar('disc/reward_lcv', rewards_lcv.mean().item(), i_episode)
            writer.add_scalar('disc/reward_fv', rewards_fv.mean().item(), i_episode)

        i_episode += 1
        scene_id += 1

        if scene_id > 150:
            scene_id = 0
            # ================= 模型保存 =================
            torch.save(disc_lcv, os.path.join(checkpoints_dir, 'disc_lcv.pt'))
            torch.save(disc_fv, os.path.join(checkpoints_dir, 'disc_fv.pt'))

            if USE_SOCIAL_POTENTIAL and social_phi_net is not None:
                torch.save(social_phi_net, os.path.join(checkpoints_dir, 'social_phi.pt'))
                print("Saved social_phi.pt")

            torch.save(model_lcv, os.path.join(checkpoints_dir, 'model_lcv.pt'))
            torch.save(model_fv, os.path.join(checkpoints_dir, 'model_fv.pt'))
            print(f"Models saved at episode {i_episode}")
            # ============================================

    # 训练结束关闭 Writer
    writer.close()
    env.close()


if __name__ == '__main__':
    run()
