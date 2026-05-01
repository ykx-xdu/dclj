import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt

from src.algorithm.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.algorithm.maac_agent import MAACAgent
from src.battleField import BattleField, generating_unit  # 导入战场环境

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后台，以免误弹出GUI窗口

if __name__ == "__main__":
    # 创建保存模型和图形的目录
    os.makedirs("models_path0", exist_ok=True)
    os.makedirs("figures_path0", exist_ok=True)

    # 设置随机种子，保证实验可重复
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # 超参数设置
    state_dim = (20 - 1) * 3
    action_dim = 3  # 每个单位的动作维度为3 (速度分量 [vx, vy, vz])
    max_episodes = 200
    batch_size = 512  # 每次从经验缓存中取出的经验数
    replay_buffer_size = 500000
    update_after = 1500 * 5   # 前5个Episode不更新模型，即随机探索过程
    update_every = 10  # 更新频率，每交互 10 步执行一次更新

    # 日志容器
    actor_losses, critic_losses = [], []
    agent_rewards_history = [[] for _ in range(5)]  # 5 个智能体
    sum_efficiency_averaged_history = []
    win_history = [] # 记录每回合是否获胜(击毁蓝方驱逐舰)
    win_rate_history = [] # 统计胜率


    # 定义状态的归一化函数
    def normalize_state(state):
        s = state.copy()
        # 对每个维度分别归一化
        s[..., 0::3] = s[..., 0::3] / 1000.0  # x 轴归一化到 [-1, 1]
        s[..., 1::3] = s[..., 1::3] / 1000.0  # y 轴归一化
        s[..., 2::3] = s[..., 2::3] / 10.0  # z 轴归一化
        # 将 [-1, 1] 映射到 [0, 1]
        s = (s + 1.0) * 0.5
        return s.astype(np.float32)

    # 获取智能体数量（仅红方战斗机）
    num_agents = 5 #（仅红方5架战斗机）
    agent = MAACAgent(state_dim=state_dim, action_dim=action_dim, num_agents=num_agents)
    replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size, alpha=0.6, beta=0.4)
    Average_rewards = []

    for episode in range(max_episodes):
        win_step = []
        # 初始化战场环境
        battleField = BattleField('air-air',
                                  timer=0,
                                  size=[1000, 1000, 10],
                                  acceleration=100,
                                  frameTimeInterval=0.02,
                                  simTimeInterval=1,
                                  simDuration=60 * 180)

        # 红方预警机（Early Warning Aircraft）
        redEarlyWarningAircraft = generating_unit('EWA', 'red', 'air', battleField, 1)
        redEarlyWarningAircraft.jammer.on = False  # 关闭干扰功能
        redEarlyWarningAircraft.monitor.on = False  # 关闭监视功能
        redEarlyWarningAircraft.radar.turn_on()  # 打开雷达
        redEarlyWarningAircraft.radar.beamPower = 12 * 1e6
        # 设置预警机雷达波束角度为360度（全向扫描）
        for radarBeam in redEarlyWarningAircraft.radar.beamList:
            radarBeam.set_angleRange(360)
        # 定义预警机导航路径（3个航点坐标）
        path0 = np.array([[150, 100, 7],
                          [900, 500, 9],
                          [500, 100, 8]])
        redEarlyWarningAircraft.navigator.set_path1(list(path0))

        # 红方干扰机（Electronic Jamming Aircraft）
        redJammingAircraft = generating_unit('EJA', 'red', 'air', battleField, 1)
        redJammingAircraft.radar.turn_on()
        redJammingAircraft.jammer.set_maxEffectiveDistance(400)
        redJammingAircraft.jammer.set_beamAngleRange(1)
        redJammingAircraft.jammer.set_beamPitchRange([-60, 60])

        # 红方战斗机（Red Fighter）× 5 架
        for i in range(5):
            redPlane = generating_unit('RedFighter', 'red', 'air', battleField, i)
            redPlane.navigator.v_sample = 0.03
            redPlane.radar.turn_on()  # 是否打开/关闭战斗机雷达
            redPlane.jammer.set_beamAngleRange(1)
            redPlane.jammer.beamPower = 40 * 1e3

        # 加入驱逐舰
        reddestroyer = generating_unit('Destroyer', 'red', 'sea', battleField, 1)
        reddestroyer.jammer.on = True
        reddestroyer.navigator.v_sample = 0.005
        reddestroyer.monitor.on = False
        reddestroyer.radar.turn_on()
        # 设置干扰机干扰角度 unit类中self.jammer
        reddestroyer.jammer.set_beamAngleRange(1)
        reddestroyer.jammer.set_beamPitchRange([-70, 90])
        for radarBeam in reddestroyer.radar.beamList:
            radarBeam.set_angleRange(360)
        path0 = np.array([[100, 600, 0],
                          [500, 100, 0]])
        reddestroyer.navigator.set_path1(list(path0))
        for i in range(2):
            surface_ship = generating_unit('SurfaceShip', 'red', 'sea', battleField, i)
            surface_ship.jammer.on = True
            surface_ship.monitor.on = False  # TODO
            surface_ship.radar.turn_on()
            # surface_ship.radar.beamPower = 10 * 1e6
            for radarBeam in surface_ship.radar.beamList:
                radarBeam.set_angleRange(360)
            # 设置干扰机干扰角度 unit类中self.jammer
            surface_ship.jammer.set_beamAngleRange(1)
            surface_ship.jammer.set_beamPitchRange([-70, 70])
            surface_ship.navigator.v_sample = 0.005
            path0 = np.array([[100, 600, 0],
                              [500, 100, 0]])
            surface_ship.navigator.set_path1(list(path0))

        # 蓝方预警机（Blue Early Warning Aircraft）
        blueEarlyWarningAircraft = generating_unit('EWA', 'blue', 'air', battleField, 1)
        blueEarlyWarningAircraft.jammer.on = False
        blueEarlyWarningAircraft.monitor.on = False
        blueEarlyWarningAircraft.radar.turn_on()
        blueEarlyWarningAircraft.radar.beamPower = 12 * 1e6
        for radarBeam in blueEarlyWarningAircraft.radar.beamList:
            radarBeam.set_angleRange(360)
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])
        blueEarlyWarningAircraft.navigator.set_path1(list(path0))

        # 蓝方干扰机（Blue Jamming Aircraft）
        blueJammingAircraft = generating_unit('EJA', 'blue', 'air', battleField, 1)
        blueJammingAircraft.radar.turn_on()
        blueJammingAircraft.jammer.set_maxEffectiveDistance(400)
        blueJammingAircraft.jammer.set_beamAngleRange(1)
        blueJammingAircraft.jammer.set_beamPitchRange([-60, 60])
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])
        blueJammingAircraft.navigator.set_path1(list(path0))

        # 蓝方战斗机（Blue Fighter）×5 架
        path = np.array([[786, 835, 8]])
        for i in range(5):
            bluePlane = generating_unit('BlueFighter', 'blue', 'air', battleField, i)
            bluePlane.radar.turn_on()
            bluePlane.jammer.set_beamAngleRange(1)
            bluePlane.jammer.beamPower = 40 * 1e3
            bluePlane.navigator.v_sample = 0.028
            bluePlane.navigator.set_path1(list(path))

        # 蓝方驱逐舰
        destroyer = generating_unit('Destroyer', 'blue', 'sea', battleField, 1)
        destroyer.jammer.on = True
        destroyer.navigator.v_sample = 0.005
        destroyer.monitor.on = False
        destroyer.radar.turn_on()
        # 设置干扰机干扰角度 unit类中self.jammer
        destroyer.jammer.set_beamAngleRange(1)
        destroyer.jammer.set_beamPitchRange([-70, 90])
        for radarBeam in destroyer.radar.beamList:
            radarBeam.set_angleRange(360)
        path0 = np.array([[900, 600, 0],
                          [100, 600, 0],
                          [500, 100, 0]])
        destroyer.navigator.set_path1(list(path0))

        # 加入驱逐舰
        for i in range(2):
            surface_ship = generating_unit('SurfaceShip', 'blue', 'sea', battleField, i)
            surface_ship.jammer.on = True
            surface_ship.monitor.on = False  # TODO
            surface_ship.radar.turn_on()
            # surface_ship.radar.beamPower = 10 * 1e6
            for radarBeam in surface_ship.radar.beamList:
                radarBeam.set_angleRange(360)
            # 设置干扰机干扰角度 unit类中self.jammer
            surface_ship.jammer.set_beamAngleRange(1)
            surface_ship.jammer.set_beamPitchRange([-70, 70])
            surface_ship.navigator.v_sample = 0.005
            path0 = np.array([[900, 600, 0],
                              [100, 600, 0],
                              [500, 100, 0]])
            surface_ship.navigator.set_path1(path0)

        env = battleField

        state = env.reset()
        state = normalize_state(state)  # 将状态进行归一化处理
        agent.reset_noise()  # 重置探索噪声（OU噪声状态归零）
        episode_reward = 0
        agent_episode_rewards = np.zeros(num_agents, dtype=np.float32)  # 统计每个智能体的奖励
        step = 0
        done = False
        global_sum_efficiency = 0.0

        while not done:
            step += 1
            # 1. 选择动作（策略网络输出 + OU 探索噪声）
            actions = agent.select_action(state) # → shape = (5, 3)
            # 2. 与环境交互
            next_state, reward, done, sum_efficiency_per_frame, is_win= env.rl_step(actions)
            next_state = normalize_state(next_state)  # 将状态进行归一化处理
            win_step.append(is_win)
            # 3. 累积奖励并存储经验
            episode_reward += np.mean(reward)
            agent_episode_rewards += reward          # 逐智能体
            replay_buffer.add((state, actions, reward, next_state, done))

            # 记录当前时间平均频谱利用率
            global_sum_efficiency += sum_efficiency_per_frame
            sum_efficiency_averaged = global_sum_efficiency / step

            # 4. 更新模型参数（延迟一定步数后开始，每隔 update_every 步更新一次）
            if len(replay_buffer) >= update_after and step % update_every == 0:
                agent.update(replay_buffer, batch_size)

            if agent.last_actor_loss is not None:
                actor_losses.append(agent.last_actor_loss)
            if agent.last_critic_loss is not None:
                critic_losses.append(agent.last_critic_loss)

            # 5. 转换到下一个状态
            state = next_state

            if not done:
                env.render(sum_efficiency_per_frame, sum_efficiency_averaged)

            if done or step == 1500:  # 限制单回合最大步长，防止无尽循环
                env.close()
                break

        # 记录本轮的奖励
        avg_episode_reward = episode_reward / step
        Average_rewards.append(episode_reward / step)
        avg_agent_rewards = agent_episode_rewards / step  # 每个智能体的平均奖励

        # 记录本轮是否获胜
        win_episode = any(win_step)
        win_history.append(win_episode)
        win_rate_history.append(np.mean(win_history))

        for i in range(num_agents):
            agent_rewards_history[i].append(avg_agent_rewards[i])

        print(
            f"Episode:{episode + 1:4d}, "
            f"Step:{step:4d}, "
            f"Avg_Reward:{avg_episode_reward: .3f}, "
            f"Agent_Rewards:{[round(x, 3) for x in avg_agent_rewards]},"
            f"是否获胜：{win_episode}"
        )

        # 一轮的平均频谱利用率
        sum_efficiency_averaged = global_sum_efficiency / step
        sum_efficiency_averaged_history.append(sum_efficiency_averaged)

        # === 定期保存模型和所有曲线 ===
        if (episode + 1) % 10 == 0:
            # 保存每个智能体的 Actor 模型参数,以便用于后续测试
            for agent_idx, actor in enumerate(agent.actors):
                torch.save(actor.state_dict(), f"models_path0/actor_{agent_idx}_episode_{episode + 1}.pth")

            # ① 所有智能体的平均奖励
            plt.figure(figsize=(8, 5))
            x = range(1, episode + 2)
            plt.plot(x, Average_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("Total Average Reward")
            plt.grid(True)  # 添加网格线
            plt.savefig(f"figures_path0/rewards_episode_{episode + 1}.png")
            plt.close()

            # ② Actor loss 曲线
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(actor_losses) + 1), actor_losses)
            plt.xlabel("Update Step")
            plt.ylabel("Actor Loss")
            plt.title("Actor Loss Curve")
            plt.grid(True)
            plt.savefig(f"figures_path0/actor_loss_episode_{episode + 1}.png")
            plt.close()

            # ③ Critic loss 曲线
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(critic_losses) + 1), critic_losses)
            plt.xlabel("Update Step")
            plt.ylabel("Critic Loss")
            plt.title("Critic Loss Curve")
            plt.grid(True)
            plt.savefig(f"figures_path0/critic_loss_episode_{episode + 1}.png")
            plt.close()

            # ④ 每个智能体奖励
            plt.figure(figsize=(8, 5))
            x_epi = range(1, len(agent_rewards_history[0]) + 1)
            for i in range(num_agents):
                plt.plot(x_epi, agent_rewards_history[i], label=f'Agent {i}')
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title("Per-Agent Reward Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"figures_path0/agent_rewards_episode_{episode + 1}.png")
            plt.close()

            # ⑤ 频谱利用率曲线
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(sum_efficiency_averaged_history) + 1), sum_efficiency_averaged_history)
            plt.xlabel("Episode")
            plt.ylabel("Spectrum Utilization")
            plt.title("Spectrum Utilization Curve")
            plt.grid(True)
            plt.savefig(f"figures_path0/Spectrum Utilization_episode_{episode + 1}.png")
            plt.close()

            # ⑥ 本轮是否获胜曲线
            plt.figure(figsize=(8, 5))
            x_epi = range(1, len(win_history) + 1)
            plt.plot(x_epi, win_history)
            plt.xlabel("Episode")
            plt.ylabel("Is Win(0/1)")
            plt.title("Is Win(0/1)")
            plt.ylim(-0.2, 1.2)
            plt.grid(True)
            plt.savefig(f"figures_path0/is_win_episode_{episode + 1}.png")
            plt.close()
