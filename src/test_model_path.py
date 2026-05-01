import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from algorithm.maac_agent import MAACAgent
from battleField import BattleField, generating_unit

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后台，以免弹出GUI窗口

os.makedirs("../dclj/test_figures_path", exist_ok=True)


# Critic 只在训练时给出价值评估，测试阶段动作完全由 Actor 决定，因此只需加载 Actor。
def load_actor_model(agent, episode_num, path_idx):
    model_dir = f"models_path{path_idx}"              # 4 个子目录：models_path0 ~ models_path3
    for i, actor in enumerate(agent.actors):
        path = os.path.join(model_dir, f"actor_{i}_episode_{episode_num}.pth")
        actor.load_state_dict(torch.load(path, map_location=agent.device))
        agent.actor_targets[i].load_state_dict(actor.state_dict())
        actor.eval()
        agent.actor_targets[i].eval()

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

def test_model(episode_num=50, test_episodes=5, render=True):
    num_agents = 5
    agent = MAACAgent(state_dim=19 * 3, action_dim=3, num_agents=num_agents)

    # -------------测试循环-----------------
    # 日志容器
    sum_efficiency_averaged_history = []  # 记录每回合的平均频谱利用率
    win_history = []  # 记录每回合是否获胜(击毁蓝方驱逐舰)
    win_rate_history = []  # 统计胜率
    decision_time_log_step = []  # 记录每步决策时间
    decision_time_log_episode = []  # 记录本轮平均决策时间

    for episode in range(test_episodes):
        win_step = []

        # === (1)  按一定概率选一条路径 & 对应模型目录 ===========
        path_idx = np.random.choice(4, p=[0.25, 0.25, 0.25, 0.25],replace=False)  # 0 – 3
        load_actor_model(agent, episode_num, path_idx)  # 保证模型与路径一一对应
        print(f"\nEpisode{episode + 1}:选用第{path_idx+1}条路径\n")
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
        red_ewa_path0 = np.array([[150, 100, 7],
                                  [900, 500, 9],
                                  [500, 100, 8]])
        red_ewa_path1 = np.array([[100, 400, 7],
                                  [900, 500, 9],
                                  [500, 100, 8]])
        red_ewa_path2 = np.array([[100, 400, 7],
                                  [900, 500, 9],
                                  [500, 100, 8]])
        red_ewa_path3 = np.array([[100, 400, 7],
                                  [900, 500, 9],
                                  [500, 100, 8]])
        red_ewa_path4 = np.array([[100, 400, 7],
                                  [900, 500, 9],
                                  [500, 100, 8]])
        red_ewa_path = [red_ewa_path0, red_ewa_path1, red_ewa_path2, red_ewa_path3, red_ewa_path4]
        redEarlyWarningAircraft.navigator.set_path1(list(red_ewa_path[path_idx]))

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

        # 红方驱逐舰
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
        red_destroyer_path = np.array([[100, 600, 0],
                          [500, 100, 0]])
        reddestroyer.navigator.set_path1(list(red_destroyer_path))

        # 红方水面舰艇
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
            red_surface_path = np.array([[100, 600, 0],
                              [500, 100, 0]])
            surface_ship.navigator.set_path1(list(red_surface_path))

        # 蓝方预警机（Blue Early Warning Aircraft）
        blueEarlyWarningAircraft = generating_unit('EWA', 'blue', 'air', battleField, 1)
        blueEarlyWarningAircraft.jammer.on = False
        blueEarlyWarningAircraft.monitor.on = False
        blueEarlyWarningAircraft.radar.turn_on()
        blueEarlyWarningAircraft.radar.beamPower = 12 * 1e6
        for radarBeam in blueEarlyWarningAircraft.radar.beamList:
            radarBeam.set_angleRange(360)
        blue_ewa_path0 = np.array([[100, 600, 7],
                                   [900, 500, 9],
                                   [500, 100, 8]])
        blue_ewa_path1 = np.array([[120, 450, 7],
                                   [900, 400, 9],
                                   [500, 100, 8]])
        blue_ewa_path2 = np.array([[400, 150, 8],
                                   [400, 900, 8],
                                   [900, 350, 8]])
        blue_ewa_path3 = np.array([[150, 450, 7],
                                   [800, 800, 9],
                                   [550, 250, 8]])
        blue_ewa_path4 = np.array([[450, 300, 8],
                                   [800, 800, 8],
                                   [800, 400, 8]])
        blue_ewa_path = [blue_ewa_path0, blue_ewa_path1, blue_ewa_path2, blue_ewa_path3, blue_ewa_path4]
        blueEarlyWarningAircraft.navigator.set_path1(list(blue_ewa_path[path_idx]))

        # 蓝方干扰机（Blue Jamming Aircraft）
        blueJammingAircraft = generating_unit('EJA', 'blue', 'air', battleField, 1)
        blueJammingAircraft.radar.turn_on()
        blueJammingAircraft.jammer.set_maxEffectiveDistance(400)
        blueJammingAircraft.jammer.set_beamAngleRange(1)
        blueJammingAircraft.jammer.set_beamPitchRange([-60, 60])
        blue_eja_path0 = np.array([[100, 600, 7],
                                   [900, 500, 9],
                                   [500, 100, 8]])
        blue_eja_path1 = np.array([[120, 450, 7],
                                   [900, 400, 9],
                                   [500, 100, 8]])
        blue_eja_path2 = np.array([[400, 150, 8],
                                   [400, 900, 8],
                                   [900, 350, 8]])
        blue_eja_path3 = np.array([[150, 450, 7],
                                   [800, 800, 9],
                                   [550, 250, 8]])
        blue_eja_path4 = np.array([[450, 300, 8],
                                   [800, 800, 8],
                                   [800, 400, 8]])
        blue_eja_path = [blue_eja_path0, blue_eja_path1, blue_eja_path2, blue_eja_path3, blue_eja_path4]
        blueJammingAircraft.navigator.set_path1(list(blue_eja_path[path_idx]))

        # 蓝方战斗机（Blue Fighter）×5 架
        blue_fighter_path0 = np.array([[786, 835, 8]])
        blue_fighter_path1 = np.array([[450, 120, 8], [500, 900, 8]])
        blue_fighter_path2 = np.array([[150, 420, 7], [900, 420, 9]])
        blue_fighter_path3 = np.array([[400, 150, 8], [800, 800, 8]])
        blue_fighter_path4 = np.array([[200, 300, 7], [800, 620, 9]])
        blue_fighter_path = [blue_fighter_path0, blue_fighter_path1, blue_fighter_path2,
                             blue_fighter_path3, blue_fighter_path4]
        for i in range(5):
            bluePlane = generating_unit('BlueFighter', 'blue', 'air', battleField, i)
            bluePlane.radar.turn_on()
            bluePlane.jammer.set_beamAngleRange(1)
            bluePlane.jammer.beamPower = 40 * 1e3
            bluePlane.navigator.v_sample = 0.028
            bluePlane.navigator.set_path1(list(blue_fighter_path[path_idx]))

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
        blue_destroyer_path = np.array([[900, 600, 0],
                          [100, 600, 0],
                          [500, 100, 0]])
        destroyer.navigator.set_path1(list(blue_destroyer_path))

        # 蓝方水面舰艇
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
            blue_surface_path = np.array([[900, 600, 0],
                              [100, 600, 0],
                              [500, 100, 0]])
            surface_ship.navigator.set_path1(blue_surface_path)

        env = battleField

        state = env.reset()
        state = normalize_state(state)  # 将状态进行归一化处理
        step = 0
        done = False
        global_sum_efficiency = 0.0

        while not done:
            start_time = time.time()
            with torch.no_grad():  # 禁用梯度计算
                step += 1
                # 选择动作（关闭噪声）
                actions = agent.select_action(state, noise_bool=False)
                end_time = time.time()
                step_decision_time = end_time - start_time  # 单步决策耗时
                decision_time_log_step.append(step_decision_time)

                # 与环境交互
                next_state, reward, done, sum_efficiency_per_frame, is_win = env.rl_step(actions)
                next_state = normalize_state(next_state)  # 将状态进行归一化处理

                win_step.append(is_win)

                # 记录当前时间平均频谱利用率
                global_sum_efficiency += sum_efficiency_per_frame
                sum_efficiency_averaged = global_sum_efficiency / step

                state = next_state

                # 实时渲染
                if render:
                    env.render(sum_efficiency_per_frame, sum_efficiency_averaged,path_history=True)

                if done or step == 1500:  # 限制单回合最大步长，防止无尽循环
                    env.close()
                    break

        # 记录本轮是否获胜
        win_episode = any(win_step)
        win_history.append(win_episode)
        win_rate_history.append(np.mean(win_history))

        # 统计频谱利用率
        # 一轮的平均频谱利用率
        sum_efficiency_averaged = global_sum_efficiency / step
        sum_efficiency_averaged_history.append(sum_efficiency_averaged)

        if decision_time_log_step:  # 避免除零
            avg_step_time = np.mean(decision_time_log_step[-step:])  # 本局平均
        else:
            avg_step_time = 0.0
        decision_time_log_episode.append(avg_step_time*1e3)

        print(
            f"Test Episode {episode + 1:3d}/{test_episodes},第{path_idx+1}条路径,step:{step:4d},win_episode:{sum(win_history):3d}/{episode + 1:3d},"
            f"efficiency_averaged:{sum_efficiency_averaged_history[-1]:.2f},本轮平均决策时间：{avg_step_time*1e3:.2f} ms")
        decision_time_log_episode.append(avg_step_time * 1e3)  # 单位 ms

    # 输出胜率
    print(f"win_rate: {win_rate_history[-1] * 100:.2f}%")

    # 绘制胜率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(win_rate_history) + 1), win_rate_history)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Curve")
    plt.ylim(0.8, 1.2)
    plt.grid(True)
    plt.savefig(f"test_figures_path/win_rate.png")
    plt.close()

    # 绘制频谱利用率曲线
    plt.plot(range(1, len(sum_efficiency_averaged_history) + 1), sum_efficiency_averaged_history)
    plt.xlabel("Episode")
    plt.ylabel("Spectrum Utilization")
    plt.title("Spectrum Utilization Curve")
    plt.grid(True)
    plt.savefig(f"test_figures_path/Spectrum Utilization.png")
    plt.close()

    # 绘制决策时间曲线
    plt.plot(range(1, len(decision_time_log_episode) + 1), decision_time_log_episode)
    plt.xlabel("Episode")
    plt.ylabel("Decision Time(ms)")
    plt.title("Decision Time(ms)")
    plt.grid(True)
    plt.savefig(f"test_figures_path/Decision Time.png")
    plt.close()


if __name__ == "__main__":
    # 参数说明：
    # episode_num: 要加载的模型对应的训练轮次（需已存在对应模型文件）
    # test_episodes: 测试次数
    # render: 是否开启可视化
    test_model(episode_num=50, test_episodes=5, render=True)
