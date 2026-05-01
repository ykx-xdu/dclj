from gc import set_debug

import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
import json
import matplotlib.pyplot as plt
import random


#战场类, 用于设定战场的时空参数，创建战场画布，计算时空量，统计战场上的所有装备
class BattleField:
    def __init__(self,
                 environment_name,
                 timer=0,
                 size=(1000, 1000, 10),
                 frameTimeInterval=0.1,  # 帧时隙 0.1秒 帧之间的时间间隔
                 acceleration=100,  # 加速倍数 100倍
                 simTimeInterval=1,  # 模拟时隙 1秒
                 simDuration=45 * 60,
                 ):
        self.environment_name = environment_name
        self.size = size
        self.timer = timer
        self.frameTimeInterval = frameTimeInterval
        self.simTimeInterval = simTimeInterval
        self.simDuration = simDuration
        self.acceleration = acceleration
        self.simNumPerFrame = frameTimeInterval * acceleration // simTimeInterval  # 每帧的模拟次数（运行多少步显示一次）

        self.unitList = []
        self.crashedUnitList = []
        self.flyingMissileList = []

        self.positionList = []
        self.distanceMatrix = []
        self.orientationMatrix = []
        # 加入俯仰角矩阵
        self.pitchMatrix = []

        self.maxInfluenceDistanceList = []
        self.influenceMatrix = []

        self.linkList = []
        self.jammedLinkList = []
        self.singleConectedLinkList = []
        self.maxLinkDistanceMatrix = None
        self.currentLinkDistanceMatrix = None
        self.linkMatrix = []
        self.linkMatrixChanged = False
        self.networkList = []
        self.networkConectionMatrix = None

        self.window = tk.Tk()
        self.window.title("Battle Field")
        self.window.geometry('%dx%d' % (size[0], size[1]))
        self.canvas = tk.Canvas(self.window,
                                bg="lightblue",
                                width=size[0],
                                height=size[1], )
        self.pause = False
        self.canvas.pack()
        self.canvas.grid()

        self.jammedDict = {}  # 受扰字典，结构为 {target: [fc, Bw, Pi],...}

    #添加装备
    def add_unit(self, unit):
        unit.unitIndex = len(self.unitList)
        self.unitList.append(unit)
        self.positionList.append(unit.position)
        self.maxInfluenceDistanceList.append([unit.maxInfluenceDistance])

    def initialize_links(self):
        unitNum = len(self.unitList)
        self.linkMatrix = np.zeros([unitNum, unitNum], dtype=np.bool_)  #单位之间的链接关系，初始时所有链接都是 False
        self.maxLinkDistanceMatrix = -np.ones([unitNum, unitNum])  #初始化为 -1，存储单位之间的最大链接距离，初始时表示未定义或不可达的状态。
        for sender in self.unitList:
            for receiver in self.unitList:
                #print('linkEnds =', [sender.unitIndex, receiver.unitIndex])
                if sender.color == receiver.color and sender != receiver:
                    sender.communicator.build_link(receiver)

    def communication(self):
        previousLinkMatrix = self.linkMatrix.copy()
        #begin0 = time.time()
        #计算网络是否发生变化

        # todo: 更改最大连接距离矩阵的计算方式
        self.currentLinkDistanceMatrix = self.maxLinkDistanceMatrix.copy()
        self.jammedLinkList = list(set(self.jammedLinkList))  #去重干扰链接jammedLinkList，确保每个链接只出现一次。
        # print('jammedLinkList =', len(self.jammedLinkList))
        for link in self.jammedLinkList:

            if not link.ends[0].crashed and not link.ends[1].crashed:
                linkDistance = link.get_linkDistanceUnderJamming()

                # 被通信干扰后,不能发但能收
                # self.currentLinkDistanceMatrix[link.ends[0].unitIndex, link.ends[1].unitIndex] = linkDistance
                # self.currentLinkDistanceMatrix[link.ends[1].unitIndex, link.ends[0].unitIndex] = linkDistance

                # 只要被通信干扰，既不能收也不能发
                for num in range(len(self.currentLinkDistanceMatrix[0])):  #遍历列
                    if link.ends[0].color == self.unitList[num].color and link.ends[0] != self.unitList[num]:
                        self.currentLinkDistanceMatrix[link.ends[0].unitIndex, num] = linkDistance

        self.linkMatrix = ((self.currentLinkDistanceMatrix - self.distanceMatrix) > 0)
        #end0 = time.time()
        #print('time0 =', end0-begin0)

        # begin1 = time.time()
        #当网络发生变化时,重新计算视野共享关系
        if (self.linkMatrix != previousLinkMatrix).any():
            #找出单向链路
            linkMatrix = self.linkMatrix.copy()
            self.singleConectedLinkList = []
            for link in self.jammedLinkList:
                if not linkMatrix[link.ends[0].unitIndex, link.ends[1].unitIndex] \
                        and linkMatrix[link.ends[1].unitIndex, link.ends[0].unitIndex]:
                    self.singleConectedLinkList.append([link.ends[1], link.ends[0]])
                    linkMatrix[link.ends[1].unitIndex, link.ends[0].unitIndex] = False

            #print('singleConectedLinkList =')
            #for link in self.singleConectedLinkList:
            #    print([link[0].unitIndex, link[1].unitIndex])

            #去除单连通链路后,找出所有子网
            self.networkList = []
            unitNum = len(self.unitList)
            pendingUnitIndexList = list(range(unitNum))
            status = np.ones([unitNum])
            while pendingUnitIndexList:
                unitIndex = pendingUnitIndexList[0]
                if self.unitList[unitIndex].crashed:
                    del pendingUnitIndexList[0]
                else:
                    network = [unitIndex]
                    for sender in network:

                        if status[sender]:
                            receivers = list(np.where(linkMatrix[sender])[0])
                            network += receivers
                            status[sender] = False
                    network = set(network)
                    for unitIndex in network:
                        self.unitList[unitIndex].communicator.networkIndex = len(self.networkList)
                    self.networkList.append(network)
                    pendingUnitIndexList = list(set(pendingUnitIndexList).difference(network))
            #print('networkList =')
            #print(self.networkList)

            #根据单向链路的传输方向,计算各子网之间的连接关系
            networkNum = len(self.networkList)
            self.networkConectionMatrix = np.zeros([networkNum, networkNum], dtype=np.bool_)
            if networkNum > 2:
                for link in self.singleConectedLinkList:
                    senderNetworkIndex = link[0].communicator.networkIndex
                    receiverNetworkIndex = link[1].communicator.networkIndex
                    self.networkConectionMatrix[senderNetworkIndex, receiverNetworkIndex] = True
        # end1 = time.time()
        # print('time1 =', end1-begin1)

        # begin2 = time.time()
        #计算各子网内部的共享视野
        self.networkVisionList = []
        for network in self.networkList:
            networkVision = []
            for unitIndex in network:
                unit = self.unitList[unitIndex]
                if unit.withRadar and unit.radar.on:
                    networkVision += unit.radar.detectedUnits
            networkVision = list(set(networkVision))
            self.networkVisionList.append(networkVision)
        # end2 = time.time()
        # print('time2 =', end2-begin2)

        # begin3 = time.time()
        # 计算子网之间的共享视野
        networkNum = len(self.networkList)
        for receiverNetwork in range(networkNum):
            if networkNum > 2:
                for senderNetwork in range(networkNum):
                    if senderNetwork != receiverNetwork and self.networkConectionMatrix[senderNetwork, receiverNetwork]:
                        self.networkVisionList[receiverNetwork] += self.networkVisionList[senderNetwork]
            self.networkVisionList[receiverNetwork] = list(set(self.networkVisionList[receiverNetwork]))
            #print('network =', self.networkList[receiverNetwork], end=' ')
            #print('visiton = ', end='{')
            #for unit in self.networkVisionList[receiverNetwork]:
            #    print(unit.unitIndex, end=', ')
            #print('}')

            #将共享视野分发给各单位
            network = self.networkList[receiverNetwork]
            for unitIndex in network:
                unit = self.unitList[unitIndex]
                unit.network = network
                unit.radar.detectedUnits = self.networkVisionList[receiverNetwork]
                unit.communicator.linkedUnitList = network
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                # print(unit.color + unit.name + str(unit.unitIndex) + "通信距离" + str(unit.linkDistance))
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        #for link in self.jammedLinkList:
        #    link.jammed = False
        self.jammedLinkList = []

    # end3 = time.time()
    # print('time3 =', end3-begin3)
    # print('--------------------')

    def displayLinks(self):
        for i in range(len(self.unitList)):
            position_i = self.unitList[i].position

            for j in range(len(self.unitList)):
                position_j = self.unitList[j].position

                if self.linkMatrix[i, j] and self.linkMatrix[j, i]:
                    self.linkImage = self.canvas.create_line(position_i[0],
                                                             position_i[1],
                                                             position_j[0],
                                                             position_j[1],
                                                             fill='green',
                                                             dash=(1, 5),
                                                             width=0.1,
                                                             )
                #lif self.unitList[i].

    #def create_arc(self, position, direction, angle, ):

    #计算战场中设备两两之间的距离和角度
    def battlefield_situation(self):
        positionList = np.array(self.positionList)
        unitNum = len(self.unitList)
        diff = np.zeros([unitNum, unitNum, 3], dtype=np.float32)  #存储(x, y, z)坐标差
        for i in range(unitNum):
            diff[i] = positionList - positionList[i]  #diff[i]包含所有单位相对于第 i 个单位的位置差
        diff += 1e-10  #加一个很小的数字避免出现除以0的情况
        self.distanceMatrix = np.sqrt(np.sum(np.square(diff), axis=2))  #三维距离

        # 计算方向
        horizontal_distance = np.sqrt(np.square(diff[:, :, 0]) + np.square(diff[:, :, 1]))  #水平距离
        self.orientationMatrix = np.arccos(diff[:, :, 0] / (horizontal_distance + 1e-10)) * np.sign(diff[:, :, 1])

        # 计算俯仰角(-兀/2，兀/2)
        self.pitchMatrix = np.arctan2(diff[:, :, 2], horizontal_distance)  # z轴与水平距离的比值

        #self.orientationMatrix = np.pi + (np.arccos(diff[:, :, 0] / self.distanceMatrix) - np.pi) * np.sign(diff[:, :, 1])
        # self.orientationMatrix = np.arccos(diff[:, :, 0] / self.distanceMatrix) * np.sign(diff[:, :, 1])
        self.influenceMatrix = ((self.distanceMatrix - np.array(self.maxInfluenceDistanceList)) < 0)

        #self.detectionMatrix = \
        #    ((self.distanceMatrix - np.array(self.unitDetectionDistanceList)) < 0) \
        #    & ((np.abs(self.orientationMatrix - np.array(self.unitRadarDirectionList))
        #        - self.unitDetectionAngleList) < 0)

        #self.attackMatrix = ((self.distanceMatrix - np.array(self.unitAttackRangeList)) < 0) \
        #                    & self.detectionMatrix

    #计算两点之间的距离
    def get_distance(self, point1, point2):
        return np.sqrt(np.square(point2[0] - point1[0]) + np.square(point2[1] - point1[1])
                       + np.square(point2[2] - point1[2]))

    #计算两点连线的角度（弧度制）
    def get_orientation(self, point1, point2):
        distance = self.get_distance(point1, point2)

        # 计算x和y的差
        xDiff = point2[0] - point1[0]
        yDiff = point2[1] - point1[1]

        # 计算水平距离
        horizontal_distance = np.sqrt(xDiff ** 2 + yDiff ** 2)

        # 计算方向角（平面上的角度）
        orientation = np.arccos(xDiff / (horizontal_distance + 1e-10))  # 避免除以0
        if yDiff < 0:
            orientation = 2 * np.pi - orientation

        # 计算俯仰角
        zDiff = point2[2] - point1[2]
        pitch = np.arctan2(zDiff, horizontal_distance)

        return orientation, pitch

    def Pause(self):
        time.sleep(10)
        self.pause = False

    def simulator(self):
        self.initialize_links()
        #self.unitList[0].silence_on()
        num = 0

        env_picture = GetEnvironment(self.environment_name)
        picture = env_picture.unit_data['picture']

        img = Image.open(picture)

        tkimg = ImageTk.PhotoImage(img)

        EWAimg = ImageTk.PhotoImage(Image.open("../EWA.png"))

        for unit in self.unitList:
            if not unit.crashed:
                unit.display()
        #time.sleep()

        global_sum_efficiency = 0.0
        sim_step = 0

        while True:
            simNum = 0
            sumTime1 = 0
            sumTime2 = 0
            sumTime3 = 0
            sumTime4 = 0
            sumTime5 = 0
            sumTime6 = 0
            begin = time.time()
            sum_efficiency_per_frame = 0.0

            while simNum < self.simNumPerFrame:  # 运行的步数小于更新图像所需的步数，就继续运行
                sim_step += 1
                # 导弹行动
                begin1 = time.time()
                for missile in self.flyingMissileList:
                    missile.action()
                end1 = time.time()
                time1 = end1 - begin1
                sumTime1 += time1  # 记录导弹的行动时间，并累加到 sumTime1
                #print('missile_time =', time1)

                # 战场情况更新（距离、方向角矩阵）
                begin2 = time.time()
                self.battlefield_situation()
                end2 = time.time()
                time2 = end2 - begin2
                sumTime2 += time2
                #print('situation_time =', time2)

                # 干扰器行动
                begin4 = time.time()
                for unit in self.unitList:
                    if not unit.crashed and unit.jammer.on:
                        #print('unitIndex =', unit.unitIndex)
                        # if unit.name == 'fighter' and unit.color == 'red':

                        unit.jammer.jamming()
                end4 = time.time()
                time4 = end4 - begin4
                sumTime4 += time4
                #print('jamming_time =', time4)

                # 雷达检测
                begin6 = time.time()
                for unit in self.unitList:
                    if not unit.crashed and unit.radar.on:
                        unit.radar.detection()
                end6 = time.time()
                time6 = end6 - begin6
                sumTime6 += time6
                #print('radarTime =', end6-begin6)

                for unit in self.unitList:
                    if not unit.crashed and unit.monitor.on:
                        unit.monitor.monitoring()
                begin5 = time.time()
                self.communication()
                end5 = time.time()
                time5 = end5 - begin5
                sumTime5 += time5
                #print('linkTime =', end5-begin5)

                #for unit in self.unitList:
                #    if not unit.crashed and unit.communicator.on:
                #        unit.communicator.communication()

                begin3 = time.time()
                sum_efficiency_per_step = 0.0
                num_of_red_alive = 0
                for unit in self.unitList:
                    # 频谱利用率
                    unit.spectrum_efficiency = 0

                    if not unit.crashed:
                        #print('unitIndex =', unit.unitIndex)
                        unit.action()
                        #如果通信链路全断
                        if np.all(~self.linkMatrix[unit.unitIndex,:]):
                            unit.spectrum_efficiency += 0
                        else:
                            unit.spectrum_efficiency += 0.5
                        unit.radar.cal_detection_range()

                        if unit.radar.dis > 100 and unit.name == 'Destroyer':
                            unit.spectrum_efficiency += 0.5
                        elif unit.radar.dis > 200 and unit.name == 'EWA':
                            unit.spectrum_efficiency += 0.5
                        elif unit.radar.dis > 50 and unit.name != 'EWA' and unit.name != 'Destroyer':
                            unit.spectrum_efficiency += 0.5

                        if unit.color == 'red':
                            num_of_red_alive += 1
                            sum_efficiency_per_step += unit.spectrum_efficiency

                        print(unit.color + unit.name + str(unit.unitIndex) + "频谱利用率" + str(unit.spectrum_efficiency) +
                              "雷达探测距离" + str(unit.radar.dis))

                sum_efficiency_per_step /= num_of_red_alive

                sum_efficiency_per_frame += sum_efficiency_per_step
                global_sum_efficiency += sum_efficiency_per_step

                end3 = time.time()
                time3 = end3 - begin3
                sumTime3 += time3
                #print('action_time =', time3)
                #print('----------------------')

                self.timer += self.simTimeInterval
                simNum += 1
                num += 1
                #print('simNum =', simNum)
                #if num % 100 == 0:
                #    avgTime4 = sumTime4 / num
                #print('avgJammingTimeRatio =', avgTime4)

            sum_efficiency_per_frame /= self.simNumPerFrame


            end = time.time()
            runtime = end - begin

            sleepingTime = max(self.frameTimeInterval - runtime, 1e-10)
            sleepRatio = sleepingTime / self.frameTimeInterval
            #print('sleepRatio = ', sleepRatio)
            # if sleepRatio < 0.01:
            #     print('runTime =', runtime)
            #     print('missile_time =', sumTime1)
            #     print('situation_time =', sumTime2)
            #     print('jamming_time =', sumTime4)
            #     print('radar_time =', sumTime6)
            #     print('linkTime =', sumTime5)
            #     print('action_time =', sumTime3)
            #     print('----------------------')

            # time.sleep(sleepingTime)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tkimg)

            for unit in self.crashedUnitList:
                unit.display()
            for unit in self.unitList:
                if not unit.crashed:
                    unit.display()
                    unit.display_spectrum_efficiency()

            # # TODO 只有通信距离大于两单位之间的距离时，才会显示通信连接
            # for link in self.linkList:
            #     distance = self.get_distance(link.ends[0].position, link.ends[1].position)
            #     linkDistance = link.get_linkDistanceUnderJamming()
            #     # 调试信息：查看通信距离和两单位之间的距离
            #     print(f'通信距离：{linkDistance},距离：{distance}')
            #     if linkDistance >= distance:
            self.displayLinks()

            for missile in self.flyingMissileList:
                missile.display()

            sum_efficiency_averaged = global_sum_efficiency / sim_step
            self.canvas.create_text(100, 560, text="平均频谱利用率：{:.4f}".format(sum_efficiency_averaged), fill='red',
                                    font=('Arial', 12))
            self.canvas.create_text(100, 580, text="时隙频谱利用率：{:.4f}".format(sum_efficiency_per_frame), fill='red',
                                    font=('Arial', 12))


            self.window.update()
            if self.timer > self.simDuration:
                break
        self.window.mainloop()

        sum_efficiency_averaged = global_sum_efficiency / sim_step
        print(f'整局己方平均频谱利用率：{sum_efficiency_averaged}')

    def step(self):
        self.initialize_links()
        # self.unitList[0].silence_on()
        num = 0
        env_picture = GetEnvironment(self.environment_name)
        picture = env_picture.unit_data['picture']
        # print("pic", picture)
        img = Image.open(picture)

        tkimg = ImageTk.PhotoImage(img)

        EWAimg = ImageTk.PhotoImage(Image.open("../EWA.png"))

        for unit in self.unitList:
            if not unit.crashed:
                unit.display()
        # time.sleep()
        while True:
            simNum = 0
            sumTime1 = 0
            sumTime2 = 0
            sumTime3 = 0
            sumTime4 = 0
            sumTime5 = 0
            sumTime6 = 0
            begin = time.time()
            while simNum < self.simNumPerFrame:  # 运行的步数小于更新图像所需的步数，就继续运行
                # 导弹行动
                begin1 = time.time()
                for missile in self.flyingMissileList:
                    missile.action()
                end1 = time.time()
                time1 = end1 - begin1
                sumTime1 += time1  # 记录导弹的行动时间，并累加到 sumTime1
                # print('missile_time =', time1)

                # 战场情况更新（距离、方向角矩阵）
                begin2 = time.time()
                self.battlefield_situation()
                end2 = time.time()
                time2 = end2 - begin2
                sumTime2 += time2
                # print('situation_time =', time2)

                # 干扰器行动
                begin4 = time.time()
                for unit in self.unitList:
                    # print("self", self.unitList)
                    if not unit.crashed and unit.jammer.on:
                        # print('unitIndex =', unit.unitIndex)
                        unit.jammer.jamming()
                end4 = time.time()
                time4 = end4 - begin4
                sumTime4 += time4
                # print('jamming_time =', time4)

                # 雷达检测
                begin6 = time.time()
                for unit in self.unitList:
                    if not unit.crashed and unit.radar.on:
                        unit.radar.detection()
                end6 = time.time()
                time6 = end6 - begin6
                sumTime6 += time6
                # print('radarTime =', end6-begin6)

                for unit in self.unitList:
                    if not unit.crashed and unit.monitor.on:
                        unit.monitor.monitoring()
                begin5 = time.time()
                self.communication()
                end5 = time.time()
                time5 = end5 - begin5
                sumTime5 += time5
                # print('linkTime =', end5-begin5)

                # for unit in self.unitList:
                #    if not unit.crashed and unit.communicator.on:
                #        unit.communicator.communication()

                begin3 = time.time()
                for unit in self.unitList:
                    # print("list", self.unitList)
                    if not unit.crashed:
                        # print('unitIndex =', unit.unitIndex)
                        unit.action()
                end3 = time.time()
                time3 = end3 - begin3
                sumTime3 += time3
                # print('action_time =', time3)
                # print('----------------------')

                self.timer += self.simTimeInterval
                simNum += 1
                num += 1
                # print('simNum =', simNum)
                # if num % 100 == 0:
                #    avgTime4 = sumTime4 / num
                # print('avgJammingTimeRatio =', avgTime4)

            end = time.time()
            runtime = end - begin

            sleepingTime = max(self.frameTimeInterval - runtime, 1e-10)
            sleepRatio = sleepingTime / self.frameTimeInterval
            # print('sleepRatio = ', sleepRatio)
            if sleepRatio < 0.01:
                # print('runTime =', runtime)
                # print('missile_time =', sumTime1)
                # print('situation_time =', sumTime2)
                # print('jamming_time =', sumTime4)
                # print('radar_time =', sumTime6)
                # print('linkTime =', sumTime5)
                # print('action_time =', sumTime3)
                # print('----------------------')
                pass

            # time.sleep(sleepingTime)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tkimg)
            for unit in self.crashedUnitList:
                unit.display()
            for unit in self.unitList:
                if not unit.crashed:
                    unit.display()

            self.displayLinks()
            for missile in self.flyingMissileList:
                missile.display()
            self.window.update()
            if self.timer > self.simDuration:
                break
        self.window.mainloop()


class Network:
    def __init__(self,
                 battleField,
                 name=None,
                 type=None,
                 displayEnable=True,
                 ):
        self.battleField = battleField
        self.name = name
        self.type = type
        self.displayEnable = displayEnable
        self.nodeList = []
        self.linkList = []
        self.vision = []


class Radar:
    def __init__(self,
                 battleField,
                 name='radar',
                 type='radar',
                 unit=None,
                 beamAngleRange=120,
                 minPitch=-60,
                 maxPitch=60,
                 maxEffectiveDistance=100,
                 maxBeamNum=1,
                 beamPower=50 * 1e4,
                 gain=20 * 1e3,
                 #on=False,
                 crashed=False,
                 displayEnable=True,
                 # TODO 修改部分参数
                 noiseFigure=0.5,  # 噪声系数（dB）  #
                 lossFactor=0.5,  # 系统损耗
                 Threshold=0.2  # 阈值
                 ):
        self.battleField = battleField
        self.name = name
        self.type = type  #雷达类型
        self.unit = unit  #雷达所在单位
        self.ou = False
        self.maxEffectiveDistance = maxEffectiveDistance
        self.beamAngleRange = [beamAngleRange / 180 * np.pi]
        self.beamPitchRange = [minPitch / 180 * np.pi, maxPitch / 180 * np.pi]  #转换成弧度制
        self.maxBeamNum = maxBeamNum
        self.beamPower = beamPower
        self.gain = gain
        self.beamList = []

        self.directionList = []
        self.directionPitchList = []
        self.maxEffectiveDistanceList = []
        self.currentEffectiveDistanceList = []
        self.angleRangeList = []
        self.pitchRangeList = []
        self.centerFrequencyList = []
        self.bandwidthList = []  # TODO 新增属性，存储每个波束的带宽

        #self.directionList = []
        #self.maxEffectiveDistanceList = np.zeros([beamNum])
        #self.currentEffectiveDistanceList = np.zeros([beamNum])
        #self.angleRangeList = np.zeros([beamNum, 1])
        #self.frequencyRangeList = np.zeros([beamNum, 2])
        #self.centerFrequencyList = np.zeros([beamNum])
        #self.bandWidthList = np.zeros([beamNum])
        #self.sweepSpeedList = np.zeros([beamNum])
        #self.powerList = np.zeros([beamNum])

        self.jammedByBeamList = []
        self.on = False  #雷达是否开机
        self.crashed = crashed  #是否已被摧毁
        self.displayEnable = displayEnable
        self.schedule = []  #任务日程表
        #self.radarPlan = []   #雷达规划
        self.detectedUnits = []  #已探测到的单位
        self.image = None
        self.unit.equipments.append(self)

        self.noiseFigure = noiseFigure  # 噪声系数，以 dB 计算
        self.lossFactor = lossFactor  # 系统损耗
        self.Threshold = Threshold  # 阈值
        self.dis = 0 # 探测距离

        #for i in range(beamNum):
        #    radarBeam = Beam(battleField=battleField,
        #                     source=self,
        #                     type='radarBeam',
        #                     )
        #    radarBeam.set_centerFrequency(10 * 1e9)
        #    radarBeam.set_bandwidth(200 * 1e6)

    def cal_detection_range(self):
        k = 1.38e-23  # 玻尔兹曼常数
        T0 = 290  # 噪声温度，单位：开尔文
        co_channel_interference = 0

        for u in self.battleField.unitList:
            if not u.crashed and u != self.unit:
                for radarBeam in u.radar.beamList:
                    TargetOrientation = self.battleField.orientationMatrix[u.unitIndex, self.unit.unitIndex]
                    TargetPitch = self.battleField.pitchMatrix[u.unitIndex, self.unit.unitIndex]
                    TargetDistance = self.battleField.distanceMatrix[u.unitIndex, self.unit.unitIndex]
                    AimingBeamIndexList1 = np.where(
                        ((np.pi - np.abs(np.pi - np.abs(np.array(u.radar.directionList)[:, 0] - TargetOrientation))
                          - np.array(u.radar.angleRangeList)[:, 0] / 2) < 0)
                    )[0]
                    for beamIndex in AimingBeamIndexList1:
                        beam = self.beamList[beamIndex]
                        # 检查俯仰角范围
                        beamPitchRange = beam.pitchRange
                        if beamPitchRange[0] <= TargetPitch <= beamPitchRange[1]:
                            if u.radar.on and u.radar.centerFrequencyList[0] and u != self.unit:
                                if beam.centerFrequency == u.radar.centerFrequencyList[0]:
                                    path_loss_db = 20 * np.log10(beam.centerFrequency / 1e6) + 20 * np.log10(
                                        TargetDistance) + 32.4 \
                                                   - 10 * np.log10(1.23) - 10 * np.log10(0.5)
                                    path_loss = 10 ** (path_loss_db / 10)
                                    co_channel_interference += (beam.power / path_loss) * 0.005
            # 根据雷达方程计算信号功率
            RCS = 3  # 假设默认RCS为3平米
            # 计算目标对应的干扰功率 Pi_radar
            Pi_radar = 0
            if self.battleField.jammedDict and self.unit.unitIndex in self.battleField.jammedDict:
                # 遍历仅与当前目标相关的干扰波束
                if self.battleField.jammedDict[self.unit.unitIndex]:
                    interferer_centerFreq, interferer_bandwidth, interference_power = \
                        self.battleField.jammedDict[self.unit.unitIndex][
                            0]  # TODO 通过 targetIndex 从 self.battleField.jammedDict 中提取当前目标的干扰波束，避免使用所有干扰波束
                    for i, centerFrequency in enumerate(self.centerFrequencyList):
                        bandwidth = self.bandwidthList[i]
                        overlap_percentage = self.calculate_overlap_percentage(
                            interferer_centerFreq, interferer_bandwidth,
                            centerFrequency, bandwidth
                        )
                        Pi_radar += interference_power * overlap_percentage  # TODO 仅对当前目标累加与其频率范围有重叠的干扰波束功率

            # 计算噪声功率
            Pn = k * T0 * self.beamList[0].bandwidth * 10 ** (self.noiseFigure / 10)  # 噪声功率
            # 计算雷达极限探测距离
            self.dis = (
                               self.beamPower * (self.gain ** 2) * (
                                   (3e8 / self.beamList[0].centerFrequency) ** 2) * RCS / \
                               ((4 * np.pi) ** 3 * self.Threshold * (
                                           Pi_radar + co_channel_interference + Pn))
                       ) ** (1 / 4) * 1e-3
            # if self.unit.name == 'fighter' and self.unit.color == 'red':
            #     print(1)








    def calculate_overlap_percentage(self, centerFreq1, bandwidth1, centerFreq2, bandwidth2):
        range1_min = centerFreq1 - bandwidth1 / 2
        range1_max = centerFreq1 + bandwidth1 / 2
        range2_min = centerFreq2 - bandwidth2 / 2
        range2_max = centerFreq2 + bandwidth2 / 2

        # 计算重叠部分
        overlap_min = max(range1_min, range2_min)
        overlap_max = min(range1_max, range2_max)

        if overlap_min >= overlap_max:  # 没有重叠
            return 0.0
        overlap = overlap_max - overlap_min
        return overlap / bandwidth2


    #执行探测，计算战场上的目标是否能被探测到
    def detection(self):
        co_channel_interference = 0
        self.detectedUnits = []  # 每次检测前清空已检测到的目标列表
        k = 1.38e-23  # 玻尔兹曼常数
        T0 = 290  # 噪声温度，单位：开尔文
        targetIndexList = []

        for u in self.battleField.unitList:
            if not u.crashed and u != self.unit:
                for radarBeam in u.radar.beamList:
                    TargetOrientation = self.battleField.orientationMatrix[u.unitIndex, self.unit.unitIndex]
                    TargetPitch = self.battleField.pitchMatrix[u.unitIndex, self.unit.unitIndex]
                    TargetDistance = self.battleField.distanceMatrix[u.unitIndex, self.unit.unitIndex]
                    AimingBeamIndexList1 = np.where(
                        ((np.pi - np.abs(np.pi - np.abs(np.array(u.radar.directionList)[:, 0] - TargetOrientation))
                          - np.array(u.radar.angleRangeList)[:, 0] / 2) < 0)
                    )[0]
                    for beamIndex in AimingBeamIndexList1:
                        beam = self.beamList[beamIndex]
                        # 检查俯仰角范围
                        beamPitchRange = beam.pitchRange
                        if beamPitchRange[0] <= TargetPitch <= beamPitchRange[1]:
                            if u.radar.on and u.radar.centerFrequencyList[0] and u != self.unit:
                                if beam.centerFrequency == u.radar.centerFrequencyList[0]:
                                    path_loss_db = 20 * np.log10(beam.centerFrequency / 1e6) + 20 * np.log10(
                                        TargetDistance) + 32.4 \
                                                   - 10 * np.log10(1.23) - 10 * np.log10(0.5)
                                    path_loss = 10 ** (path_loss_db / 10)
                                    co_channel_interference += (beam.power / path_loss) * 0.005

        for targetIndex,target in enumerate(self.battleField.unitList):
            if not target.crashed and target.color != self.unit.color:
                targetIndexList.append(targetIndex)

            for targetIndex in targetIndexList:
                target = self.battleField.unitList[targetIndex]
                targetOrientation = self.battleField.orientationMatrix[self.unit.unitIndex, target.unitIndex]
                targetPitch = self.battleField.pitchMatrix[self.unit.unitIndex, target.unitIndex]
                targetDistance = self.battleField.distanceMatrix[self.unit.unitIndex, target.unitIndex]

                # 检查角度范围
                AimingBeamIndexList = np.where(
                    ((np.pi - np.abs(np.pi - np.abs(np.array(self.directionList)[:, 0] - targetOrientation))
                      - np.array(self.angleRangeList)[:, 0] / 2) < 0)
                )[0]

                for beamIndex in AimingBeamIndexList:
                    beam = self.beamList[beamIndex]

                    # 检查俯仰角范围
                    beamPitchRange = beam.pitchRange
                    if beamPitchRange[0] <= targetPitch <= beamPitchRange[1]:
                        if target.withMonitor and targetDistance <= target.monitor.maxMonitoringDistance:
                            target.monitor.recvBeamList.append(beam)

                        # 根据雷达方程计算信号功率
                        RCS = target.rcs if hasattr(target, 'rcs') else 3  # 假设默认RCS为3平米
                        Pr = (self.beamPower * (self.gain ** 2) * ((3e8 / beam.centerFrequency) ** 2) * RCS) / \
                             ((4 * np.pi) ** 3 * ((targetDistance * 1e3) ** 4))  # 信号功率

                        # 计算目标对应的干扰功率 Pi_radar
                        Pi_radar = 0
                        if self.battleField.jammedDict and self.unit.unitIndex in self.battleField.jammedDict:
                            # 遍历仅与当前目标相关的干扰波束
                            if self.battleField.jammedDict[self.unit.unitIndex]:
                                interferer_centerFreq, interferer_bandwidth, interference_power = \
                                    self.battleField.jammedDict[self.unit.unitIndex][
                                        0]  # TODO 通过 targetIndex 从 self.battleField.jammedDict 中提取当前目标的干扰波束，避免使用所有干扰波束
                                for i, centerFrequency in enumerate(self.centerFrequencyList):
                                    bandwidth = self.bandwidthList[i]
                                    overlap_percentage = self.calculate_overlap_percentage(
                                        interferer_centerFreq, interferer_bandwidth,
                                        centerFrequency, bandwidth
                                    )
                                    Pi_radar += interference_power * overlap_percentage   # TODO 仅对当前目标累加与其频率范围有重叠的干扰波束功率

                        # 计算噪声功率
                        Pn = k * T0 * beam.bandwidth * 10 ** (self.noiseFigure / 10)  # 噪声功率

                        SINR = Pr / (Pi_radar+ co_channel_interference + Pn)
                        # 调试信息
                        # print('Pn：{:}'.format(Pn))  # 查看噪声功率
                        # print('Pi_radar:{:}'.format(Pi_radar))  # 查看干扰功率

                        # print('SINR:{:.2f}'.format(SINR))  # 查看信干噪比

                        # 判断目标是否可见
                        if SINR >= self.Threshold:
                            beam.targetList.append(target)
                            self.detectedUnits.append(target)

                    # 计算雷达极限探测距离
                    # self.dis = (
                    #             self.beamPower * (self.gain ** 2) * ((3e8 / beam.centerFrequency) ** 2) * RCS / \
                    #                    ((4 * np.pi) ** 3 * self.Threshold * (Pi_radar + co_channel_interference + Pn))
                    #            ) ** (1 / 4) * 1e-3
                            # self.unit.navigator.set_path1([target.position])  # 红方飞机朝向蓝方飞机，（蓝方飞机不能朝向红方飞机）
                            # print("^^^^^^^^^^^^^^^^^^^^^")
                            # print("{:s}{:s} can see {:s}".format(self.unit.color, self.unit.name, target.name))
                            # print("^^^^^^^^^^^^^^^^^^^^^")
                        # elif target.color == 'blue' and SNR >= self.Threshold:
                        #     beam.targetList.append(target)
                        #     self.detectedUnits.append(target)
                        #     self.unit.goal = target.position
                        #     print("^^^^^^^^^^^^^^^^^^^^^")
                        #     print("{:s}{:s} can see {:s}".format(self.unit.color, self.unit.name, target.name))
                        #     print("^^^^^^^^^^^^^^^^^^^^^")

    def get_newRadarBeam(self, type, workMode, centerFrequency, bandwidth):
        if len(self.beamList) < self.maxBeamNum:
            newRadarBeam = Beam(battleField=self.battleField,
                                source=self,
                                type=type,
                                workMode=workMode,

                                angleRange=self.beamAngleRange,
                                pitchRange=self.beamPitchRange,
                                centerFrequency=centerFrequency,
                                bandwidth=bandwidth,
                                power=self.beamPower,
                                )
            self.beamList.append(newRadarBeam)
            self.directionList.append(newRadarBeam.direction)
            self.directionPitchList.append(newRadarBeam.direction_pitch)

            self.angleRangeList.append(newRadarBeam.angleRange)
            self.pitchRangeList.append(newRadarBeam.pitchRange)
            self.centerFrequencyList.append(newRadarBeam.centerFrequency)
            self.bandwidthList.append(newRadarBeam.bandwidth)  # TODO 更新带宽列表
        else:
            print('Reach MaxRadarBeamNum!')
            newRadarBeam = None
        return newRadarBeam

    def turn_on(self):
        self.on = True
        newRadarBeam = self.get_newRadarBeam(type='radarBeam',
                                             workMode='fix',
                                             centerFrequency=2 * 1e9,
                                             bandwidth=200 * 1e6,
                                             )

    def turn_off(self):
        self.on = False
        for beam in self.beamList:
            beam.close()

    def crash(self):
        self.crashed = True

    def display(self):
        if self.on and self.displayEnable:
            for beam in self.beamList:
                beam.display()


#电磁波
class Beam:
    def __init__(self,
                 battleField,
                 type=None,
                 source=None,
                 workMode=None,
                 maxEffectiveDistance=None,
                 angleRange=None,
                 pitchRange=None,
                 centerFrequency=None,
                 bandwidth=None,
                 sweepSpeed=None,
                 power=None,

                 ):
        self.battleField = battleField
        self.unit = source.unit
        self.type = type
        self.source = source  #辐射源
        self.on = True
        self.lockedBeamList = []
        self.lockedUnitList = []  #被干扰的单位，代替lockedBeamList
        self.lockedByBeamList = []
        self.targetBeamList = []
        self.jammedByBeamList = []

        self.workMode = workMode
        self.direction = self.unit.moveDirection
        # self.source.directionList.append(self.direction)
        # 方向加上俯仰角
        self.direction_pitch = self.unit.moveDirectionPitch

        self.maxEffectiveDistance = maxEffectiveDistance
        #self.source.maxEffectiveDistanceList[self.beamIndex] = self.maxEffectiveDistance

        self.currentEffectiveDistance = maxEffectiveDistance
        #self.source.currentEffectiveDistanceList[self.beamIndex] = self.currentEffectiveDistance

        self.angleRange = angleRange
        #self.source.angleRangeList[self.beamIndex] = self.angleRange
        self.pitchRange = pitchRange  #每个雷达束有一个 pitchRange 属性（形式为 [minPitch, maxPitch]）

        self.centerFrequency = centerFrequency
        #self.source.centerFrequencyList[self.beamIndex] = self.centerFrequency

        self.bandwidth = bandwidth
        #self.source.bandWidthList[self.beamIndex] = self.bandwidth

        self.sweepSpeed = sweepSpeed
        #self.source.sweepSpeedList[self.beamIndex] = self.sweepSpeed

        self.power = power
        #self.source.powerList[self.beamIndex] = self.power

        self.jammedByBeamList = []
        self.targetList = []
        self.beamPlan = []

    def set_workMode(self, mode):
        self.workMode = mode

    def set_type(self, type):
        self.type = type

    # 设置目标方向，再加上俯仰角方向
    def set_direction(self, direction):
        self.direction = direction
        beamIndex = self.source.beamList.index(self)
        self.source.directionList[beamIndex] = direction

    def set_direction_pitch(self, pitch):
        self.direction_pitch = pitch
        beamIndex = self.source.beamList.index(self)
        self.source.directionPitchList[beamIndex] = pitch

    # def set_maxEffectiveDistance(self, distance):
    #     self.maxEffectiveDistance = distance
    #     beamIndex = self.source.beamList.index(self)
    #     self.source.maxEffectiveDistanceList[beamIndex] = distance
    #
    # # 战斗机和我方雷达机当前有效探测距离
    # def set_currentEffectiveDistance(self, distance):
    #     self.currentEffectiveDistance = distance
    #     beamIndex = self.source.beamList.index(self)
    #     self.source.currentEffectiveDistanceList[beamIndex] = distance

    def set_angleRange(self, angleRange):
        self.angleRange[0] = angleRange
        beamIndex = self.source.beamList.index(self)
        self.source.angleRangeList[beamIndex][0] = angleRange / 180 * np.pi

    # 设置俯仰角
    def set_pitchRange(self, pitchRange):
        self.pitchRange = pitchRange
        beamIndex = self.source.beamList.index(self)
        pitchRange1 = [item / 180 * np.pi for item in pitchRange]  #抓换成弧度制
        self.source.pitchRangeList[beamIndex][0] = pitchRange1

    def set_centerFrequency(self, frequency):
        self.centerFrequency = frequency

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth

    def set_sweepSpeed(self, sweepSpeed):
        self.sweepSpeed = sweepSpeed

    def set_power(self, power):
        self.power = power

    # def set_effectiveDistanceUnderJamming(self):
    #     self.set_currentEffectiveDistance(self.maxEffectiveDistance)
    #     if self.jammedByBeamList:
    #         jammingSource = self.jammedByBeamList[0].unit
    #         jammingSourceDistance = self.battleField.distanceMatrix[self.unit.unitIndex, jammingSource.unitIndex]
    #         self.set_currentEffectiveDistance(max(20, 0.618*self.maxEffectiveDistance*np.square(jammingSourceDistance/self.jammedByBeamList[0].maxEffectiveDistance)))

    def close(self):
        #print('closed BeamUnit =', self.unit.unitIndex)
        # for lockedBeam in self.lockedBeamList:
        # #    print('lockeBeamUnit =', lockedBeam.unit.unitIndex)
        #     lockedBeam.lockedByBeamList.remove(self)
        # for lockedByBeam in self.lockedByBeamList:
        #     lockedByBeam.lockedBeamList.remove(self)

        # （self干扰别人）关闭被干扰单位的被干扰波束列表的干扰波
        for lockedUnit in self.lockedUnitList:
            lockedUnit.lockedByBeamList.remove(self)
            if lockedUnit.unitIndex in self.battleField.jammedDict:
                del self.battleField.jammedDict[lockedUnit.unitIndex]
        # （别人干扰self）关闭自己被干扰波束列表的干扰波
        for lockedByBeam in self.lockedByBeamList:
            lockedByBeam.lockedUnitList.remove(self.unit)
            if lockedUnit.unitIndex in self.battleField.jammedDict:
                del self.battleField.jammedDict[lockedUnit.unitIndex]
            # del self.battleField.jammedDict[self.unit.unitIndex]

        self.on = False
        beamIndex = self.source.beamList.index(self)
        del self.source.beamList[beamIndex]
        del self.source.directionList[beamIndex]

        del self.source.angleRangeList[beamIndex]
        del self.source.pitchRangeList[beamIndex]
        if self.source.type == 'radar':
            del self.source.centerFrequencyList[beamIndex]
        elif self.source.type == 'communicator':
            del self.source.recvCenterFrequencyList[beamIndex]

    def beam_planing(self):
        pass

    # 用于雷达波束锁定目标
    # 没用到该函数
    def lock_targetUnit(self, target):
        if not target.crashed and target in self.unit.radar.detectedUnits:
            self.set_direction([self.battleField.orientationMatrix[self.unit.unitIndex, target.unitIndex]])
            self.set_direction_pitch([self.battleField.pitchMatrix[self.unit.unitIndex, target.unitIndex]])
            self.unit.add_schedule(func=self.lock_targetUnit,
                                   args=(target,),
                                   delay=1,
                                   )
        else:
            # self.direction = self.unit.moveDirection
            # self.direction_pitch = self.unit.moveDirectionPitch
            self.close()
            # print('aaaaaaaaaaa')

    # 用于干扰波束锁定目标波束
    # def lock_targetBeam(self, targetBeam):#todo
    #     if not targetBeam.unit.crashed and targetBeam.unit in self.unit.radar.detectedUnits:
    #         self.set_direction([self.battleField.orientationMatrix[self.unit.unitIndex, targetBeam.unit.unitIndex]])
    #         self.set_direction_pitch([self.battleField.pitchMatrix[self.unit.unitIndex, targetBeam.unit.unitIndex]])
    #         self.unit.add_schedule(func=self.lock_targetBeam,
    #                                args=(targetBeam,),
    #                                delay=1,
    #                                )
    #     else:
    #         # targetBeam.lockedByBeamList.remove(self)
    #         self.close()
    #         # targetBeam.jammed = False

    def display(self):
        if self.source.displayEnable and self.source.on:
            if self.type == 'radarBeam':
                fillColor = None
            elif self.type == 'radarJammingBeam':
                fillColor = 'LightGray'
            elif self.type == 'linkJammingBeam':
                fillColor = 'Aqua'

            # 获取画布的尺寸
            canvas_width = battleField.canvas.winfo_width()
            canvas_height = battleField.canvas.winfo_height()

            # 使用一个大的值来模拟无限远（这里采用画布的对角线长度的两倍）
            radius = (canvas_width ** 2 + canvas_height ** 2) ** 0.5 * 2

            # 计算圆或圆弧的边界
            position_x, position_y = self.source.unit.position[:2]
            bounds = (position_x - radius, position_y - radius, position_x + radius, position_y + radius)
            if self.unit.color == 'red' or self.unit.color == 'blue':
                if self.angleRange[0] < (2 * np.pi):
                    # 绘制圆弧
                    self.image = battleField.canvas.create_arc(
                        bounds,
                        start=(-self.direction[0] - self.angleRange[0] / 2) / np.pi * 180,
                        extent=self.angleRange[0] / np.pi * 180,
                        fill=fillColor,
                        outline=fillColor,
                    )
                else:
                    # 绘制完整的圆
                    self.image = battleField.canvas.create_oval(
                        bounds,
                        fill=fillColor,
                        outline=fillColor,
                    )


#干扰
class Jammer:
    def __init__(self,
                 battleField,
                 name='jammer',
                 type='jammer',
                 unit=None,
                 beamPower=1 * 1e6,
                 maxEffectiveDistance=0,
                 beamAngleRange=0,
                 beamPitchRange=None,
                 gain=1000,
                 polarLoss=0,
                 maxBeamNum=2,
                 displayEnable=True,
                 ):
        if beamPitchRange is None:
            beamPitchRange = []
        self.battleField = battleField
        self.name = name
        self.type = type
        self.unit = unit
        self.beamPower = beamPower
        self.maxEffectiveDistance = maxEffectiveDistance
        self.beamAngleRange = [beamAngleRange / 180 * np.pi]
        self.beamPitchRange = [item * np.pi / 180 for item in beamPitchRange]  #形式为[minPitch, maxPitch]弧度制
        self.gain = gain
        self.polarLoss = polarLoss
        self.maxBeamNum = maxBeamNum
        self.beamList = []
        self.targetBeamList = []
        self.targetUnitList = []
        self.on = True
        self.displayEnable = displayEnable

        self.directionList = []
        self.directionPitchList = []
        self.maxEffectiveDistanceList = []
        self.currentEffectiveDistanceList = []
        self.angleRangeList = []
        self.pitchRangeList = []

        self.crashed = False
        self.unit.equipments.append(self)

    def set_maxEffectiveDistance(self, distance):
        self.maxEffectiveDistance = distance

    def set_beamAngleRange(self, angle):  #有默认值，可以不用这个函数
        self.beamAngleRange[0] = angle / 180 * np.pi

    def set_beamPitchRange(self, pitch):  #同上
        self.beamPitchRange = [item * np.pi / 180 for item in pitch]

    def crash(self):
        self.crashed = True
        while self.beamList:
            self.beamList[0].close()

    def set_on(self, state):
        self.on = state

    def receiver_jamming_power(self, beamIndex, path_loss):
        beam = self.beamList[beamIndex]
        receiverJammingPower = beam.power * self.gain / path_loss  # 使用路径损耗来计算接收干扰功率
        return receiverJammingPower  #todo:判断中心频率和带宽的关系

    def get_newJammingBeam(self, type, workMode, centerFrequency, bandwidth):
        if len(self.beamList) < self.maxBeamNum:
            newJammingBeam = Beam(battleField=self.battleField,
                                  source=self,
                                  type=type,
                                  workMode=workMode,
                                  angleRange=self.beamAngleRange,
                                  pitchRange=self.beamPitchRange,
                                  centerFrequency=centerFrequency,
                                  bandwidth=bandwidth,
                                  power=self.beamPower,
                                  )
            self.beamList.append(newJammingBeam)
            self.directionList.append(newJammingBeam.direction)
            self.directionPitchList.append(newJammingBeam.direction_pitch)

            self.angleRangeList.append(newJammingBeam.angleRange)
            self.pitchRangeList.append(newJammingBeam.pitchRange)
        else:
            print('Reach MaxBeamNum!')
            newJammingBeam = None
        return newJammingBeam

    # def jam_targetBeam(self, targetBeam):
    #     #print('%s\'s jammedBeamNum =' % targetBeam.unit.unitIndex, len(targetBeam.jammedByBeamList))
    #
    #     if not targetBeam.lockedByBeamList:#检查 targetBeam 是否已经被其他波束锁定（干扰）。如果 lockedByBeamList 列表为空，则表示当前没有其他波束锁定该目标波束。
    #         if targetBeam.type == 'radarBeam':
    #             if targetBeam.workMode == 'fix':
    #                 newJammingBeam = self.get_newJammingBeam(type='radarJammingBeam',
    #                                                          workMode='fix',
    #                                                          centerFrequency=targetBeam.centerFrequency,
    #                                                          bandwidth=200*1e6,
    #                                                          )
    #                 if newJammingBeam:
    #                     targetBeam.lockedByBeamList.append(newJammingBeam)
    #                     newJammingBeam.lockedBeamList.append(targetBeam)
    #                     newJammingBeam.lock_targetBeam(targetBeam)
    #                     #print('detectedUnits_before =',)
    #                     #for unitIndex in targetBeam.unit.network:
    #                     #    unit = self.battleField.unitList[unitIndex]
    #                     #    print('%s\'s dectectedUnits =' % unitIndex, end='')
    #                     #    for detectedUnit in unit.radar.detectedUnits:
    #                     #        print(detectedUnit.unitIndex, end=',')
    #                     #    print('\n')
    #
    #                     #if self.unit in targetBeam.unit.radar.detectedUnits:
    #                     #    targetBeam.unit.radar.detectedUnits.remove(self.unit)
    #
    #                     #print('detectedUnits_after =',)
    #                     #for unitIndex in targetBeam.unit.network:
    #                     #    unit = self.battleField.unitList[unitIndex]
    #                     #    print('%s\'s dectectedUnits =' % unitIndex, end='')
    #                     #    for detectedUnit in unit.radar.detectedUnits:
    #                     #        print(detectedUnit.unitIndex, end=',')
    #                     #    print('\n')
    #                     #print('------------------------------------')
    #
    #             elif targetBeam.workMode == 'hop':
    #                 pass
    #
    #         elif targetBeam.type == 'linkBeam':
    #             if targetBeam.workMode == 'fix':
    #                 newJammingBeam = self.get_newJammingBeam(type='linkJammingBeam',
    #                                                          workMode='fix',
    #                                                          centerFrequency=targetBeam.centerFrequency,
    #                                                          bandwidth=1*1e6,
    #                                                          )
    #                 if newJammingBeam:
    #                     targetBeam.lockedByBeamList.append(newJammingBeam)
    #                     newJammingBeam.lockedBeamList.append(targetBeam)
    #                     newJammingBeam.lock_targetBeam(targetBeam)
    #             elif targetBeam.workMode == 'hop':
    #                 pass
    #     else:
    #         pass
    #
    #         #self.on = True

    def jam_targetUnit(self, targetUnit):
        #print('targetUnit =', targetUnit.unitIndex)
        # if self.battleField.distanceMatrix[self.unit.unitIndex, targetUnit.unitIndex] <= self.maxEffectiveDistance:
        # 取消干扰距离约束，只要雷达探测到即可干扰
        # if targetUnit in self.unit.radar.detectedUnits and self.on:# 判断目标是否在雷达探测范围内，且有干扰能力
        #     for radarBeam in targetUnit.radar.beamList:
        #         self.jam_targetBeam(radarBeam)
        #     for link in targetUnit.communicator.recvLinkList:
        #         self.jam_targetBeam(link)
        if targetUnit in self.unit.radar.detectedUnits and self.on:
            if not targetUnit.lockedByBeamList:  # 检查 targetUnit 是否已经被干扰波束锁定。
                newRadarJammingBeam = self.get_newJammingBeam(type='radarJammingBeam',
                                                              workMode='fix',
                                                              centerFrequency=2e9,  #随便设
                                                              bandwidth=2 * 1e6,  #随便设
                                                              )
                if newRadarJammingBeam:
                    targetUnit.lockedByBeamList.append(newRadarJammingBeam)
                    newRadarJammingBeam.lockedUnitList.append(targetUnit)
                    newRadarJammingBeam.lock_targetUnit(targetUnit)

                newLinkJammingBeam = self.get_newJammingBeam(type='linkJammingBeam',
                                                             workMode='fix',
                                                             centerFrequency=1000 * 1e6,  #随便设
                                                             bandwidth=200 * 1e6,  #随便设
                                                             )
                if newLinkJammingBeam:
                    targetUnit.lockedByBeamList.append(newLinkJammingBeam)
                    newLinkJammingBeam.lockedUnitList.append(targetUnit)
                    newLinkJammingBeam.lock_targetUnit(targetUnit)
            else:
                pass

    def calculate_overlap_percentage(self, centerFreq1, bandwidth1, centerFreq2, bandwidth2):
        range1_min = centerFreq1 - bandwidth1 / 2
        range1_max = centerFreq1 + bandwidth1 / 2
        range2_min = centerFreq2 - bandwidth2 / 2
        range2_max = centerFreq2 + bandwidth2 / 2

        # 计算重叠部分
        overlap_min = max(range1_min, range2_min)
        overlap_max = min(range1_max, range2_max)

        if overlap_min >= overlap_max:  # 没有重叠
            return 0.0
        overlap = overlap_max - overlap_min
        return overlap / bandwidth2

    def jamming(self):
        # 获取雷达检测到的目标列表
        targetList = self.unit.radar.detectedUnits

        for target in targetList:
            targetOrientation = self.battleField.orientationMatrix[self.unit.unitIndex, target.unitIndex]
            targetPitch = self.battleField.pitchMatrix[self.unit.unitIndex, target.unitIndex]
            targetDistance = self.battleField.distanceMatrix[self.unit.unitIndex, target.unitIndex]

            if not target.crashed and target.color != self.unit.color and self.beamList:
                # 计算哪些干扰束能够对目标产生影响
                aimingBeamIndexList = np.where(
                    ((np.pi - np.abs(np.pi - np.abs(np.array(self.directionList)[:, 0] - targetOrientation))
                      - np.array(self.angleRangeList)[:, 0] / 2) < 0)
                )[0]

                for beamIndex in aimingBeamIndexList:
                    jammingBeam = self.beamList[beamIndex]

                    # 检查俯仰角范围
                    jammingBeamPitchRange = jammingBeam.pitchRange
                    if jammingBeamPitchRange[0] <= targetPitch <= jammingBeamPitchRange[1]:
                        # 建立自由空间路损模型
                        targetFrequency = jammingBeam.centerFrequency  # F in Hz
                        path_loss_db = 20 * np.log10(targetFrequency / 1e6) + 20 * np.log10(targetDistance) + 32.4 - \
                                       10 * np.log10(1.23) - 10 * np.log10(0.5)
                        path_loss = 10 ** (path_loss_db / 10)  # 将 dB 转换为线性单位
                        receiverJammingPower = self.receiver_jamming_power(beamIndex, path_loss)

                        # 更新干扰字典，确保干扰波束与目标一一对应
                        if target.unitIndex not in self.battleField.jammedDict:  # TODO jammedDict 按目标索引（target.unitIndex）存储
                            self.battleField.jammedDict[target.unitIndex] = []
                        if len(self.battleField.jammedDict[target.unitIndex]) < 2:  #TODO 暂时只考虑一个飞机施加的干扰波束
                            self.battleField.jammedDict[target.unitIndex].append([
                                jammingBeam.centerFrequency,
                                jammingBeam.bandwidth,
                                receiverJammingPower
                            ])
                        # 受扰字典的结构：{target.unitIndex：[[fc1,Bw1,Pi1],[fc2,Bw2,Pi2],...],...}，列表中存储了所有对该目标产生干扰的干扰波束信息（中心频率、带宽、接收功率）

                        # 针对雷达干扰
                        if jammingBeam.type == 'radarJammingBeam':
                            jammedRadarBeamIndexList = []
                            for i, centerFrequency in enumerate(target.radar.centerFrequencyList):
                                overlap_percentage = self.calculate_overlap_percentage(
                                    centerFrequency, target.radar.bandwidthList[i],
                                    jammingBeam.centerFrequency, jammingBeam.bandwidth
                                )
                                if overlap_percentage > 0:  # todo：干扰到了才添加到受扰字典中
                                    jammedRadarBeamIndexList.append(i)
                            for beamIndex in jammedRadarBeamIndexList:
                                jammedRadarBeam = target.radar.beamList[beamIndex]
                                if jammingBeam not in jammedRadarBeam.jammedByBeamList:
                                    jammedRadarBeam.jammedByBeamList.append(jammingBeam)

                        # 针对通信干扰
                        elif jammingBeam.type == 'linkJammingBeam':
                            jammedLinkIndexList = []
                            for i, recvCenterFrequency in enumerate(target.communicator.recvCenterFrequencyList):
                                overlap_percentage = self.calculate_overlap_percentage(
                                    recvCenterFrequency, target.communicator.recvbandwidthList[i],
                                    jammingBeam.centerFrequency, jammingBeam.bandwidth
                                )
                                if overlap_percentage > 0:
                                    jammedLinkIndexList.append(i)
                            for beamIndex in jammedLinkIndexList:
                                jammedLink = target.communicator.recvLinkList[beamIndex]
                                if jammingBeam not in jammedLink.jammedByBeamList:
                                    jammedLink.jammedByBeamList.append(jammingBeam)
                                self.battleField.jammedLinkList.append(jammedLink)  # 更新被干扰的连接列表

                        # 监视目标的干扰
                        if target.withMonitor and targetDistance <= target.monitor.maxMonitoringDistance:
                            target.monitor.recvBeamList.append(jammingBeam)

    def display(self):
        if self.on:
            for beam in self.beamList:
                beam.display()


#无源监测设备
class Monitor:
    def __init__(self,
                 battleField,
                 name='monitor',
                 type='monitor',
                 unit=None,
                 sensitivity=0,
                 maxMonitoringDistance=100,
                 ):
        self.battleField = battleField
        self.name = name
        self.type = type
        self.unit = unit
        self.on = True
        self.recvBeamList = []
        self.sensitivity = sensitivity
        self.maxAttackableDistance = 100
        self.attackableTargetList = []
        self.maxMonitoringDistance = maxMonitoringDistance

    def set_maxMonitoringDistance(self, distance):
        self.maxMonitoringDistance = distance

    def set_sensitivity(self, value):
        self.sensitivity = value

    def get_recvBeamPower(self, beam):
        recvPower = 0.1
        return recvPower

    def monitoring(self):
        self.attackableTargetList = []
        if self.recvBeamList:
            self.recvBeamList = list(set(self.recvBeamList))
            for beam in self.recvBeamList:
                #self.unit.jammer.jam_targetBeam(beam)
                recvPower = self.get_recvBeamPower(beam)
                target = beam.unit
                targetDistance = self.battleField.distanceMatrix[self.unit.unitIndex, target.unitIndex]
                if recvPower >= self.sensitivity and targetDistance <= self.maxAttackableDistance:
                    self.attackableTargetList.append(beam.unit)
        self.recvBeamList = []
        self.attackableTargetList = list(set(self.attackableTargetList))

    # if self.jammer.on:
    #     for targetBeam in self.monitor.recvBeamList:
    #         self.jammer.jam_targetBeam(targetBeam)
    # print('%s\'s radar lockedByBeamList =' % targetBeam.unit.unitIndex,
    #       len(targetBeam.lockedByBeamList))
    #self.recvBeamList = []
    #if self.attackableTargetList:
    #    print('%d_attackableTargetList =' % self.unit.unitIndex, end='{')
    #    for target in self.attackableTargetList:
    #        print(target.unitIndex, end=', ')
    #    print('}')


class Link:
    def __init__(self,
                 battleField,
                 name='link',
                 index=None,
                 ends=None,
                 centerFrequency=int(909 * 1e6),
                 bandwidth=int(25 * 1e3),
                 mode='fixed',
                 workMode='fix',
                 type='linkBeam'):
        # TODO 修改跳频序列，与communicator类中的一样
        self.hopFrequencyList = (np.array([909, 912, 915, 918, 921, 924, 927, 930, 933, 936,
                                           939, 942, 945, 948, 993, 996, 999, 1002, 1005, 1053,
                                           1056, 1059, 1062, 1065, 1068, 1071, 1074, 1077, 1080, 1083,
                                           1086, 1089, 1092, 1095, 1098, 1101, 1104, 1107, 1110, 1113,
                                           1116, 1119, 1122, 1125, 1128, 1131, 1134, 1137, 1140, 1143,
                                           1146]) * 1e6).astype(np.int32)
        self.battleField = battleField
        self.name = name
        self.type = type
        self.unit = ends[0]
        self.workMode = workMode
        self.index = index
        self.ends = ends
        self.centerFrequency = centerFrequency
        self.bandwidth = bandwidth
        self.mode = mode
        self.lockedByBeamList = []

        # 确定链路类型并设置最大距离
        if self.ends[0].type == 'plane' and self.ends[1].type == 'plane':
            self.linkType = 'air2air'
            self.maxLinkDistance = 560
        elif self.ends[0].type != 'plane' and self.ends[1].type != 'plane':
            self.linkType = 'surface2surface'
            self.maxLinkDistance = 280
        else:
            self.linkType = 'air2surface'
            self.maxLinkDistance = 50

        self.linkDistance = self.calculate_distance()
        self.battleField.maxLinkDistanceMatrix[self.ends[0].unitIndex,
        self.ends[1].unitIndex] = self.linkDistance
        if self.battleField.get_distance(self.ends[0].position, self.ends[1].position) <= self.linkDistance:
            self.on = True
        else:
            self.on = False
        self.jammed = False
        self.hopPlan = []
        self.jammedByBeamList = []

    def calculate_overlap_percentage(self, centerFreq1, bandwidth1, centerFreq2, bandwidth2):
        range1_min = centerFreq1 - bandwidth1 / 2
        range1_max = centerFreq1 + bandwidth1 / 2
        range2_min = centerFreq2 - bandwidth2 / 2
        range2_max = centerFreq2 + bandwidth2 / 2

        # 计算重叠部分
        overlap_min = max(range1_min, range2_min)
        overlap_max = min(range1_max, range2_max)

        if overlap_min >= overlap_max:  # 没有重叠
            return 0.0
        overlap = overlap_max - overlap_min
        return overlap / bandwidth2  # 相对于干扰波束的带宽

    def calculate_distance(self):
        k = 1.38e-23  # 玻尔兹曼常数 (单位：J/K)
        T0 = 290  # 噪声温度 (单位：K)

        # 参数设置
        lossFactor = 0.5  # 系统损耗
        noiseFigure = 0.5  # 噪声因子（dB）
        Threshold = 0.8  # SINR 阈值
        beamPower = 50  # 雷达发射功率 (单位：W)
        gain = 5  # 接收增益 (单位：dB)

        # 固定参数
        RCS = 3  # 雷达截面积 (单位：m^2)
        c = 3e8  # 光速 (单位：m/s)

        # 噪声功率
        Pn = k * T0 * self.bandwidth * 10 ** (noiseFigure / 10)
        Pn += 1e-14

        # 初始化通信干扰功率
        Pi_link = 0

        # 调试信息只打印蓝方
        # if self.unit.color == 'blue':
        #     print("==============================")
        #     print(self.unit.color + self.unit.name + str(
        #         self.unit.unitIndex) + f" 此时的中心频率: {self.centerFrequency} Hz")
        #     print(self.battleField.jammedDict)
        #     print("==============================")

        # 计算干扰功率
        if self.battleField.jammedDict and self.unit.unitIndex in self.battleField.jammedDict:
            if self.battleField.jammedDict[self.unit.unitIndex][1]:
                # 遍历仅与当前单位相关的干扰波束
                interferer_centerFreq, interferer_bandwidth, interference_power = \
                    self.battleField.jammedDict[self.unit.unitIndex][1]  # TODO 仅遍历与当前单位相关的干扰波束
                overlap_percentage = self.calculate_overlap_percentage(
                    self.centerFrequency, self.bandwidth,
                    interferer_centerFreq, interferer_bandwidth
                )
                Pi_link += interference_power * overlap_percentage

        # 有效噪声功率
        effective_noise = Pi_link + Pn
        # print(f'effective_noise: {effective_noise}')

        if effective_noise == 0:  # 避免除零错误
            raise ValueError("有效噪声功率为零，无法计算通信距离。")

        # 波长计算
        wavelength = c / self.centerFrequency

        # 通信距离计算
        linkDistance = (
                               (beamPower * (gain**2) * (wavelength ** 2)) /
                               ((4 * np.pi) ** 2 * Threshold * effective_noise)
                       ) ** 0.5 / 1000  # 单位：km

        # 更新通信距离
        self.unit.linkDistance = linkDistance

        # 打印通信距离
        # if self.unit.color == 'blue':
        #     print(self.unit.color + self.unit.name + str(self.unit.unitIndex) + f' 通信距离: {linkDistance}')

        return linkDistance

    def set_centerFrequency(self, frequency):
        self.centerFrequency = frequency

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth

    def set_hopPlan(self, plan):
        self.hopPlan = plan

    def set_jammed(self):
        self.jammed = True

    # todo: 获取干扰下的链路距离，应改为SINR计算
    def get_linkDistanceUnderJamming(self):
        # linkDistance = -1
        linkDistance = self.calculate_distance()
        # todo 调试信息：查看battleField.jammedDict
        # print("==============================")
        # print(battleField.jammedDict)
        # print("==============================")
        return linkDistance


class Communicator:
    def __init__(self,
                 battleField,
                 name='communicator',
                 type='communicator',
                 unit=None,
                 maxCommunicationDistance=0,
                 ):
        self.battleField = battleField
        self.name = name
        self.type = type
        self.unit = unit
        self.networkIndex = None
        self.maxCommunicationDistance = maxCommunicationDistance
        self.availableFrequencyIndexList = list(range(51))
        self.recvLinkList = []
        self.recvCenterFrequencyList = []
        self.recvbandwidthList = []  # TODO 新增属性
        self.sendLinkList = []
        self.linkedUnitList = []
        self.jammedLinkList = []
        self.connectingFriendList = []

        self.crashed = False
        self.on = True
        self.unit.equipments.append(self)

        self.hopFrequencyList = (np.array([909, 912, 915, 918, 921, 924, 927, 930, 933, 936,
                                           939, 942, 945, 948, 993, 996, 999, 1002, 1005, 1053,
                                           1056, 1059, 1062, 1065, 1068, 1071, 1074, 1077, 1080, 1083,
                                           1086, 1089, 1092, 1095, 1098, 1101, 1104, 1107, 1110, 1113,
                                           1116, 1119, 1122, 1125, 1128, 1131, 1134, 1137, 1140, 1143,
                                           1146]) * 1e6).astype(np.int32)

    def crash(self):
        self.crashed = True
        self.battleField.maxLinkDistanceMatrix[self.unit.unitIndex] = -1
        self.battleField.maxLinkDistanceMatrix[:, self.unit.unitIndex] = -1

    def set_on(self, state):
        self.on = state
        if not state:
            for link in self.linkList:
                link.on = False

    def initialize_linkFrequency(self, link):
        sender = link.ends[0]
        receiver = link.ends[1]
        validFrequencyList = list(self.hopFrequencyList)
        for recvLink in sender.communicator.recvLinkList:
            if recvLink.centerFrequency in validFrequencyList:
                validFrequencyList.remove(recvLink.centerFrequency)
        for recvLink in receiver.communicator.recvLinkList:
            if recvLink.centerFrequency in validFrequencyList:
                validFrequencyList.remove(recvLink.centerFrequency)
        link.set_centerFrequency(validFrequencyList[0])
        link.set_bandwidth(25 * 1e3)

    def build_link(self, receiver):
        link = Link(battleField=self.battleField,
                    index=len(self.sendLinkList),
                    ends=[self.unit, receiver],
                    )
        self.initialize_linkFrequency(link)
        freeFrequencyList = list(self.hopFrequencyList)
        # 遍历发送方接收链路，判断是否有空闲频率
        for recvLink in self.recvLinkList:
            if recvLink.centerFrequency in freeFrequencyList:
                freeFrequencyList.remove(recvLink.centerFrequency)
        # 遍历接收方接收链路，判断是否有空闲频率
        for recvLink in receiver.communicator.recvLinkList:
            if recvLink.centerFrequency in freeFrequencyList:
                freeFrequencyList.remove(recvLink.centerFrequency)

        # todo: 设置链路频率与带宽
        link.set_centerFrequency(freeFrequencyList[0])
        link.set_bandwidth(25 * 1e3)

        # 添加链路到发送方和接收方的链路列表
        self.sendLinkList.append(link)
        receiver.communicator.recvLinkList.append(link)
        receiver.communicator.recvCenterFrequencyList.append(link.centerFrequency)
        receiver.communicator.recvbandwidthList.append(link.bandwidth)
        self.battleField.linkList.append(link)
        #print('link1 =', [link.ends[0].unitIndex, link.ends[1].unitIndex, link.centerFrequency])

    def communication(self):
        for linkedUnit in self.linkedUnitList:
            self.unit.radar.detectedUnits = set.union(linkedUnit.radar.detectedUnits,
                                                      self.unit.radar.detectedUnits)

    def display(self):
        pass


class Navigator:
    def __init__(self,
                 battleField,
                 name='navigator',
                 type=None,
                 unit=None,
                 speed=0,
                 acc=0,
                 autoNavigation=False,
                 displayEnable=True,
                 #path=[],
                 ):
        self.battleField = battleField
        self.name = name
        self.type = type
        self.unit = unit
        self.autoNavigation = autoNavigation
        self.crashed = False
        self.displayEnable = displayEnable
        self.path = []
        self.movePlan = []
        self.unit.equipments.append(self)
        self.arg = 1
        self.arg_acc = 1
        self.speed = speed
        self.acc = acc
        #三维DWA算法
        self.dt = battleField.simTimeInterval
        # #各轴最大速度
        # self.v_max = [self.unit.speed*self.arg, self.unit.speed*self.arg, self.unit.speed*self.arg]
        # #各轴最小速度
        # self.v_min = [-self.unit.speed*self.arg, -self.unit.speed*self.arg, -self.unit.speed*self.arg] # km/s
        # #各轴加速度
        # self.a_max = [self.unit.acc*self.arg_acc, self.unit.acc*self.arg_acc, self.unit.acc*self.arg_acc]

        #各轴最大速度
        self.v_max = [self.speed * self.arg, self.speed * self.arg, self.speed * self.arg]
        #各轴最小速度
        self.v_min = [-self.speed * self.arg, -self.speed * self.arg, -self.speed * self.arg]  # km/s
        #各轴加速度
        self.a_max = [self.acc * self.arg_acc, self.acc * self.arg_acc, self.acc * self.arg_acc]

        #采样分辨率
        self.v_sample = 0.01  # 每隔0.01km/s取一组速度值
        self.predict_time = 2
        # 轨迹评价函数系数
        self.alpha = 120.0  # 方位角评价函数
        # self.beta = 500.0
        self.gamma = 10  # 速度评价

    def __cal_vel_limit(self):
        return self.v_min, self.v_max

    def __cal_accel_limit(self, v):
        v_low = [v[i] - self.a_max[i] * self.dt for i in range(len(v))]
        v_high = [v[i] + self.a_max[i] * self.dt for i in range(len(v))]
        return v_low, v_high

    def cal_dynamic_window_vel(self, v):
        V1_min, V1_max = self.__cal_vel_limit()  # 速度本身限制
        V2_min, V2_max = self.__cal_accel_limit(v)  # 加速度限制

        V_min = [max([V1_min[i], V2_min[i]]) for i in range(len(V1_min))]
        V_max = [min([V1_max[i], V2_max[i]]) for i in range(len(V1_max))]
        return V_min, V_max

    def kinematicModel(self, state, v, dt):
        """
        机器人运动学模型
        :param state:状态量——x,y,z,vx,vy,vz
        :param v: [Vx,Vy,Vz]
        :param dt: 离散时间
        :return: 下一步的状态
        """
        state[0] += v[0] * dt
        state[1] += v[1] * dt
        state[2] += v[2] * dt
        state[3] = v[0]
        state[4] = v[1]
        state[5] = v[2]
        return state

    def trajectory_predict(self, state_init, v):
        state = np.array(state_init)
        trajectory = state
        time = 0
        # 在预测时间里
        while time <= self.predict_time:
            x = self.kinematicModel(state, v, self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt
        return trajectory

    # 利用三维向量点积计算运动方向与当前位置到目标终点的连线之间的夹角
    def __heading(self, trajectory, end_point):
        """
            计算运动方向向量和当前位置与终点连线的向量的夹角（弧度制）。

            参数：
            trajectory: 包含当前坐标 (x, y, z) 和运动方向向量 (vx, vy, vz) 的数组或列表。dim:n*6
            end_point: 终点坐标 (x, y, z) 的数组或列表。

            返回：
            heading: 夹角（弧度制）。
            """
        state = trajectory[-1]
        # 计算当前位置与终点的向量
        position_to_endpoint = np.array(end_point) - np.array(state[:3])

        # 运动方向向量
        velocity_vector = np.array(state[3:6])

        # 计算夹角
        dot_product = np.dot(position_to_endpoint, velocity_vector)
        norm_position_to_endpoint = np.linalg.norm(position_to_endpoint)
        norm_velocity_vector = np.linalg.norm(velocity_vector)

        # 使用反余弦函数计算角度
        theta = np.pi - np.arccos(dot_product / (norm_position_to_endpoint * norm_velocity_vector))

        # 确定 heading 的值
        if theta <= 0:
            heading = -theta
        else:
            heading = theta

        return heading

    # def angle_range_corrector(self,angle):
    #     if angle > np.pi:
    #         while angle > np.pi:
    #             angle -= 2 * np.pi
    #     elif angle < - np.pi:
    #         while angle < - np.pi:
    #             angle += 2 * np.pi
    #
    #     return angle

    def __velocity(self, trajecytory):
        """
        速度评价函数，表示当前速度大小，zxy三个方向速度的平方和表示
        :param trajecytory: dim n*6
        :return: 合速度
        """
        state = trajecytory[-1]
        vx = state[3]
        vy = state[4]
        vz = state[5]
        return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    def __distence(self, trajectory, end_point):
        state = trajectory[-1]
        location = [state[0], state[1], state[2]]
        distence = battleField.get_distance(location, end_point)
        if distence != 0:
            return 10 / distence
        else:
            return 10000

    def trajectory_evaluation(self, state, end_point):
        G_max = -float('inf')  # 最优评价
        trajectory_opt = state  # 最优轨迹
        v_opt = [0., 0., 0.]  # 最优速度
        v_now = [state[3], state[4], state[5]]
        dynamic_window_vel_min, dynamic_window_vel_max = self.cal_dynamic_window_vel(v_now)

        # sum_heading, sum_dis, sum_vel = 0, 0, 0 #各个评价之和
        sum_heading, sum_vel = 0, 0  # 各个评价之和
        for vx in np.arange(dynamic_window_vel_min[0], dynamic_window_vel_max[0], self.v_sample):
            for vy in np.arange(dynamic_window_vel_min[1], dynamic_window_vel_max[1], self.v_sample):
                for vz in np.arange(dynamic_window_vel_min[2], dynamic_window_vel_max[2], self.v_sample):
                    if np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) > self.unit.speed:
                        continue
                    v_total = [vx, vy, vz]
                    trajectory = self.trajectory_predict(state, v_total)

                    heading_eval = self.__heading(trajectory, end_point)
                    vel_eval = self.__velocity(trajectory)
                    # dis_eval = self.__distence(trajectory,end_point)
                    sum_vel += vel_eval
                    # sum_dis += dis_eval
                    sum_heading += heading_eval

        for vx in np.arange(dynamic_window_vel_min[0], dynamic_window_vel_max[0], self.v_sample):
            for vy in np.arange(dynamic_window_vel_min[1], dynamic_window_vel_max[1], self.v_sample):
                for vz in np.arange(dynamic_window_vel_min[2], dynamic_window_vel_max[2], self.v_sample):
                    if np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) > self.unit.speed:
                        continue
                    v_total = [vx, vy, vz]
                    trajectory = self.trajectory_predict(state, v_total)

                    heading_eval1 = self.alpha * self.__heading(trajectory, end_point) / sum_heading
                    vel_eval1 = self.gamma * self.__velocity(trajectory) / sum_vel
                    # dis_eval1 = self.beta * self.__distence(trajectory,end_point) / sum_dis

                    # G = vel_eval1 + dis_eval1 + heading_eval1
                    G = vel_eval1 + heading_eval1
                    if G_max < G:
                        G_max = G
                        trajectory_opt = trajectory
                        v_opt = [vx, vy, vz]
        return v_opt, trajectory_opt

    def dwa_control(self, state, goal):
        v, trajectory = self.trajectory_evaluation(state, goal)
        return v, trajectory

    def cal_heading(self, state):  #计算飞机当前二维朝向
        return np.arctan2(state[4], state[3])

    def set_path1(self, path):
        self.path = path

    def set_path(self, path):
        if not path or len(path) < 2:
            raise ValueError("Path must contain at least two points.")

        self.path = path
        self.movePlan = []
        stride = self.unit.speed * battleField.simTimeInterval
        startPoint = self.unit.position
        for i in range(len(path)):
            nextPointDistance = self.battleField.get_distance(startPoint, path[i])
            # 加入俯仰角变成三维
            moveDirection, moveDirectionPitch = self.battleField.get_orientation(startPoint, path[i])
            moveDirectionPitch1 = np.pi / 2 - moveDirectionPitch
            xStride = stride * np.sin(moveDirectionPitch1) * np.cos(moveDirection)
            yStride = stride * np.sin(moveDirectionPitch1) * np.sin(moveDirection)
            zStride = stride * np.cos(moveDirectionPitch1)
            moveNum = max(1, nextPointDistance // stride)
            # moveNum = nextPointDistance // stride
            self.movePlan.append([xStride,
                                  yStride,
                                  zStride,
                                  moveDirection,
                                  moveDirectionPitch1,
                                  moveNum,
                                  ])
            turningPoint = (startPoint[0] + xStride * moveNum,
                            startPoint[1] + yStride * moveNum,
                            startPoint[2] + zStride * moveNum
                            )
            residual = stride - nextPointDistance % stride
            # residual = nextPointDistance % stride
            if i != (len(path) - 1):
                next_moveDirection, next_moveDirectionPitch = self.battleField.get_orientation(path[i], path[i + 1])
                next_moveDirectionPitch1 = np.pi / 2 - next_moveDirectionPitch
                startPoint = (path[i][0] + residual * np.sin(next_moveDirectionPitch1) * np.cos(next_moveDirection),
                              path[i][1] + residual * np.sin(next_moveDirectionPitch1) * np.sin(next_moveDirection),
                              path[i][2] + residual * np.cos(next_moveDirectionPitch1))
                xStride = startPoint[0] - turningPoint[0]
                yStride = startPoint[1] - turningPoint[1]
                zStride = startPoint[2] - turningPoint[2]
                moveDirection, moveDirectionPitch = self.battleField.get_orientation(turningPoint, startPoint)
                self.movePlan.append([xStride,
                                      yStride,
                                      zStride,
                                      moveDirection,
                                      np.pi / 2 - moveDirectionPitch,
                                      1])
            else:
                xStride = path[i][0] - turningPoint[0]
                yStride = path[i][1] - turningPoint[1]
                zStride = path[i][2] - turningPoint[2]
                self.movePlan.append([xStride,
                                      yStride,
                                      zStride,
                                      moveDirection,
                                      np.pi / 2 - moveDirectionPitch,
                                      1])
            # startPoint = path[i]

    def crash(self):
        self.crashed = True

    def display(self):
        if self.displayEnable:
            pass


class Missile:
    def __init__(self,
                 battleField,
                 unit=None,
                 name=None,
                 type=None,
                 speed=0,
                 range=100,
                 anirange=200,
                 hitRate=0.8,
                 displayEnable=True,
                 ):
        self.battleField = battleField
        self.unit = unit
        self.name = name
        self.type = type
        self.color = unit.color
        self.attackChannel = None
        self.target = None
        self.position = unit.position.copy()
        self.speed = speed / 3600
        self.stride = self.speed * battleField.simTimeInterval
        self.moveDirection = [0]
        self.moveDirectionPitch = [0]
        self.range = range
        self.anirange = anirange
        self.hitRate = hitRate
        self.launched = False
        self.target = None
        self.crashed = False
        self.displayEnable = displayEnable
        self.unit.equipments.append(self)
        self.equipments = []
        self.image = None

    def launch(self):
        assert self.target
        self.position = self.unit.position.copy()
        self.launched = True
        self.unit.attackEnable = False
        self.unit.add_schedule(func=self.unit.set_attackEnable,
                               args=(True,),
                               delay=2,
                               )
        if self.type == 'normal':
            self.unit.normalMissileList.remove(self)
            print('%s launches normal missile to %s !' % (self.unit.unitIndex, self.target.unitIndex))
        elif self.type == 'antiRadiation':
            self.unit.antiRadiationMissileList.remove(self)
            print('%s launches antiRadiation missile!' % self.unit.unitIndex)
        self.unit.equipments.remove(self)
        self.battleField.flyingMissileList.append(self)

    def track(self):
        assert self.launched
        if self.target.crashed:
            self.crash()
        else:
            targetDistance = self.battleField.get_distance(self.position, self.target.position)
            if targetDistance <= self.stride:
                self.crash()
                if np.random.randint(0, 100) < (100 * self.hitRate):
                    print('%s destroyed %s' % (self.unit.unitIndex, self.target.unitIndex))
                    #print('%s\'s jammer is locking ' % self.target.unitIndex, end=',  ')
                    #for jammingBeam in self.target.jammer.beamList:
                    #    for lockedBeam in jammingBeam.lockedBeamList:
                    #        print(lockedBeam.unit.unitIndex, end=', ')
                    #print('\n')

                    #print('monitoringRecvBeamList =', len(self.target.monitor.recvBeamList))
                    self.target.crash()

                # print('%s\'s jammer is locking ' % self.target.unitIndex, end=',  ')
                # for jammingBeam in self.target.jammer.beamList:
                #     for lockedBeam in jammingBeam.lockedBeamList:
                #         print(lockedBeam.unit.unitIndex, end=', ')
                # print('\n')
                # print('-----------------------------------')

            else:
                targetOrientation, targetPitch = self.battleField.get_orientation(self.position, self.target.position)
                targetPitch1 = np.pi / 2 - targetPitch
                xStride = self.stride * np.sin(targetPitch1) * np.cos(targetOrientation)
                yStride = self.stride * np.sin(targetPitch1) * np.sin(targetOrientation)
                zStride = self.stride * np.cos(targetPitch1)
                self.position[0] += xStride
                self.position[1] += yStride
                self.position[2] += zStride
                self.moveDirection[0] = targetOrientation
                self.moveDirectionPitch[0] = targetPitch1

    def action(self):
        self.track()

    def crash(self):
        self.crashed = True
        if self.launched:
            self.battleField.flyingMissileList.remove(self)
            self.target.trackedByMissileList.remove(self)
            self.attackChannel.delete_missile(self)
            if len(self.attackChannel.missileList) == 0:
                self.attackChannel.reset()

    def display(self):  #只展示二维俯视图
        if self.launched and not self.crashed and self.displayEnable:
            self.image = battleField.canvas.create_oval(self.position[0] - 4,
                                                        self.position[1] - 4,
                                                        self.position[0] + 4,
                                                        self.position[1] + 4,
                                                        fill=self.color
                                                        )
            for equipment in self.equipments:
                equipment.display()


class AttackChannel:
    def __init__(self,
                 unit=None
                 ):
        self.unit = unit
        self.target = None
        self.missileList = []

    def set_target(self, target):
        assert not self.target
        self.target = target
        self.unit.availableAttackChannelNum -= 1

    def reset(self):
        assert not self.missileList
        self.target = None
        self.unit.availableAttackChannelNum += 1

    def add_missile(self, missile):
        self.missileList.append(missile)
        missile.attackChannel = self
        missile.target = self.target

    def delete_missile(self, missile):
        self.missileList.remove(missile)


#装备基类
class Unit:
    def __init__(self,
                 battleField,
                 name='newOb',
                 type='plane',
                 color='black',
                 goal=None,
                 position=None,
                 speed=0,
                 acc=0,
                 v_init=None,
                 moveDirection=0,
                 moveDirectionPitch=0,
                 maxInfluenceDistance=0,
                 withCommunicator=True,
                 withRadar=True,
                 withNavigator=True,
                 withJammer=True,
                 withMonitor=True,
                 displayEnable=True,
                 normalMissileNum=0,
                 antiRadiationMissileNum=0,
                 maxAttackChannelNum=1,
                 attackRange=150,
                 antiAttackRange=100,
                 rcs=3,
                 attackEnable=True,
                 ):
        self.battleField = battleField
        self.name = name
        self.type = type
        self.unitIndex = None
        self.color = color
        self.goal = goal
        self.position = position
        self.speed = speed / 3600
        self.acc = acc
        self.v_init = v_init
        self.state = position + v_init
        self.moveDirection = [moveDirection / 180 * np.pi]
        self.moveDirectionPitch = [moveDirectionPitch]
        self.maxInfluenceDistance = maxInfluenceDistance
        self.withCommunicator = withCommunicator
        self.network = []
        self.withRadar = withRadar
        self.withNavigator = withNavigator
        self.withJammer = withJammer
        self.withMonitor = withMonitor
        self.normalMissileNum = normalMissileNum
        self.antiRadiationMissileNum = antiRadiationMissileNum
        self.maxAttackChannelNum = maxAttackChannelNum
        self.availableAttackChannelNum = maxAttackChannelNum
        self.attackChannelList = []
        for i in range(self.maxAttackChannelNum):
            self.attackChannelList.append(AttackChannel(unit=self))
        self.attackRange = attackRange
        self.antiAttackRange = antiAttackRange
        self.rcs = rcs
        self.attackEnable = attackEnable
        self.trackedByMissileList = []
        self.detectedBy = []
        self.targetList = []
        self.aimingTargetList = []
        self.crashed = False
        self.fightingFlag = False
        self.guardFlag = False
        self.targetedByList = []
        self.displayEnable = displayEnable
        self.equipments = []
        self.linkDistance = 0
        # 频谱利用率
        self.spectrum_efficiency = 0

        self.lockedByBeamList = []

        if self.withMonitor:
            self.monitor = Monitor(battleField=battleField,
                                   unit=self,
                                   type='monitor',
                                   sensitivity=0, )

        if self.withJammer:
            if self.name == 'fighter':
                self.jammer = Jammer(battleField=battleField,
                                     unit=self,
                                     maxBeamNum=2,
                                     maxEffectiveDistance=100,
                                     beamAngleRange=10,
                                     beamPitchRange=[-60, 60]
                                     )
            else:
                self.jammer = Jammer(battleField=battleField,
                                     unit=self,
                                     maxBeamNum=20,
                                     maxEffectiveDistance=100,
                                     beamAngleRange=10,
                                     beamPitchRange=[-60, 60]
                                     )

        if self.withCommunicator:
            self.communicator = Communicator(battleField=battleField,
                                             type='air2air',
                                             unit=self,
                                             maxCommunicationDistance=560,
                                             )

        if self.withNavigator:
            if self.type == 'plane':
                self.navigator = Navigator(battleField=battleField,
                                           type='planeNavigator',
                                           speed=self.speed,
                                           acc=self.acc,
                                           unit=self)
            if self.type == 'ship':
                self.navigator = Navigator(battleField=battleField,
                                           type='shipNavigator',
                                           speed=self.speed,
                                           acc=self.acc,
                                           unit=self)
            if self.type == 'ground':
                self.navigator = Navigator(battleField=battleField,
                                           type='groundNavigator',
                                           speed=self.speed,
                                           acc=self.acc,
                                           unit=self)

        if self.withRadar:
            self.radar = Radar(battleField=battleField, type='selfDefense', unit=self, maxEffectiveDistance=100)

        self.normalMissileList = []
        self.antiRadiationMissileList = []

        if self.name == 'fighter':
            for i in range(normalMissileNum):
                missile = Missile(battleField=battleField,
                                  name='fox%d' % i,
                                  speed=Plane_Missile_speed,
                                  type='normal',
                                  unit=self,
                                  )
                self.normalMissileList.append(missile)
            for i in range(antiRadiationMissileNum):
                missile = Missile(battleField=battleField,
                                  name='fox%d' % i,
                                  type='antiRadiation',
                                  speed=Plane_Missile_speed,
                                  unit=self,
                                  )
                self.antiRadiationMissileList.append(missile)

        if self.name == 'Destroyer':
            for i in range(normalMissileNum):
                missile = Missile(battleField=battleField,
                                  name='fox%d' % i,
                                  speed=Destroyer_Missile_speed,
                                  type='normal',
                                  unit=self,
                                  )
                self.normalMissileList.append(missile)
            for i in range(antiRadiationMissileNum):
                missile = Missile(battleField=battleField,
                                  name='fox%d' % i,
                                  type='antiRadiation',
                                  speed=Destroyer_Missile_speed,
                                  unit=self,
                                  )
                self.antiRadiationMissileList.append(missile)
        if self.name == 'ground_air_defense_missile_rack':
            for i in range(normalMissileNum):
                missile = Missile(battleField=battleField,
                                  name='fox%d' % i,
                                  speed=ground_Missile_speed,
                                  type='normal',
                                  unit=self,
                                  )
                self.normalMissileList.append(missile)
            for i in range(antiRadiationMissileNum):
                missile = Missile(battleField=battleField,
                                  name='fox%d' % i,
                                  type='antiRadiation',
                                  speed=ground_Missile_speed,
                                  unit=self,
                                  )
                self.antiRadiationMissileList.append(missile)

        self.movePlan = []
        self.schedule = []
        if self.name == 'EWA':
            if self.color == 'red':
                self.shapeImage_0 = Image.open("../EWA-10.png")
            elif self.color == 'blue':
                self.shapeImage_0 = Image.open('../blueEWA.png')
            self.shapeImage_0 = self.shapeImage_0.resize((40, 40))

        elif self.name == 'fighter':
            if self.color == 'red':
                self.shapeImage_0 = Image.open('../J-16-1.png')
            elif self.color == 'blue':
                self.shapeImage_0 = Image.open('../F16.png')
            self.shapeImage_0 = self.shapeImage_0.resize((20, 20))

        elif self.name == 'EJA':
            if self.color == 'red':
                self.shapeImage_0 = Image.open('../EJA.png')
            elif self.color == 'blue':
                self.shapeImage_0 = Image.open('../blueEJA.png')
            self.shapeImage_0 = self.shapeImage_0.resize((30, 30))

        elif self.name == 'Destroyer':
            if self.color == 'red':
                self.shapeImage_0 = Image.open('../redDestroyer.png')
            elif self.color == 'blue':
                self.shapeImage_0 = Image.open('../blueDestroyer.png')
            self.shapeImage_0 = self.shapeImage_0.resize((40, 40))

        elif self.name == 'SurfaceShip':
            if self.color == 'red':
                self.shapeImage_0 = Image.open('../redSurfaceShip.png')
            elif self.color == 'blue':
                self.shapeImage_0 = Image.open('../blueSurfaceShip.png')
            self.shapeImage_0 = self.shapeImage_0.resize((20, 20))

        elif self.name == 'ground_radar':
            self.shapeImage_0 = Image.open('../radargronud.png')  # radargronud.png
            self.shapeImage_0 = self.shapeImage_0.resize((40, 40))

        elif self.name == 'ground_air_defense_missile_rack':
            self.shapeImage_0 = Image.open('../daodanjia.png')
            self.shapeImage_0 = self.shapeImage_0.resize((20, 20))

        else:
            self.shapeImage_0 = None
        self.image = None

        battleField.add_unit(self)

        #if self.color == 'red':
        #    battleField.redUnitList.append(self)
        #elif self.color == 'blue':
        #    battleField.blueUnitList.append(self)

    def get_unitIndex(self):
        return self.unitIndex
    def set_moveDirection(self, moveDirection):
        self.moveDirection[0] = moveDirection / 180 * np.pi

    def set_attackRange(self, attackRange):
        self.attackRange = attackRange

    def set_attackEnable(self, arg):
        self.attackEnable = arg

    def get_attackChannel(self):
        for attackChannel in self.attackChannelList:
            if not attackChannel.target:
                break
        return attackChannel

    def silence_on(self):
        for link in self.communicator.sendLinkList:
            #link.on = False
            sender = link.ends[0].unitIndex
            receiver = link.ends[1].unitIndex
            self.battleField.maxLinkDistanceMatrix[sender, receiver] = -1

        for beam in self.radar.beamList:
            beam.close()
        self.radar.on = False

        for beam in self.jammer.beamList:
            beam.close()
        self.jammer.on = False

    def silence_off(self):
        for link in self.communicator.sendLinkList:
            #link.on = False
            sender = link.ends[0].unitIndex
            receiver = link.ends[1].unitIndex
            self.battleField.maxLinkDistanceMatrix[sender, receiver] = link.maxLinkDistance
        self.radar.on = True
        self.jammer.on = True

    def attack(self, target, missileType, missileNum):
        attackedFlag = False
        if self.attack_condition(target):
            #选择导弹类型
            if missileType == 'normal':
                missileList = self.normalMissileList
            elif missileType == 'antiRadiation':
                missileList = self.antiRadiationMissileList

            missileNum = min(missileNum, len(missileList))
            if missileNum and self.availableAttackChannelNum:
                #设定火力通道
                attackChannel = self.get_attackChannel()
                attackChannel.set_target(target)
                for i in range(missileNum):
                    missile = missileList[0]
                    attackChannel.add_missile(missile)
                    missile.launch()
                    target.trackedByMissileList.append(missile)
                attackedFlag = True
                #self.fightingFlag = False
        #elif not missileNum:
        #    print('Lack Missile')
        #elif not self.availableAttackChannelNum:
        #    print('Lack AttackChannel')
        self.aimingTargetList.remove(target)
        return attackedFlag

    # move by acc
    def move(self):
        if self.navigator.movePlan:
            #self.position[0] += self.navigator.movePlan[0][0]
            #self.position[1] += self.navigator. movePlan[0][1]
            self.moveDirection[0] = self.navigator.movePlan[0][3]  #传入第一个列表的moveDirection
            self.moveDirectionPitch[0] = self.navigator.movePlan[0][4]
            self.navigator.movePlan[0][-1] -= 1
            if self.navigator.movePlan[0][-1] == 0:
                del self.navigator.movePlan[0]

        # 速度、加速度都变成三维的
        # s = v0*t+0.5*a*t**2
        self.position[0] += (
                                    self.speed * battleField.simTimeInterval + 0.5 * self.acc * battleField.simTimeInterval ** 2) * np.sin(
            self.moveDirectionPitch[0]) * np.cos(self.moveDirection[0])
        self.position[1] += (
                                    self.speed * battleField.simTimeInterval + 0.5 * self.acc * battleField.simTimeInterval ** 2) * np.sin(
            self.moveDirectionPitch[0]) * np.sin(self.moveDirection[0])
        self.position[2] += (
                                    self.speed * battleField.simTimeInterval + 0.5 * self.acc * battleField.simTimeInterval ** 2) * np.cos(
            self.moveDirectionPitch[0])

        self.speed += self.acc * battleField.simTimeInterval

    def plot_plane(self, x, y, direction):
        circle = plt.Circle((x, y), 20, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(direction), np.sin(direction)]) * 2)
        plt.plot([x, out_x], [y, out_y], "-k")

    def plot_arrow(self, x, y, direction, length=40, width=0.5):
        plt.arrow(x, y, length * np.cos(direction), length * np.sin(direction),
                  head_length=width, head_width=width)
        plt.plot(x, y)

    # move by DWA (test)
    def move1(self):
        # initial state [x,y,z,vx,vy,vz]
        x = self.position + [0.05, 0.05, 0.05]  #vx,vy,vz初始化
        goals = self.navigator.path
        trajectory = np.array(x)

        #测试DWA算法
        fig, ax = plt.subplots()

        # goal = goals[0]
        # u, predicted_trajectory = self.navigator.dwa_control(x, goal)
        # x = self.navigator.kinematicModel(x, u, battleField.simTimeInterval)
        # self.position[0] += x[0]
        # self.position[1] += x[1]
        # self.position[2] += x[2]
        # trajectory = np.vstack((trajectory, x))
        # if ([x[0], x[1], x[2]] == goal).any():
        #     print("Goal 1 Reached!!")
        #     # goals = goals[1:]
        #     self.navigator.path = self.navigator.path[1:]

        while len(goals) > 0:
            goal = goals[0]
            u, predicted_trajectory = self.navigator.dwa_control(x, goal)
            x = self.navigator.kinematicModel(x, u, battleField.simTimeInterval)
            self.position[0] = x[0]
            self.position[1] = x[1]
            self.position[2] = x[2]
            trajectory = np.vstack((trajectory, x))
            if battleField.get_distance(self.position, goal) < 3:
                print("Goal 1 Reached!!")
                goals = goals[1:]
            print(goal)
            print("当前的速度是: ", x[3], x[4], x[5])
            print("当前的坐标是: ", x[0], x[1], x[2])

            # 测试DWA算法
            ax.cla()
            ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            ax.plot(x[0], x[1], "xr")
            for g in goals:
                ax.plot(g[0], g[1], "xr")
            self.plot_plane(x[0], x[1], self.navigator.cal_heading(x))
            self.plot_arrow(x[0], x[1], self.navigator.cal_heading(x))
            ax.axis("equal")
            ax.grid(True)
            plt.pause(0.001)

    def move_by_static(self):
        # if len(self.navigator.path) > 0:
        #     if battleField.get_distance(self.position, self.goal) < 3:
        #         del self.navigator.path[0]
        if self.name == 'EJA' or self.name == 'EWA' or self.name == 'fighter':
            if len(self.navigator.path) > 0:
                self.goal = self.navigator.path[0]
                # 距离目标点2km即到达
                if battleField.get_distance(self.position, self.goal) < 2:
                    # del self.navigator.path[0]
                    self.navigator.path = self.navigator.path[1:]
            # print("self.goal", self.navigator.path)
            v, predicted_trajectory = self.navigator.dwa_control(self.state, self.goal)
            self.state = self.navigator.kinematicModel(self.state, v, battleField.simTimeInterval)
            self.position[0] = self.state[0]
            self.position[1] = self.state[1]
            self.position[2] = self.state[2]
            self.moveDirection[0] = self.navigator.cal_heading(self.state)

        # todo：地面静止物体的位置控制
        else:
            last_position = self.position
            self.position[0] = last_position[0]
            self.position[1] = last_position[1]
            self.position[2] = last_position[2]

    def move_by_DWA(self):
        """
        基于目标位置，通过三维DWA算法来控制飞机飞行，每次只更新一步,更新self.position
        """

        if len(self.navigator.path) > 0:
            self.goal = self.navigator.path[0]
            # 距离目标点3km即到达
            if battleField.get_distance(self.position, self.goal) < 3:
                del self.navigator.path[0]

        v, predicted_trajectory = self.navigator.dwa_control(self.state, self.goal)
        self.state = self.navigator.kinematicModel(self.state, v, battleField.simTimeInterval)
        self.position[0] = self.state[0]
        self.position[1] = self.state[1]
        self.position[2] = self.state[2]
        self.moveDirection[0] = self.navigator.cal_heading(self.state)

        # 限制飞机在地图内
        if self.position[0] < 0:  # todo
            self.position[0] = 1
        if self.position[0] > 1000:
            self.position[0] = 999
        if self.position[1] < 0:
            self.position[1] = 1
        if self.position[1] > 1000:
            self.position[1] = 999

        # print(self.goal)
        # print("%s+%s 当前的速度是: %s %s %s" % (self.color, self.name, self.state[3], self.state[4], self.state[5]))
        # print(f"{self.color}_{self.name} 当前的速度是: {self.state[3]} {self.state[4]} {self.state[5]}")
        # print(f"{self.color}_{self.name} 当前的坐标是: {self.state[0]} {self.state[1]} {self.state[2]}")

    def move2Target(self, target):
        if not target.crashed:
            self.moveDirection[0], moveDirectionPitch0 = self.battleField.get_orientation(self.position,
                                                                                          target.position)
            self.moveDirectionPitch[0] = np.pi / 2 - moveDirectionPitch0

    def move2Target_position(self, target):
        if not target.crashed:
            self.goal = target.position

    def tracking(self, target, maxDistance):
        if not self.fightingFlag and not self.crashed:
            target.targetedByList.append(self)
            self.fightingFlag = True
            # self.speed = 2000 / 3600
            # self.speed *= 2
            if self.type == 'plane':
                self.speed = RedFighter_tracking_speed / 3600  # 没用不知道原因
            if self.name == 'ship':
                self.speed = 906 / 3600

            self.acc = RedFighter_tracking_acc  # 没用不知道原因
            self.navigator.v_max = [self.speed, self.speed, self.speed]
            self.navigator.v_min = [-self.speed, -self.speed, -self.speed]
            self.navigator.a_max = [self.acc, self.acc, self.acc]
            self.navigator.v_sample = 0.03

        if target in self.radar.detectedUnits \
                and self.battleField.distanceMatrix[0, target.unitIndex] <= maxDistance:
            self.move2Target_position(target)
            self.add_schedule(func=self.tracking,
                              args=(target,
                                    maxDistance),
                              delay=1,
                              )
        else:
            self.cancelTracking(target)

    def cancelTracking(self, target):
        self.fightingFlag = False
        target.targetedByList.remove(self)

    def guard(self, target):
        if not target.crashed:
            if self.battleField.distanceMatrix[self.unitIndex, target.unitIndex] > 30:
                self.move2Target_position(target)
                # self.speed *= 1.2
            else:
                self.speed = target.speed
                self.acc = target.acc
                self.navigator.v_max = [self.speed, self.speed, self.speed]
                self.navigator.v_min = [-self.speed, -self.speed, -self.speed]
                self.navigator.a_max = [self.acc, self.acc, self.acc]
                self.navigator.v_sample = target.navigator.v_sample
                # self.state[3] = target.state[3]
                # self.state[4] = target.state[4]
                # self.state[5] = target.state[5]
                self.moveDirection[0] = target.moveDirection[0]
                # self.moveDirection = target.moveDirection.copy()
                # self.moveDirectionPitch = target.moveDirectionPitch.copy()
                # self.goal = target.goal
                self.goal = target.goal.copy()

    def schedule_action(self):
        scheduleNum = len(self.schedule)
        i = 0
        j = 0
        while i < scheduleNum:
            action = self.schedule[j]
            if action['delay'] == 0:
                if action['args']:
                    action['func'](*action['args'])
                else:
                    action['func']()
                self.schedule.remove(action)
            else:
                action['delay'] -= 1
                j += 1
            i += 1

    def attack_condition(self, target):
        if not target.crashed \
                and len(target.trackedByMissileList) < 2:
            return True
        else:
            return False

    def action(self):

        self.schedule_action()

        for target in self.radar.detectedUnits:
            self.jammer.jam_targetUnit(target)


        # if self.unitIndex > 1 and not self.normalMissileList:
        #     print('%d_NormalMissile Lack' % self.unitIndex)
        if self.unitIndex > 1 and not self.normalMissileList:  # 海和地会显示index大于1，list为空
            # print("index", self.unitIndex)
            # print(self.normalMissileList)
            # print('%d_NormalMissile Lack'%self.unitIndex)
            pass

        if self.targetedByList:
            # print('%d_targetedBy =' % self.unitIndex, end='')
            # for unit in self.targetedByList:
            #     print(unit.unitIndex, end=',')
            # print('\n')
            pass
        # if self.unitIndex==2:
        #     print('detectedUnits =', end='')
        #     for unit in self.radar.detectedUnits:
        #         print(unit.unitIndex, end=',')
        #     print('\n')
        if self.attackEnable and self.normalMissileList:
            #if not self.radar.detectedUnits:
            #self.fightingFlag = False
            #    pass
            if self.radar.detectedUnits:
                if not self.fightingFlag:
                    # if self.unitIndex==2:
                    #     print('detectedUnits =', end='')
                    #     for unit in self.radar.detectedUnits:
                    #         print(unit.unitIndex, end=',')
                    #     print('\n')
                    for target in self.radar.detectedUnits:
                        if target.color != self.color and not target.targetedByList and not self.fightingFlag:
                            if self.unitIndex == 2:
                                maxTrackingDistance = 150
                            else:
                                maxTrackingDistance = 400

                            # 加入蓝色战斗机的追击距离
                            if self.color == 'blue' and self.name == 'fighter':
                                maxTrackingDistance = 300

                            if self.battleField.distanceMatrix[0, target.unitIndex] <= maxTrackingDistance:
                                self.tracking(target, maxTrackingDistance)
                                break
                for target in self.radar.detectedUnits:
                    if target.color != self.color:
                        #self.move2Target(target)
                        #self.fightingFlag = True
                        #self.speed = 1800/3600
                        if self.battleField.distanceMatrix[self.unitIndex, target.unitIndex]< self.attackRange:
                            self.aimingTargetList.append(target)
                            self.add_schedule(func=self.attack,
                                              args=(target, 'normal', 1),
                                              delay=5,
                                              )

        if self.attackEnable and self.antiRadiationMissileList:
            #todo:修改了反辐射雷达发射条件:有敌方波束照到搭载反辐射导弹的单位且波束功率过大
            if self.lockedByBeamList:
                for lockedBeam in self.lockedByBeamList:
                    if lockedBeam.unit not in self.aimingTargetList and not lockedBeam.unit.crashed and \
                            len(lockedBeam.unit.trackedByMissileList)<1 and \
                            self.battleField.distanceMatrix[self.unitIndex, lockedBeam.unit.unitIndex] \
                            < self.antiAttackRange:
                        # 设置反辐射雷达导弹的发射范围为：antiAttackRange即普通导弹+100
                        self.aimingTargetList.append(lockedBeam.unit)
                        self.add_schedule(func=self.attack,
                                          args=(lockedBeam.unit, 'antiRadiation', 1),
                                          delay=5, )
        #print('fightingFlag =', self.battleField.unitList[2].fightingFlag)
        #只让eja和ewa使用guard策略
        if self.color == 'red' and self.name == 'EJA' and self.unitIndex != 0 and not self.fightingFlag:
            self.guard(self.battleField.unitList[0])

        if self.battleField.environment_name == 'air-air' or self.battleField.environment_name == 'air-sea':
            if self.color == 'red' and self.name == 'fighter':
                self.move_by_MADDPG()
            else:
                self.move_by_DWA()
        elif self.battleField.environment_name == 'air-ground':
            self.move_by_static()

    def crash(self):
        self.crashed = True
        self.battleField.crashedUnitList.append(self)
        for equipment in self.equipments:
            equipment.crash()
        print('%s crashed' % self.unitIndex)

    def add_schedule(self, func, args, delay):
        item = {'func': func,
                'args': args,
                'delay': delay
                }
        self.schedule.append(item)

    def display_spectrum_efficiency(self):
        if self.color == 'red':
            # 计算文本显示的位置
            x = 100  # 在单位位置的右侧显示
            y = 600+self.unitIndex*20  # 在单位位置的下方显示

            # 在画布上显示频谱利用率
            self.battleField.canvas.create_text(
                x, y,
                text=f"{self.color}{self.name}{self.unitIndex}:频谱利用率: {self.spectrum_efficiency:.2f}",
                fill='red', font=('Arial', 12)
            )


    def display(self):
        if self.displayEnable:
            if not self.crashed:
                for equipment in self.equipments:
                    equipment.display()

                if self.name == 'fighter':

                    self.shapeImage = ImageTk.PhotoImage(self.shapeImage_0.rotate(-self.moveDirection[0] / np.pi * 180))
                    self.image = battleField.canvas.create_image(self.position[0] - 10,
                                                                 self.position[1] - 10,
                                                                 anchor=tk.NW,
                                                                 image=self.shapeImage)
                    self.textImage = battleField.canvas.create_text(self.position[0],
                                                                    self.position[1],
                                                                    text=self.unitIndex,
                                                                    font=('Purisa', 15),
                                                                    fill='white'
                                                                    )
                elif self.name == 'EJA':

                    self.shapeImage = ImageTk.PhotoImage(self.shapeImage_0.rotate(-self.moveDirection[0] / np.pi * 180))
                    self.image = battleField.canvas.create_image(self.position[0] - 15,
                                                                 self.position[1] - 15,
                                                                 anchor=tk.NW,
                                                                 image=self.shapeImage)
                elif self.name == 'EWA':

                    self.shapeImage = ImageTk.PhotoImage(self.shapeImage_0.rotate(-self.moveDirection[0] / np.pi * 180))
                    self.image = battleField.canvas.create_image(self.position[0] - 20,
                                                                 self.position[1] - 20,
                                                                 anchor=tk.NW,
                                                                 image=self.shapeImage)

                elif self.name == 'Destroyer':
                    self.shapeImage = ImageTk.PhotoImage(self.shapeImage_0.rotate(-self.moveDirection[0] / np.pi * 180))
                    self.image = battleField.canvas.create_image(self.position[0] - 20,
                                                                 self.position[1] - 20,
                                                                 anchor=tk.NW,
                                                                 image=self.shapeImage)
                elif self.name == 'SurfaceShip':
                    self.shapeImage = ImageTk.PhotoImage(self.shapeImage_0.rotate(-self.moveDirection[0] / np.pi * 180))
                    self.image = battleField.canvas.create_image(self.position[0] - 20,
                                                                 self.position[1] - 20,
                                                                 anchor=tk.NW,
                                                                 image=self.shapeImage)
                elif self.name == 'ground_radar':
                    self.shapeImage = ImageTk.PhotoImage(self.shapeImage_0.rotate(-self.moveDirection[0] / np.pi * 180))
                    self.image = battleField.canvas.create_image(self.position[0] - 20,
                                                                 self.position[1] - 20,
                                                                 anchor=tk.NW,
                                                                 image=self.shapeImage)

                elif self.name == 'ground_air_defense_missile_rack':
                    self.shapeImage = ImageTk.PhotoImage(self.shapeImage_0.rotate(-self.moveDirection[0] / np.pi * 180))
                    self.image = battleField.canvas.create_image(self.position[0] - 20,
                                                                 self.position[1] - 20,
                                                                 anchor=tk.NW,
                                                                 image=self.shapeImage)


            else:
                self.image = battleField.canvas.create_oval(self.position[0] - 8,
                                                            self.position[1] - 8,
                                                            self.position[0] + 8,
                                                            self.position[1] + 8,
                                                            fill=self.color
                                                            )
                self.crashFlag0 = battleField.canvas.create_line(self.position[0] - 8,
                                                                 self.position[1] - 8,
                                                                 self.position[0] + 8,
                                                                 self.position[1] + 8,
                                                                 fill='black',
                                                                 width=2,
                                                                 )
                self.crashFlag1 = battleField.canvas.create_line(self.position[0] + 8,
                                                                 self.position[1] - 8,
                                                                 self.position[0] - 8,
                                                                 self.position[1] + 8,
                                                                 fill='black',
                                                                 width=2,
                                                                 )
                self.textImage = battleField.canvas.create_text(self.position[0],
                                                                self.position[1],
                                                                text=self.unitIndex,
                                                                font=('Purisa', 8),
                                                                fill='white'
                                                                )

    def move_by_MADDPG(self):
        self.moveDirection[0] = self.navigator.cal_heading(self.state)
        pass


class Get_Unit:
    def __init__(self, name, json_file='../json_sr.json'):
        self.name = name
        self.unit_data = self.load_unit_data(json_file)

    def load_unit_data(self, json_file):
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
                # 遍历JSON数组，寻找匹配的单位名称
                for unit in data:
                    if unit['name'] == self.name:
                        return unit
                # 如果遍历完成都没有找到，输出名字不存在的错误信息
                print(f"No unit with the name '{self.name}' found in the JSON file.")
                return None
        except FileNotFoundError:
            print(f"JSON file '{json_file}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file '{json_file}'.")
            return None

    def get_property(self, property_name):
        return self.unit_data.get(property_name) if self.unit_data else None


class GetEnvironment:
    def __init__(self, name, json_file='../environment.json'):
        self.name = name
        self.unit_data = self.load_unit_data(json_file)

    def load_unit_data(self, json_file):
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
                # 遍历JSON数组，寻找匹配的单位名称
                for unit in data:
                    if unit['name'] == self.name:
                        return unit
                # 如果遍历完成都没有找到，输出名字不存在的错误信息
                print(f"No unit with the name '{self.name}' found in the JSON file.")
                return None
        except FileNotFoundError:
            print(f"JSON file '{json_file}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file '{json_file}'.")
            return None

    def get_property(self, property_name):
        return self.unit_data.get(property_name) if self.unit_data else None


#  name要和json文件里的名字一样;color是'red','blue';geospatial是'air','sea'，'ground';number是要生成几个实体
#  返回值b是json文件里面的其他参数，a是生成的实体


def generating_unit(name, color, geospatial, battleField, number):
    a = None
    b = None
    if color == 'red':
        if geospatial == 'air':
            if name == 'Multi_role_fighter':#todo：多用途战斗机，跟fighter战斗机一样，暂时未配图
                b = Get_Unit('Multi_role_fighter')
                a = Unit(battleField,
                         name='Multi_role_fighter',
                         type='plane',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],  # 随便设的
                         position=[855, 435, 10],  # 随便设的
                         normalMissileNum=0,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=0,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
            if name == 'RedFighter':
                b = Get_Unit('RedFighter')
                a = Unit(battleField,
                         name='fighter',
                         type='plane',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[-0.05, 0.05, 0.05],
                         position=[900 + 50 * np.sin(number * np.pi / 6), 400 + 50 * np.cos(number * np.pi / 6), 10],
                         # position=[200 + 50 * np.sin(number * np.pi / 6), 250 + 50 * np.cos(number * np.pi / 6), 10],
                         normalMissileNum=20,
                         antiRadiationMissileNum=5,
                         maxAttackChannelNum=2,
                         maxInfluenceDistance=100,
                         )
            if name == 'consumable_UAV': #todo：可消耗无人机，暂时未配图
                b = Get_Unit('consumable_UAV')
                a = Unit(battleField,
                         name='fighter',
                         type='plane',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],  # 随便设的
                         position=[900 + 50 * np.sin(number * np.pi / 6), 400 + 50 * np.cos(number * np.pi / 6), 10],
                         # 随便设的
                         normalMissileNum=20,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=2,  # 随便设的
                         maxInfluenceDistance=100  # 随便设的
                         )
            if name == 'EWA':
                b = Get_Unit('EWA')
                a = Unit(battleField,
                         name='EWA',
                         type='plane',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],
                         position=[900, 400, 10],
                         normalMissileNum=0,
                         antiRadiationMissileNum=0,
                         maxAttackChannelNum=0,
                         maxInfluenceDistance=400,
                         )
            if name == 'EJA':
                b = Get_Unit('EJA')
                a = Unit(battleField,
                         name='EJA',
                         type='plane',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],
                         position=[855, 435, 10],
                         normalMissileNum=0,
                         antiRadiationMissileNum=0,
                         maxAttackChannelNum=0,
                         maxInfluenceDistance=400,
                         )
            if name == 'electronic_reconnaissance_plane': #todo：电子侦察机，暂时未配图
                b = Get_Unit('electronic_reconnaissance_plane')
                a = Unit(battleField,
                         name='EJA',
                         type='plane',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],  # 随便设的
                         position=[855, 435, 10],  # 随便设的
                         normalMissileNum=0,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=0,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
        if geospatial == 'sea':
            if name == 'Destroyer':
                b = Get_Unit('Destroyer')
                a = Unit(battleField,
                         name='Destroyer',
                         type='ship',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0],  # 随便设的
                         position=[650, 700, 0],  # 随便设的
                         normalMissileNum=90,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=2,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
            if name == 'SurfaceShip':
                b = Get_Unit('SurfaceShip')
                a = Unit(battleField,
                         name='SurfaceShip',
                         type='ship',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0],  # 随便设的
                         position=[650 + random.randint(-50, 50), 740 + random.randint(-50, 50),0],  # 随便设的
                         normalMissileNum=0,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=2,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
        if geospatial == 'ground':
            if name == 'ground_radar':
                b = Get_Unit('ground_radar')
                a = Unit(battleField,
                         name='ground_radar',
                         type='radar',
                         color='red',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],  # 随便设的
                         position=[855, 435, 10],  # 随便设的
                         normalMissileNum=0,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=0,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
    if color == 'blue':
        if geospatial == 'air':
            if name == 'Multi_role_fighter':
                b = Get_Unit('Multi_role_fighter')
                a = Unit(battleField,
                         name='Multi_role_fighter',
                         type='plane',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],  # 随便设的
                         position=[855, 435, 10],  # 随便设的
                         normalMissileNum=0,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=0,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
            if name == 'BlueFighter':
                b = Get_Unit('BlueFighter')
                a = Unit(battleField,
                         name='fighter',
                         type='plane',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],
                         position=[150+random.randint(-100, 100), 100+random.randint(-100, 100), 10],  # 150, 100, 10
                         normalMissileNum=6,
                         antiRadiationMissileNum=20,
                         maxAttackChannelNum=2,
                         maxInfluenceDistance=100,
                         )
            if name == 'consumable_UAV':
                b = Get_Unit('consumable_UAV')
                a = Unit(battleField,
                         name='fighter',
                         type='plane',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],  # 随便设的
                         position=[900 + 50 * np.sin(number * np.pi / 6), 400 + 50 * np.cos(number * np.pi / 6), 10],
                         # 随便设的
                         normalMissileNum=20,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=2,  # 随便设的
                         maxInfluenceDistance=100  # 随便设的
                         )
            if name == 'EWA':
                b = Get_Unit('EWA')
                a = Unit(battleField,
                         name='EWA',
                         type='plane',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],
                         position=[150, 100, 10],
                         normalMissileNum=0,
                         antiRadiationMissileNum=0,
                         maxAttackChannelNum=0,
                         maxInfluenceDistance=400,
                         )
            if name == 'EJA':
                b = Get_Unit('EJA')
                a = Unit(battleField,
                         name='EJA',
                         type='plane',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],
                         position=[150, 150, 10],  # 855, 435, 10
                         normalMissileNum=0,
                         antiRadiationMissileNum=0,
                         maxAttackChannelNum=0,
                         maxInfluenceDistance=400,
                         )
            if name == 'electronic_reconnaissance_plane':
                b = Get_Unit('electronic_reconnaissance_plane')
                a = Unit(battleField,
                         name='electronic_reconnaissance_plane',
                         type='plane',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0.05],  # 随便设的
                         position=[855, 435, 10],  # 随便设的
                         normalMissileNum=0,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=0,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
        if geospatial == 'sea':
            if name == 'Destroyer':
                b = Get_Unit('Destroyer')
                a = Unit(battleField,
                         name='Destroyer',
                         type='ship',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0],  # 随便设的
                         position=[120, 200, 0],  # 随便设的  150, 100, 10
                         normalMissileNum=18,  # 随便设的
                         antiRadiationMissileNum=20,  # 随便设的
                         attackRange=150,
                         maxAttackChannelNum=10,  # 随便设的
                         maxInfluenceDistance=500,  # 随便设的
                         )
            if name == 'SurfaceShip':
                b = Get_Unit('SurfaceShip')
                a = Unit(battleField,
                         name='SurfaceShip',
                         type='ship',
                         color='blue',
                         speed=b.unit_data['max_rate'],
                         acc=b.unit_data['max_acceleration'] / 1000,
                         v_init=[0.05, 0.05, 0],  # 随便设的
                         position=[120 + random.randint(-70, 70), 200 + random.randint(-70, 70), 0],  # 随便设的
                         normalMissileNum=2,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=2,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
        if geospatial == 'ground':
            if name == 'ground_radar':
                b = Get_Unit('ground_radar')
                a = Unit(battleField,
                         name='ground_radar',
                         type='ground',
                         color='blue',
                         speed=0,
                         acc=0,
                         v_init=[0, 0, 0],  # 随便设的
                         position=[250, 300, 0],  # 随便设的
                         normalMissileNum=0,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         maxAttackChannelNum=0,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
            if name == 'ground_air_defense_missile_rack':
                a = Unit(battleField,
                         name='ground_air_defense_missile_rack',
                         type='ground',
                         color='blue',
                         speed=0,
                         acc=0,
                         v_init=[0, 0, 0],  # 随便设的
                         position=[250 + random.randint(-150, 150), 300 + random.randint(-150, 150), 0],  # 随便设的
                         normalMissileNum=90,  # 随便设的
                         antiRadiationMissileNum=0,  # 随便设的
                         attackRange=700,
                         maxAttackChannelNum=2,  # 随便设的
                         maxInfluenceDistance=400,  # 随便设的
                         )
    return a




if __name__ == '__main__':
    # air-sea、air-air、air-ground
    env = "air-air"

    if env == 'air-air':
        RedFighter = Get_Unit('RedFighter')
        # 红方战斗机追踪目标最大速度(km/h)2410
        RedFighter_tracking_speed = RedFighter.unit_data['max_rate_for_tracking']  #todo需要这个吗？
        # 红方战斗机追踪目标时的加速度(km/s**2)0.075
        RedFighter_tracking_acc = RedFighter.unit_data['max_acceleration_for_tracking'] / 1000  # todo
        # 导弹速度(km/h)7200
        # todo暂时设置红方蓝方导弹速度相同
        Plane_Missile_speed = RedFighter.unit_data.get('air_to_air_missile', {}).get('max_rate')

        battleField = BattleField('air-air',
                                  timer=0,
                                  size=[1000, 1000, 10],
                                  acceleration=100,
                                  frameTimeInterval=0.02,
                                  simTimeInterval=1,
                                  simDuration=60 * 180,  #180min
                                  )
        redEarlyWarningAircraft = generating_unit('EWA', 'red', 'air', battleField, 1)
        # redEarlyWarningAircraft = Unit(battleField,
        #                                name='EWA',
        #                                type='plane',
        #                                color='red',
        #                                speed=EWA_speed,
        #                                acc=EWA_acc,
        #                                v_init=EWA_speed_init,
        #                                position=[900, 400, 10],
        #                                normalMissileNum=0,
        #                                antiRadiationMissileNum=0,
        #                                maxAttackChannelNum=0,
        #                                maxInfluenceDistance=400,
        #                                )
        redEarlyWarningAircraft.jammer.on = False
        redEarlyWarningAircraft.monitor.on = False
        redEarlyWarningAircraft.radar.turn_on()
        redEarlyWarningAircraft.radar.beamPower = 12 * 1e6
        for radarBeam in redEarlyWarningAircraft.radar.beamList:
            # radarBeam.set_maxEffectiveDistance(400)
            # radarBeam.set_currentEffectiveDistance(400)
            radarBeam.set_angleRange(360)
            # radarBeam.set_pitchRange([-60,60])

        # path0 = np.random.randint(0, 1000, size=(10, 3))#10行2列
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])

        #path = list(path)

        # 如果预警机被击落则其他飞机无法到达目的地，因为只给预警机设置了path
        redEarlyWarningAircraft.navigator.set_path1(list(path0))
        # redEarlyWarningAircraft.navigator.set_path(list(path0))

        redJammingAircraft = generating_unit('EJA', 'red', 'air', battleField, 1)

        # redJammingAircraft = Unit(battleField,
        #                           name='EJA',
        #                           type='plane',
        #                           color='red',
        #                           speed=EJA_speed,
        #                           acc=EJA_acc,
        #                           v_init=EJA_speed_init,
        #                           position=[855, 435, 10],
        #                           normalMissileNum=0,
        #                           antiRadiationMissileNum=0,
        #                           maxAttackChannelNum=0,
        #                           maxInfluenceDistance=400,
        #                           )
        #redJammingAircraft.jammer.on = False
        #redJammingAircraft.monitor.on = False
        redJammingAircraft.radar.turn_on()
        redJammingAircraft.jammer.set_maxEffectiveDistance(400)
        # 设置干扰机干扰角度 unit类中self.jammer
        redJammingAircraft.jammer.set_beamAngleRange(1)
        redJammingAircraft.jammer.set_beamPitchRange([-60, 60])
        # redJammingAircraft.navigator.set_path1(list(path0))
        # radarBeam.set_maxEffectiveDistance(400)
        #radarBeam.set_currentEffectiveDistance(400)
        # 设置预警机探测角度
        # radarBeam.set_angleRange(360)
        #path = path0 + 50
        #path = list(np.random.randint(0, 1000, size=(10, 2)))
        #redJammingAircraft.navigator.set_path(list(path))

        for i in range(5):
            redPlane = generating_unit('RedFighter', 'red', 'air', battleField, i)

            # for i in range(2):
            #     redPlane = Unit(battleField,
            #                     name='fighter',
            #                     type='plane',
            #                     color='red',
            #                     speed=RedFighter_speed,
            #                     acc=RedFighter_acc,
            #                     v_init=RedFighter_speed_init,
            #                     position=[900+50*np.sin(i*np.pi/6), 400+50*np.cos(i*np.pi/6), 10],
            #                     normalMissileNum=20,
            #                     antiRadiationMissileNum=0,
            #                     maxAttackChannelNum=2,
            #                     maxInfluenceDistance=100,
            #                     )
            redPlane.navigator.v_sample = 0.03

            redPlane.radar.turn_on()
            redPlane.jammer.set_beamAngleRange(1)
            redPlane.jammer.beamPower = 40 * 1e3
            #path = list(np.random.randint(0, 1000, size=(100, 2)))
            #redPlane.navigator.set_path(path)
        # redPlane1 = generating_unit('RedFighter', 'red', 'air', battleField, 4)
        # redPlane1.position=[250,200,10]

        # 加入驱逐舰
        Destroyer = Get_Unit('Destroyer')
        Destroyer_Missile_speed = Destroyer.unit_data.get('ground_to_air_missile', {}).get('max_rate')
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
            surface_ship.navigator.set_path1(path0)

        blueEarlyWarningAircraft = generating_unit('EWA', 'blue', 'air', battleField, 1)

        blueEarlyWarningAircraft.jammer.on = False
        blueEarlyWarningAircraft.monitor.on = False
        blueEarlyWarningAircraft.radar.turn_on()
        blueEarlyWarningAircraft.radar.beamPower = 12 * 1e6

        for radarBeam in blueEarlyWarningAircraft.radar.beamList:
            # radarBeam.set_maxEffectiveDistance(400)
            # radarBeam.set_currentEffectiveDistance(400)
            radarBeam.set_angleRange(360)
            # radarBeam.set_pitchRange([-60,60])

        # path0 = np.random.randint(0, 1000, size=(10, 3))#10行2列
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])
        blueEarlyWarningAircraft.navigator.set_path1(list(path0))
        #for i in range(1):
        #    blueEWA = Unit(battleField,
        #                     name='EWA',
        #                     type='plane',
        #                     color='blue',
        #                     speed=0,
        #                     position=[900, 500],
        #                     normalMissileNum=0,
        #                     antiRadiationMissileNum=0,
        #                     maxAttackChannelNum=0,
        #                     maxInfluenceDistance=300,
        #                     )
        #    blueEWA.radar.turn_on()
        #    blueEWA.jammer.on = False
        #    blueEWA.monitor.on = False

        #    for radarBeam in blueEWA.radar.beamList:
        #        radarBeam.set_maxEffectiveDistance(300)
        #        radarBeam.set_currentEffectiveDistance(300)
        #        radarBeam.set_angleRange(360)
        #    #path = list(np.random.randint(0, 1000, size=(100, 2)))
        #bluePlane.navigator.set_path(path)

        blueJammingAircraft = generating_unit('EJA', 'blue', 'air', battleField, 1)
        blueJammingAircraft.jammer.set_maxEffectiveDistance(400)
        # 设置干扰机干扰角度 unit类中self.jammer
        blueJammingAircraft.jammer.set_beamAngleRange(1)
        blueJammingAircraft.jammer.set_beamPitchRange([-60, 60])
        blueJammingAircraft.radar.turn_on()
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])
        blueJammingAircraft.navigator.set_path1(list(path0))

        # path = list(np.random.randint(0, 1000, size=(100, 3)))
        path = np.array([[786, 835, 8]])


        for i in range(5):
            bluePlane = generating_unit('BlueFighter', 'blue', 'air', battleField, i)
            # for i in range(12):
            #     bluePlane = Unit(battleField,
            #                      name='fighter',
            #                      type='plane',
            #                      color='blue',
            #                      speed=BlueFighter_speed,
            #                      acc=BlueFighter_acc,
            #                      v_init=BlueFighter_speed_init,
            #                      position=[150, 100, 10],
            #                      normalMissileNum=6,
            #                      antiRadiationMissileNum=20,
            #                      maxAttackChannelNum=2,
            #                      maxInfluenceDistance=100,
            #                      )

            # 关闭蓝战斗机雷达
            bluePlane.radar.turn_on()
            bluePlane.jammer.set_beamAngleRange(1)
            bluePlane.jammer.beamPower = 40 * 1e3

            # 测试蓝方能否打到红方 （可）
            # bluePlane.radar.gain = 10*1e8

            bluePlane.navigator.v_sample = 0.028

            # 将path高度限制在10以内
            # for i in range(len(path)):
            #     path[i][2] = 7
                # path[i][2] = 10
            print(path)

            bluePlane.navigator.set_path1(list(path))

        # 加入驱逐舰
        Destroyer = Get_Unit('Destroyer')
        Destroyer_Missile_speed = Destroyer.unit_data.get('ground_to_air_missile', {}).get('max_rate')
        destroyer = generating_unit('Destroyer', 'blue', 'sea', battleField, 1)
        destroyer.jammer.on = True
        destroyer.navigator.v_sample = 0.005
        destroyer.monitor.on = False
        destroyer.radar.turn_on()
        destroyer.radar.beamPower = 5 * 1e6
        # 设置干扰机干扰角度 unit类中self.jammer
        destroyer.jammer.set_beamAngleRange(1)
        destroyer.jammer.set_beamPitchRange([-70, 90])
        for radarBeam in destroyer.radar.beamList:
            radarBeam.set_angleRange(360)
        path0 = np.array([[900, 600, 0],
                          [100, 600, 0],
                          [500, 100, 0]])

        destroyer.navigator.set_path1(list(path0))

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



            #bluePlane.jammer.on = False
            #bluePlane.radar.beamList[0].maxEffectiveDistance = 30
            #bluePlane.radar.maxDetectionDistance = 30
            #bluePlane.jammer.on = False

        battleField.simulator()

        print('end!')

    if env == 'air-sea':
        RedFighter = Get_Unit('RedFighter')
        # 红方战斗机追踪目标最大速度(km/h)2410
        RedFighter_tracking_speed = RedFighter.unit_data['max_rate_for_tracking']  # todo需要这个吗？
        # 红方战斗机追踪目标时的加速度(km/s**2)0.075
        RedFighter_tracking_acc = RedFighter.unit_data['max_acceleration_for_tracking'] / 1000  # todo
        # 导弹速度(km/h)7200
        # todo暂时设置红方蓝方导弹速度相同
        Plane_Missile_speed = RedFighter.unit_data.get('air_or_ship_to_ship_missile', {}).get('max_rate')
        battleField = BattleField('air-sea',
                                  timer=0,
                                  size=[1000, 1000, 10],
                                  acceleration=100,
                                  frameTimeInterval=0.02,
                                  simTimeInterval=1,
                                  simDuration=60 * 180,  # 180min
                                  )
        redEarlyWarningAircraft = generating_unit('EWA', 'red', 'air', battleField, 1)

        redEarlyWarningAircraft.jammer.on = False
        redEarlyWarningAircraft.monitor.on = False
        redEarlyWarningAircraft.radar.turn_on()
        for radarBeam in redEarlyWarningAircraft.radar.beamList:
            # radarBeam.set_maxEffectiveDistance(400)
            # radarBeam.set_currentEffectiveDistance(400)
            radarBeam.set_angleRange(360)
            # radarBeam.set_pitchRange([-60,60])

        # path0 = np.random.randint(0, 1000, size=(10, 3))#10行2列
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])

        # path = list(path)

        # 如果预警机被击落则其他飞机无法到达目的地，因为只给预警机设置了path
        redEarlyWarningAircraft.navigator.set_path1(list(path0))
        # redEarlyWarningAircraft.navigator.set_path(list(path0))

        Destroyer = Get_Unit('Destroyer')
        Destroyer_Missile_speed = Destroyer.unit_data.get('ground_to_air_missile', {}).get('max_rate')


        redJammingAircraft = generating_unit('EJA', 'red', 'air', battleField, 1)


        redJammingAircraft.jammer.set_maxEffectiveDistance(400)
        # 设置干扰机干扰角度 unit类中self.jammer
        redJammingAircraft.jammer.set_beamAngleRange(10)
        redJammingAircraft.jammer.set_beamPitchRange([-60, 60])


        for i in range(5):
            redPlane = generating_unit('RedFighter', 'red', 'air', battleField, i)
            redPlane.navigator.v_sample = 0.03

        destroyer = generating_unit('Destroyer', 'blue', 'sea', battleField, 1)
        destroyer.jammer.on = True
        destroyer.monitor.on = False
        # destroyer.fightingFlag = False
        destroyer.radar.turn_on()
        destroyer.radar.beamPower = 10 * 1e6
        destroyer.jammer.set_maxEffectiveDistance(1000)# maxEffectiveDistance没用
        # 设置干扰机干扰角度 unit类中self.jammer
        destroyer.jammer.set_beamAngleRange(1)
        destroyer.jammer.set_beamPitchRange([-70, 90])
        for radarBeam in destroyer.radar.beamList:
            radarBeam.set_angleRange(360)
        path0 = np.array([[900, 500, 0],
                          [100, 600, 0],
                          [500, 100, 0]])

        destroyer.navigator.set_path1(list(path0))

        for i in range(3):
            surface_ship = generating_unit('SurfaceShip', 'blue', 'sea', battleField, i)
            surface_ship.jammer.on = True
            surface_ship.monitor.on = False  # TODO
            surface_ship.radar.turn_on()
            surface_ship.radar.beamPower = 10 * 1e6
            for radarBeam in surface_ship.radar.beamList:
                radarBeam.set_angleRange(360)
            surface_ship.jammer.set_maxEffectiveDistance(700)
            # 设置干扰机干扰角度 unit类中self.jammer
            surface_ship.jammer.set_beamAngleRange(1)
            surface_ship.jammer.set_beamPitchRange([-70, 70])
            # surface_ship.navigator.v_sample = 0.03
            surface_ship.radar.turn_on()
            surface_ship.radar.beamPower = 10 * 1e6
            path0 = np.array([[900, 500, 0],
                              [100, 600, 0],
                              [500, 100, 0]])
            surface_ship.navigator.set_path1(path0)

        battleField.simulator()

        print('end!')

    if env == 'air-ground':
        # 导弹架的导弹
        ground = Get_Unit('ground_radar')
        ground_Missile_speed = ground.unit_data.get('ground_to_air_missile', {}).get('max_rate')


        RedFighter = Get_Unit('RedFighter')
        # 红方战斗机追踪目标最大速度(km/h)2410
        RedFighter_tracking_speed = RedFighter.unit_data['max_rate_for_tracking']  # todo需要这个吗？
        # 红方战斗机追踪目标时的加速度(km/s**2)0.075
        RedFighter_tracking_acc = RedFighter.unit_data['max_acceleration_for_tracking'] / 1000  # todo
        # 导弹速度(km/h)7200
        # todo暂时设置红方蓝方导弹速度相同
        Plane_Missile_speed = RedFighter.unit_data.get('air_or_ship_to_ship_missile', {}).get('max_rate')
        battleField = BattleField('air-ground',
                                  timer=0,
                                  size=[1000, 1000, 10],
                                  acceleration=100,
                                  frameTimeInterval=0.02,
                                  simTimeInterval=1,
                                  simDuration=60 * 180,  # 180min
                                  )
        redEarlyWarningAircraft = generating_unit('EWA', 'red', 'air', battleField, 1)

        redEarlyWarningAircraft.jammer.on = False
        redEarlyWarningAircraft.monitor.on = False
        redEarlyWarningAircraft.radar.turn_on()
        for radarBeam in redEarlyWarningAircraft.radar.beamList:
            # radarBeam.set_maxEffectiveDistance(400)
            # radarBeam.set_currentEffectiveDistance(400)
            radarBeam.set_angleRange(360)
            # radarBeam.set_pitchRange([-60,60])

        # path0 = np.random.randint(0, 1000, size=(10, 3))#10行2列
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])

        # path = list(path)

        # 如果预警机被击落则其他飞机无法到达目的地，因为只给预警机设置了path
        redEarlyWarningAircraft.navigator.set_path1(list(path0))
        # redEarlyWarningAircraft.navigator.set_path(list(path0))

        redJammingAircraft = generating_unit('EJA', 'red', 'air', battleField, 1)

        redJammingAircraft.jammer.set_maxEffectiveDistance(400)
        # 设置干扰机干扰角度 unit类中self.jammer
        redJammingAircraft.jammer.set_beamAngleRange(1)
        redJammingAircraft.jammer.set_beamPitchRange([-60, 60])
        path0 = np.array([[100, 600, 7],
                          [900, 500, 9],
                          [500, 100, 8]])

        # path = list(path)

        # 如果预警机被击落则其他飞机无法到达目的地，因为只给预警机设置了path
        # redJammingAircraft.navigator.set_path1(list(path0))

        for i in range(5):
            redPlane = generating_unit('RedFighter', 'red', 'air', battleField, i)
            redPlane.navigator.v_sample = 0.03

        ground_radar = generating_unit('ground_radar', 'blue', 'ground', battleField, 1)
        ground_radar.jammer.on = False
        ground_radar.monitor.on = False
        # destroyer.fightingFlag = False
        ground_radar.radar.turn_on()
        ground_radar.radar.beamPower = 10 * 1e6
        # ground_radar.jammer.set_maxEffectiveDistance(700)
        # 设置干扰机干扰角度 unit类中self.jammer
        ground_radar.jammer.set_beamAngleRange(10)
        ground_radar.jammer.set_beamPitchRange([-70, 70])
        for radarBeam in ground_radar.radar.beamList:
            radarBeam.set_angleRange(360)
        # path0 = np.array([[900, 500, 9],
        #                   [100, 600, 7],
        #                   [500, 100, 8]])
        # # 如果预警机被击落则其他飞机无法到达目的地，因为只给预警机设置了path
        ground_radar.navigator.set_path1(list(path0))

        for i in range(10):
            ground_air_defense_missile_rack = generating_unit('ground_air_defense_missile_rack', 'blue', 'ground', battleField, i)
            ground_air_defense_missile_rack.jammer.on = True
            ground_air_defense_missile_rack.jammer.beamPower = 10 * 1e6
            ground_air_defense_missile_rack.monitor.on = False
            ground_air_defense_missile_rack.radar.turn_on()
            ground_air_defense_missile_rack.radar.beamPower = 10 * 1e6
            ground_air_defense_missile_rack.jammer.set_maxEffectiveDistance(700)
            # 设置干扰机干扰角度 unit类中self.jammer
            ground_air_defense_missile_rack.jammer.set_beamAngleRange(1)
            ground_air_defense_missile_rack.jammer.set_beamPitchRange([-70, 70])

            # path0 = np.array([[900, 500, 9],
            #                   [100, 600, 7],
            #                   [500, 100, 8]])
            # # ground_air_defense_missile_rack.navigator.set_path1(path0)

        battleField.simulator()

        print('end!')
