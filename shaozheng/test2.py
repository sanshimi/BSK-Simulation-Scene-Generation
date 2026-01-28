#
# trajectory_to_vizard.py
#
# 读取 DRO.csv：
#   - 第0列: time_sec（秒）
#   - 第1-6列: 地球 (x, y, z, vx, vy, vz)
#   - 第7-12列: 月球
#   - 第13-18列: 卫星 (x, y, z, vx, vy, vz)
#
# 生成 Basilisk 日志（.data + .bin），供 Vizard 可视化。
#

import os
import numpy as np
import pandas as pd

from Basilisk.simulation import spacecraft
from Basilisk.architecture import messaging
from Basilisk.utilities import SimulationBaseClass, macros, vizSupport

# ----------------------------
# 用户配置
# ----------------------------
DATA_FILE = "DRO.csv"
OUTPUT_NAME = "satellite_traj"
DT_LOG = 1.0  # 日志记录步长（秒），可与数据采样率一致

def read_trajectory(file_path):
    """读取 DRO.csv 并提取卫星轨迹"""
    df = pd.read_csv(file_path, header=None)

    if df.shape[1] < 19:
        raise ValueError("CSV 至少需要 19 列：time + Earth(6) + Moon(6) + Satellite(6)")

    times = df.iloc[:, 0].values.astype(float)
    positions = df.iloc[:, 13:16].values.astype(float)  # x, y, z
    velocities = df.iloc[:, 16:19].values.astype(float) # vx, vy, vz

    if not np.all(np.diff(times) > 0):
        raise ValueError("时间列必须严格递增")

    return times, positions, velocities

class TrajectoryPlayer:
    """在每一步插值并更新卫星状态消息"""
    def __init__(self, times, positions, velocities):
        self.times = times
        self.positions = positions
        self.velocities = velocities
        self.scStateOutMsg = messaging.SCStatesMsg()

    def updateState(self, current_time_nano):
        t = current_time_nano * macros.NANO2SEC
        r, v = self.interpolate_state(t)
        msg = messaging.SCStatesMsgPayload()
        msg.r_BN_N = r.tolist()
        msg.v_BN_N = v.tolist()
        msg.sigma_BN = [0.0, 0.0, 0.0]      # 无姿态，单位四元数对应
        msg.omega_BN_B = [0.0, 0.0, 0.0]
        self.scStateOutMsg.write(msg)

    def interpolate_state(self, sim_time):
        times = self.times
        if sim_time <= times[0]:
            return self.positions[0], self.velocities[0]
        if sim_time >= times[-1]:
            return self.positions[-1], self.velocities[-1]

        idx = np.searchsorted(times, sim_time) - 1
        t0, t1 = times[idx], times[idx + 1]
        w = (sim_time - t0) / (t1 - t0)
        r = self.positions[idx] + w * (self.positions[idx + 1] - self.positions[idx])
        v = self.velocities[idx] + w * (self.velocities[idx + 1] - self.velocities[idx])
        return r, v

def run():
    times, positions, velocities = read_trajectory(DATA_FILE)
    print(f"加载 {len(times)} 个轨迹点，时间范围: [{times[0]}, {times[-1]}] 秒")

    sim = SimulationBaseClass.SimBaseClass()
    sim.SetProgressBar(True)

    simTaskName = "mainTask"
    processName = "process"
    dynProcess = sim.CreateNewProcess(processName)
    taskRate = macros.sec2nano(DT_LOG)
    dynProcess.addTask(sim.CreateNewTask(simTaskName, taskRate))

    # 创建航天器
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "DRO_Satellite"
    scObject.setDynamicsSkip(True)  # 关键：跳过动力学积分

    # 初始化状态（任意值，会被后续 update 覆盖）
    scObject.hub.r_CN_N = positions[0].tolist()
    scObject.hub.v_CN_N = velocities[0].tolist()

    sim.AddModelToTask(simTaskName, scObject)

    # 自定义任务：每步更新状态
    class StateUpdater:
        def __init__(self, times, positions, velocities):
            self.times = times
            self.positions = positions
            self.velocities = velocities

        def UpdateState(self, current_time_nano):
            t = current_time_nano * macros.NANO2SEC
            r, v = self.interpolate_state(t)
            scObject.hub.r_CN_N = r.tolist()
            scObject.hub.v_CN_N = v.tolist()

        def interpolate_state(self, sim_time):
            times = self.times
            if sim_time <= times[0]:
                return self.positions[0], self.velocities[0]
            if sim_time >= times[-1]:
                return self.positions[-1], self.velocities[-1]
            idx = np.searchsorted(times, sim_time) - 1
            t0, t1 = times[idx], times[idx + 1]
            w = (sim_time - t0) / (t1 - t0)
            r = self.positions[idx] + w * (self.positions[idx + 1] - self.positions[idx])
            v = self.velocities[idx] + w * (self.velocities[idx + 1] - self.velocities[idx])
            return r, v

    updater = StateUpdater(times, positions, velocities)
    sim.AddModelToTask(simTaskName, updater, ModelPriority=99)  # 高优先级，早于 scObject

    # 设置仿真时长
    SIM_DURATION = times[-1] + DT_LOG
    sim.ConfigureStopTime(macros.sec2nano(SIM_DURATION))
    sim.SetTimeStep(taskRate)

    # 启用 Vizard
    vizSupport.enableUnityVisualization(sim, simTaskName, scObject)

    # 添加 recorder（可选，用于 .data/.bin）
    dataLog = scObject.scStateOutMsg.recorder()
    sim.AddModelToTask(simTaskName, dataLog)

    # 运行
    sim.InitializeSimulation()
    sim.ExecuteSimulation()

    # 重命名日志
    try:
        os.rename(dataLog.logFileName + ".data", OUTPUT_NAME + ".data")
        os.rename(dataLog.logFileName + ".bin", OUTPUT_NAME + ".bin")
        print(f"日志已保存为: {OUTPUT_NAME}.data 和 {OUTPUT_NAME}.bin")
    except Exception as e:
        print("日志重命名失败:", e)

    print("可在 Vizard 中打开 _VizFiles/ 或 .bin 文件进行可视化！")

if __name__ == "__main__":
    run()