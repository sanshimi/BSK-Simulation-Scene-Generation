"""
2026-01-26

任务： 测试gemini在给定物理参数配置下生成basilisk代码的能力。

在basilisk主目录下:
source .venv/bin/activate
cd shaozheng
python -m test3

第一次生成脚本的报错：
尝试给一个对象赋值一个它并不存在的属性。 
报错代码: earthGrav.bodyInMsgName = "earth_planet_data"
各种函数调用不规范。

解决方法：
找一个正确调用函数API的样本test2.py。让AI模仿去写LEO仿真场景的代码。

第二次生成的报错：
还是属性限制错误，同时因为样本里面没有这一块。
报错代码：scObject.hub.r_CN_N = [7000.0 * 1000, 0.0, 0.0]  # 米
scObject.hub.v_CN_N = [0.0, 7.5 * 1000, 0.0]     # 米/秒

解决方法：
引入更健壮的函数API调用示例样本scenarioBasicOrbit.py

第三次生成：脚本通过编译

"""


import os
import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody, vizSupport

def run(show_plots):
    # 1. 创建仿真容器
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    
    # 显示进度条
    scSim.SetProgressBar(True)

    # 2. 创建进程和任务
    dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(1.0) # 步长设为 1.0 秒
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # 3. 初始化航天器对象
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "LEO-Satellite"
    scSim.AddModelToTask(simTaskName, scObject)

    # 4. 设置引力体 (使用 gravBodyFactory)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    mu = earth.mu
    
    # 将引力体关联到航天器
    gravFactory.addBodiesTo(scObject)

    # 5. 配置你提供的 LEO 轨道参数
    oe = orbitalMotion.ClassicElements()
    oe.a = 7178.0 * 1000.0          # 半长轴 (m)
    oe.e = 0.001                    # 偏心率
    oe.i = 45.0 * macros.D2R        # 倾角
    oe.Omega = 90.0 * macros.D2R    # 升交点赤经
    oe.omega = 0.0 * macros.D2R     # 近地点幅角
    oe.f = 0.0 * macros.D2R         # 真近点角 (由 M=0 推导)

    # 转换为状态向量
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    # 6. 开启可视化 (使用 vizSupport)
    # 这会在当前目录下生成一个 .bin 文件供 Vizard 回放
    if True:
        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject, 
                                                  saveFile="LEO_Simulation")
        # 如果需要实时流传输，可以取消注释下面这行
        # viz.liveStream = True

    # 7. 设置仿真时间 (运行 1 个轨道周期)
    n = np.sqrt(mu / oe.a**3)
    P = 2. * np.pi / n
    simulationTime = macros.sec2nano(P)

    # 8. 初始化并运行
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    print(f"仿真完成。轨道周期: {P/60:.2f} 分钟")
    return

if __name__ == "__main__":
    run(show_plots=True)