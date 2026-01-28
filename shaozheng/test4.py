"""
2026-01-28

任务： 测试gemini在给定物理参数配置下能否生成双星交会对接场景。

在basilisk主目录下:
source .venv/bin/activate
cd shaozheng
python -m test4

第一次生成脚本：出现多余的功能幻觉
报错内容：
    vizSupport.addSpacecraftToViz(viz, chaserSc)
AttributeError: module 'Basilisk.utilities.vizSupport' has no attribute 'addSpacecraftToViz'
原因：
AI不清楚basilisk的向Vizsupport加入多星的函数调用规范，开始虚构代码。
解决办法：
找示例样本scenarioRendezVous.py,其中说明了vizsupport怎么看多个物体。
examples/mujoco/scenarioArmWithThrusters.py 这个脚本更加细致，但没使用vizsupport

第二次生成脚本: 
编译问题1:多import了一个没被使用过的orbitalMotion函数
解决办法：
无伤大雅，删掉就行。

场景问题1: 两颗卫星好像一直在地球的两侧，在仿真时间里面没有追上。
原因分析：
1. 是中间表示生成的问题，中间表示中，AI描述两颗卫星初始在轨道对面，相对距离远，后面追踪卫星会慢慢加速过去交汇。
- 语义理解的问题，AI认为交汇对接的初始状态是两颗卫星在轨道上相对地心对称分布。但实际上用户可能希望它们初始的相对距离不超过10km或者更近。
- 环境搭建者的核心职责是：定义不同的初始相位和轨道高度差，从而产生天然的“相位追赶”现象。可以调整初始状态，让两颗卫星由于轨道高度不同（导致周期不同）而产生自然的靠近过程。

2. 是大模型将中间表示用basilisk代码实现的能力不足，没设计机动追踪逻辑。
- 场景应该提供支持航天器机动的物理引擎，而具体的机动控制和规划应该交给用户做，并非场景搭建的任务范畴

场景问题2: 场景轨道线、名字都没在vizard中渲染出来。
报错代码：
# 自定义 Vizard 设置
viz.settings.orbitLinesOn = 2         # 显示轨道线
viz.settings.mainCameraTarget = "Target-Debris" # 镜头默认对准目标
# 为两个卫星指定不同的模型名（Vizard 内置模型）
viz.scData[0].modelDictionaryKey = "lsat"     # Chaser
viz.scData[1].modelDictionaryKey = "sat_box"  # Target
解决办法：
去掉这些幻觉代码。

第三次生成脚本: 
问题1: basilisk的数值积分卡住了。
原因：遗忘了之前的函数API示例样本，新代码中没有显式建立进程。dynProcess = scSim.CreateNewProcess(simProcessName) 这一步被遗忘。

第四次生成脚本：
问题1: 出现多余代码    mu = earth.mu  该变量根本没被使用过。
问题2: 返回结果显示最小相对距离还是太远了。
```bash
--- 仿真沙盒报告 ---
最小相对距离: 13122.18 km
最近窗口时间: 29.50 小时
```
AI分析: 导致这个结果的原因并不是参数“不合理”，而是因为**“追赶速度”与“初始差距”不匹配**。每绕地球一圈，跑得快的目标星（内圈）只能追上追踪星（外圈）约 $1.2^\circ$ 的相位。
只仿真了 $30$ 小时，而它们相遇需要约 $217$ 小时（约 $9$ 天）。所以在 $30$ 小时结束时，它们依然隔得非常远。
1. 缩小初始相位
2. 加大轨道高度差

试验：增加仿真时间
增加到200小时：
- 最小相对距离: 4373.84 km
- 最近窗口时间: 199.83 小时
改为300小时：
- 最小相对距离: 24.03 km
- 最近窗口时间: 256.67 小时
结论: AI会根据初始的两根轨道六根数进行一个相遇时间推算，但是这种推算与使用动力学函数计算的结果存在误差。
不同大模型的推算结果存在明显误差。qiwen-max(松鼠agent)推算是28.5小时，远小于仿真结果。这导致仿真场景的运行时间无法覆盖到最合适的交汇对接窗口。
可以提炼出：
场景时间正确性，生成的仿真时长能否覆盖到最合理的对接窗口。
关键参数： 
轨道高度差
初始相位差
场景仿真时长

"""



import os
import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion, 
                                simIncludeGravBody, vizSupport, unitTestSupport)

def run_rendezvous_sandbox():
    # 1. 创建仿真容器
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)
    
    simProcessName = "simProcess"
    simTaskName = "simTask"

    # 2. 创建进程
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # 3. 创建任务并指定步长 (严格对照样本：60.0秒)
    simulationTimeStep = macros.sec2nano(60.0)
    # 显式使用 addTask 确保调度器对齐
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # 4. 初始化航天器
    chaser = spacecraft.Spacecraft(); chaser.ModelTag = "Chaser-Sat"
    target = spacecraft.Spacecraft(); target.ModelTag = "Target-Sat"
    scSim.AddModelToTask(simTaskName, chaser)
    scSim.AddModelToTask(simTaskName, target)

    # 5. 设置引力体 (使用 gravBodyFactory)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    earth = gravFactory.createEarth()
    earth.isCentralBody = True
    gravFactory.addBodiesTo(chaser)
    gravFactory.addBodiesTo(target)

    # 6. 配置初始状态向量 (J2000 坐标系)
    # 目标卫星 (Target)
    target.hub.r_CN_NInit = np.array([6778.137, 0.0, 0.0]) * 1000.0
    target.hub.v_CN_NInit = np.array([0.0, 7.6726, 0.0]) * 1000.0
    # 追踪卫星 (Chaser)
    chaser.hub.r_CN_NInit = np.array([-6689.23, 1178.45, 0.0]) * 1000.0
    chaser.hub.v_CN_NInit = np.array([-1.328, -7.548, 0.0]) * 1000.0

    # 计算采样间隔
    samplingTime = macros.sec2nano(300.0)
    # 创建记录器并关联到任务
    chaserRec = chaser.scStateOutMsg.recorder(samplingTime)
    targetRec = target.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, chaserRec)
    scSim.AddModelToTask(simTaskName, targetRec)


    # 7. 可视化配置 (对照样本：先创建目录)
    if not os.path.exists("_VizFiles"):
        os.makedirs("_VizFiles")
    if vizSupport.vizFound:
        # 使用官方支持的列表方式
        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, [chaser, target], 
                                                  saveFile="LEO_Rendezvous")

    # 8. 初始化与执行
    scSim.InitializeSimulation()
    simulationTime = macros.hour2nano(300.0)
    scSim.ConfigureStopTime(simulationTime)
    
    print(f"正在启动 300 小时轨道相位仿真...")
    scSim.ExecuteSimulation()
    print(f"仿真顺利完成！")

    posChaser = chaserRec.r_BN_N
    posTarget = targetRec.r_BN_N
    relDist = np.linalg.norm(posChaser - posTarget, axis=1)
    
    minDist = np.min(relDist)
    minTimeHrs = chaserRec.times()[np.argmin(relDist)] * macros.NANO2SEC / 3600.0

    print(f"\n--- 仿真沙盒报告 ---")
    print(f"最小相对距离: {minDist/1000.0:.2f} km")
    print(f"最近窗口时间: {minTimeHrs:.2f} 小时")

if __name__ == "__main__":
    run_rendezvous_sandbox()