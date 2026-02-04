"""
第一次生成：基于脚本scenarioSpacecraftLoaction.py
1. 在basilisk中非地心轨道怎么建立
参考 scenarioLagrangePointOrbit.py 学会添加月球和月球根据星历运动
2. 第一次IR没有给出时间范围参数
AI缺乏训练以及合适样本参考的情况下真写不来（涉及到了多个脚本，目前是人与AI协作写）

第二次生成：增加对于通信窗口的检测
1. 一直没有通信窗口
一般地球与月球直接直接建立窗口就比较困难，由于地球阻挡，信号波束1度过窄，仿真时间过短等等。
所以这个脚本应该以功能性验证为主，能检测到极短窗口就行了。
2. 本场景设计的自然语言交互经过了多次复杂对话，中间过程目前不够清晰直观，使用户记忆有困难，不利于复现。主要难点也在于代码迭代不够清晰，每次迭代都是人为引导。
3. basilisk/examples/MultiSatBskSim 有更加复杂的多卫星编队模拟架构。
4. scenarioDataToViz.py 有一个从定义好的csv转viz的范例。
"""


import numpy as np
import matplotlib.pyplot as plt

from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
    vizSupport
)
from Basilisk.simulation import spacecraft, spacecraftLocation
from Basilisk.topLevelModules import pyswice
from Basilisk.utilities.pyswice_spk_utilities import spkRead
from datetime import datetime, timedelta


def run(show_plots=True):

    # ==================================================
    # 1. 仿真基础设置
    # ==================================================
    simTaskName = "simTask"
    simProcessName = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)
    dynProcess = scSim.CreateNewProcess(simProcessName)

    timeStep = macros.sec2nano(180.0)   # 180 秒
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, timeStep))

    simTime = macros.day2nano(30)    # 仿真 30 天

    # ==================================================
    # 2. 引力环境：地球 + 月球
    # ==================================================
    gravFactory = simIncludeGravBody.gravBodyFactory()

    earth = gravFactory.createEarth()
    earth.isCentralBody = True

    moon = gravFactory.createMoon()
    moon.isCentralBody = False

    muEarth = earth.mu
    muMoon = moon.mu

    # UTC 初始时间（全仿真时间锚点）
    timeInitString = "2026 January 04 15:00:00.0"
    spiceTimeFormat = "%Y %B %d %H:%M:%S.%f"
    timeInit = datetime.strptime(timeInitString, spiceTimeFormat)

    spiceObject = gravFactory.createSpiceInterface(
        time=timeInitString,
        epochInMsg=True
    )
    spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(simTaskName, spiceObject, 1)

    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'de430.bsp')
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'naif0012.tls')
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'de-403-masses.tpc')
    pyswice.furnsh_c(spiceObject.SPICEDataPath + 'pck00010.tpc')

    # ==================================================
    # 3. GEO 卫星
    # ==================================================
    geo = spacecraft.Spacecraft()
    geo.ModelTag = "GEO"

    geo.hub.mHub = 3500.0
    geo.hub.IHubPntBc_B = np.diag([2500., 1800., 2000.])

    gravFactory.addBodiesTo(geo)
    scSim.AddModelToTask(simTaskName, geo)

    oe_geo = orbitalMotion.ClassicElements()
    oe_geo.a = 42164e3
    oe_geo.e = 1e-4
    oe_geo.i = 0.1 * macros.D2R
    oe_geo.Omega = 90.0 * macros.D2R
    oe_geo.omega = 0.0
    oe_geo.f = 0.0

    rGEO, vGEO = orbitalMotion.elem2rv(muEarth, oe_geo)
    geo.hub.r_CN_NInit = rGEO
    geo.hub.v_CN_NInit = vGEO
    geo.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    geo.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

    # ==================================================
    # 4. LLO 卫星
    # ==================================================
    llo = spacecraft.Spacecraft()
    llo.ModelTag = "LLO"

    llo.hub.mHub = 500.0
    llo.hub.IHubPntBc_B = np.diag([300., 250., 200.])

    gravFactory.addBodiesTo(llo)
    scSim.AddModelToTask(simTaskName, llo)

    oe_llo = orbitalMotion.ClassicElements()
    oe_llo.a = (1738.0 + 100.0) * 1e3
    oe_llo.e = 0.01
    oe_llo.i = 90.0 * macros.D2R
    oe_llo.Omega = 0.0
    oe_llo.omega = 0.0
    oe_llo.f = 0.0

    rLLO_M, vLLO_M = orbitalMotion.elem2rv(muMoon, oe_llo)

    moonState = 1000 * spkRead(
        'moon',
        timeInitString,
        'J2000',
        'earth'
    )

    rMoon_N = moonState[0:3]
    vMoon_N = moonState[3:6]

    llo.hub.r_CN_NInit = rMoon_N + rLLO_M
    llo.hub.v_CN_NInit = vMoon_N + vLLO_M
    llo.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    llo.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

    # ==================================================
    # 5. GEO → LLO 通信几何（Access）
    # ==================================================
    access = spacecraftLocation.SpacecraftLocation()
    access.ModelTag = "GEO_to_LLO_Access"

    access.primaryScStateInMsg.subscribeTo(geo.scStateOutMsg)
    access.addSpacecraftToModel(llo.scStateOutMsg)

    access.aHat_B = [1, 0, 0]
    # 放宽波束，增大范围
    access.theta = np.radians(20.0)
    access.maximumRange = 5.0e8

    access.rEquator = earth.radEquator
    access.rPolar = earth.radEquator * 0.996

    scSim.AddModelToTask(simTaskName, access)

    # ==================================================
    # 6. 数据记录
    # ==================================================
    accessRec = access.accessOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, accessRec)

    # ==================================================
    # 7. Unity 可视化
    # ==================================================
    if vizSupport.vizFound:
        viz = vizSupport.enableUnityVisualization(
            scSim,
            simTaskName,
            [geo, llo],
            saveFile="GEO_LLO_Access"
        )

        vizSupport.addLocation(
            viz,
            stationName="GEO_Antenna",
            parentBodyName="GEO",
            r_GP_P=[1.5, 0.0, 0.0],
            gHat_P=[1.0, 0.0, 0.0],
            fieldOfView=2 * access.theta,
            range=access.maximumRange,
            color="cyan"
        )

        viz.settings.showLocationCommLines = 1
        viz.settings.showLocationCones = 1
        viz.settings.showLocationLabels = 1

    # ==================================================
    # 8. 运行仿真
    # ==================================================
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simTime)
    scSim.ExecuteSimulation()

    # ==================================================
    # 9. 通信窗口 → UTC（实现 A）
    # ==================================================
    times = accessRec.times() * macros.NANO2SEC
    hasAccess = accessRec.hasAccess

    accessWindows = []
    inAccess = False
    tStart = 0.0

    for t, a in zip(times, hasAccess):
        if a and not inAccess:
            tStart = t
            inAccess = True
        elif not a and inAccess:
            accessWindows.append((tStart, t))
            inAccess = False

    if inAccess:
        accessWindows.append((tStart, times[-1]))

    print("\nGEO → LLO 通信窗口（UTC）：")
    for k, (t0, t1) in enumerate(accessWindows):
        utcStart = timeInit + timedelta(seconds=t0)
        utcEnd = timeInit + timedelta(seconds=t1)
        print(
            f"Window {k+1}: "
            f"{utcStart}  →  {utcEnd}   "
            f"({(t1-t0)/60:.1f} min)"
        )

    # ==================================================
    # 10. 可视化 Access 时间轴
    # ==================================================
    if show_plots:
        plt.figure()
        plt.plot(times/3600, hasAccess, drawstyle='steps-post')
        plt.xlabel("Time since start [hours]")
        plt.ylabel("Access (1 = Yes)")
        plt.title("GEO → LLO Communication Access")
        plt.grid(True)
        plt.show()

    # ==================================================
    # 11. 卸载 SPICE
    # ==================================================
    gravFactory.unloadSpiceKernels()
    pyswice.unload_c(spiceObject.SPICEDataPath + 'de430.bsp')
    pyswice.unload_c(spiceObject.SPICEDataPath + 'naif0012.tls')
    pyswice.unload_c(spiceObject.SPICEDataPath + 'de-403-masses.tpc')
    pyswice.unload_c(spiceObject.SPICEDataPath + 'pck00010.tpc')


if __name__ == "__main__":
    run(True)
