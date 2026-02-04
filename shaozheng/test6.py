import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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


def run(show_plots=True):

    # ==========================================================
    # 1. 仿真基础配置
    # ==========================================================
    simTaskName = "dynTask"
    simProcessName = "dynProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)

    dynProcess = scSim.CreateNewProcess(simProcessName)
    timeStep = macros.sec2nano(120.0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, timeStep))

    simTime = macros.day2nano(20)

    # ==========================================================
    # 2. 引力环境 + SPICE
    # ==========================================================
    gravFactory = simIncludeGravBody.gravBodyFactory()

    earth = gravFactory.createEarth()
    earth.isCentralBody = True

    moon = gravFactory.createMoon()
    moon.isCentralBody = False

    muEarth = earth.mu
    muMoon = moon.mu

    startTimeUTC = "2026 January 04 15:00:00.0"
    timeInit = datetime.strptime(startTimeUTC, "%Y %B %d %H:%M:%S.%f")

    spice = gravFactory.createSpiceInterface(
        time=startTimeUTC,
        epochInMsg=True
    )
    spice.zeroBase = 'Earth'
    scSim.AddModelToTask(simTaskName, spice)

    pyswice.furnsh_c(spice.SPICEDataPath + "de430.bsp")
    pyswice.furnsh_c(spice.SPICEDataPath + "naif0012.tls")
    pyswice.furnsh_c(spice.SPICEDataPath + "pck00010.tpc")

    # ==========================================================
    # 3. GEO 观测卫星
    # ==========================================================
    geo = spacecraft.Spacecraft()
    geo.ModelTag = "GEO_Obs"

    geo.hub.mHub = 3000
    geo.hub.IHubPntBc_B = np.diag([2000, 2200, 1800])

    gravFactory.addBodiesTo(geo)
    scSim.AddModelToTask(simTaskName, geo)

    oe_geo = orbitalMotion.ClassicElements()
    oe_geo.a = 42164e3
    oe_geo.e = 5e-5
    oe_geo.i = 0.05 * macros.D2R
    oe_geo.Omega = 60 * macros.D2R
    oe_geo.omega = 0.0
    oe_geo.f = 0.0

    r_geo, v_geo = orbitalMotion.elem2rv(muEarth, oe_geo)

    geo.hub.r_CN_NInit = r_geo
    geo.hub.v_CN_NInit = v_geo
    geo.hub.sigma_BNInit = [[0], [0], [0]]
    geo.hub.omega_BN_BInit = [[0], [0], [0]]

    # ==========================================================
    # 4. LLO 目标卫星（月心 → 地心）
    # ==========================================================
    llo = spacecraft.Spacecraft()
    llo.ModelTag = "LLO_Target"

    llo.hub.mHub = 600
    llo.hub.IHubPntBc_B = np.diag([300, 280, 250])

    gravFactory.addBodiesTo(llo)
    scSim.AddModelToTask(simTaskName, llo)

    oe_llo = orbitalMotion.ClassicElements()
    oe_llo.a = 1838e3
    oe_llo.e = 0.005
    oe_llo.i = 10 * macros.D2R
    oe_llo.Omega = 30 * macros.D2R
    oe_llo.omega = 0.0
    oe_llo.f = 180 * macros.D2R

    rLLO_M, vLLO_M = orbitalMotion.elem2rv(muMoon, oe_llo)

    moonState = 1000 * spkRead(
        'moon',
        startTimeUTC,
        'J2000',
        'earth'
    )

    rMoon = moonState[0:3]
    vMoon = moonState[3:6]

    llo.hub.r_CN_NInit = rMoon + rLLO_M
    llo.hub.v_CN_NInit = vMoon + vLLO_M
    llo.hub.sigma_BNInit = [[0], [0], [0]]
    llo.hub.omega_BN_BInit = [[0], [0], [0]]

    # ==========================================================
    # 5. GEO → LLO 建链检测
    # ==========================================================
    access = spacecraftLocation.SpacecraftLocation()
    access.ModelTag = "GEO_LLO_Access"

    access.primaryScStateInMsg.subscribeTo(geo.scStateOutMsg)
    access.addSpacecraftToModel(llo.scStateOutMsg)

    access.aHat_B = [1, 0, 0]
    access.theta = np.deg2rad(25.0)
    access.maximumRange = 6.0e8

    access.rEquator = earth.radEquator
    access.rPolar = earth.radEquator * 0.996

    scSim.AddModelToTask(simTaskName, access)

    # ==========================================================
    # 6. 数据记录
    # ==========================================================
    accessRec = access.accessOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, accessRec)

    # ==========================================================
    # 7. 可视化（可选）
    # ==========================================================
    if vizSupport.vizFound:
        viz = vizSupport.enableUnityVisualization(
            scSim,
            simTaskName,
            [geo, llo],
            saveFile="GEO_LLO_Link"
        )

    # ==========================================================
    # 8. 运行仿真
    # ==========================================================
    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simTime)
    scSim.ExecuteSimulation()

    # ==========================================================
    # 9. 通信窗口统计
    # ==========================================================
    times = accessRec.times() * macros.NANO2SEC
    accessFlag = accessRec.hasAccess

    windows = []
    inAccess = False

    for t, a in zip(times, accessFlag):
        if a and not inAccess:
            t0 = t
            inAccess = True
        elif not a and inAccess:
            windows.append((t0, t))
            inAccess = False

    if inAccess:
        windows.append((t0, times[-1]))

    print("\nGEO → LLO 通信窗口：")
    for i, (t0, t1) in enumerate(windows):
        print(
            f"Window {i+1}: "
            f"{timeInit + timedelta(seconds=t0)} → "
            f"{timeInit + timedelta(seconds=t1)} "
            f"({(t1-t0)/60:.1f} min)"
        )

    # ==========================================================
    # 10. Access 时间轴
    # ==========================================================
    if show_plots:
        plt.figure()
        plt.step(times/3600, accessFlag, where='post')
        plt.xlabel("Time since start [hours]")
        plt.ylabel("Access")
        plt.title("GEO → LLO Access Timeline")
        plt.grid(True)
        plt.show()

    gravFactory.unloadSpiceKernels()


if __name__ == "__main__":
    run(True)
