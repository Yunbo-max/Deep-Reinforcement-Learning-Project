# import pybullet as p
# import pybullet_data #pybullet自带的一些模型
# p.connect(p.GUI) #连接到仿真环境，p.DIREACT是不显示仿真界面,p.GUI则为显示仿真界面
# p.setGravity(0,0,-10) #设定重力
# p.resetSimulation() #重置仿真环境
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
# planeId = p.loadURDF("plane.urdf") #加载外部平台模型
# #objects = p.loadURDF('Slot/urdf/Slot.urdf') #加载机械臂，flags=9代表取消自碰撞，详细教程可以参考pybullet的官方说明文档
# objects = p.loadURDF('Bolt6/urdf/Bolt6.urdf') #加载机械臂，flags=9代表取消自碰撞，详细教程可以参考pybullet的官方说明文档
# tableUid = p.loadURDF("table/table.urdf", basePosition=[0,0.3,-0.45],globalScaling=1)
# while 1:
#  p.stepSimulation()

import pybullet as p
import pybullet_data #pybullet自带的一些模型
import random

p.connect(p.GUI) #连接到仿真环境，p.DIREACT是不显示仿真界面,p.GUI则为显示仿真界面
p.setGravity(0,0,-10) #设定重力
p.resetSimulation() #重置仿真环境
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
planeId = p.loadURDF("plane.urdf") #加载外部平台模型
objects = p.loadURDF('RL_arm_under_sparse_reward_main/URDF_model/bmirobot_description/urdf/robotarm_description.urdf', flags=9) #加载机械臂，flags=9代表取消自碰撞，详细教程可以参考pybullet的官方说明文档
tableUid = p.loadURDF("table/table.urdf", basePosition=[0, 0.3, -0.45], globalScaling=1)
# loadUDRF 可以制定放置地点，比例等等
# 随即方块的位置
xpos = 0.15 + 0.2 * random.random()  # 0.35
ypos = (random.random() * 0.3) + 0.2  # 0.10 0.50
zpos = 0.2
ang = 3.14 * 0.5 + 3.1415925438 * random.random()
orn = p.getQuaternionFromEuler([0, 0, ang])
# 随机目标方块的位置（在桌子上）
xpos_target = 0.35 * random.random()  # 0.35
ypos_target = (random.random() * 0.3) + 0.2  # 0.10 0.50
zpos_target = 0.2
ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
orn_target = p.getQuaternionFromEuler([0, 0, ang_target])

blockUid = p.loadURDF("RL_arm_under_sparse_reward_main/URDF_model/cube_small_push.urdf", xpos, ypos, zpos,
                      orn[0], orn[1], orn[2], orn[3])
targetUid = p.loadURDF("RL_arm_under_sparse_reward_main/URDF_model/cube_small_target_push.urdf",
                       [xpos_target, ypos_target, zpos_target],
                       orn_target, useFixedBase=1)



for i in range(24):
     with open('bmirobot_joints_info_pybullet.txt','a') as f:
          f.write(str(p.getJointInfo(objects, i)))



while 1:
    p.stepSimulation()







