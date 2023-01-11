import numpy as np
import pybullet as p
import pybullet_data as pd
import math
import time

p.connect(p.GUI)
p.setGravity(0,0,-10)


# p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=0,\
                             cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])

orientation_wall1 = p.getQuaternionFromEuler([math.pi/2,0, 0])
orientation_wall2 = p.getQuaternionFromEuler([math.pi,0,0])

# kuka=p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf",useMaximalCoordinates=True)
# kuka=p.loadURDF("kuka_iiwa/model_free_base.urdf",useFixedBase=True)
# kuka=p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf",basePosition=[0.9,0,-0.63])
pandaUid=p.loadURDF("franka_panda/panda.urdf",useFixedBase=True)
tableUid=p.loadURDF("table/table.urdf",basePosition=[0.5,0,-0.63])
Slot=p.loadURDF("slotmodel1/URDF/slotmodel1.urdf",useFixedBase=True,basePosition=[0.72,0,0.014],baseOrientation=orientation_wall2)

objectUid=p.loadURDF("boltmodel/urdf/boltmodel.urdf",basePosition=[0.72,0,0],baseOrientation=orientation_wall1)


# for i in range(1000):
#     p.stepSimulation()
#     time.sleep(0.1)

state_durations=[1,1,1,1]
control_dt=1./240.
p.setTimeStep=control_dt
state_t=0.
current_state=0

while True:

    contacts = p.getContactPoints(bodyA=objectUid,bodyB=Slot)
    for contact in contacts:
        print("Normal force(N) =",contact[9])
        print("lateralFriction(N) =",contact[10])

    # p.applyExternalForce(objectUid, 0, (math.pi, 0, 0), (0, 0, 0), p.WORLD_FRAME)

    # for joint_index in range(joint_num):
    #     info_tuple = p.getJointInfo(robot_id, joint_index)

    state_t += control_dt
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

    if current_state == 0:
        p.setJointMotorControl2(pandaUid, 0, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL, math.pi / 4.)
        p.setJointMotorControl2(pandaUid, 2, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -math.pi / 2.)
        p.setJointMotorControl2(pandaUid, 4, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(pandaUid, 5, p.POSITION_CONTROL, 3 * math.pi / 4)
        p.setJointMotorControl2(pandaUid, 6, p.POSITION_CONTROL, -math.pi / 4.)
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, 0.08)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, 0.08)

    if current_state == 1:
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL, math.pi / 4. + .15)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -math.pi / 2. + .15)

    if current_state == 2:
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, force=5000)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, force=5000)

    if current_state == 3:
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL, math.pi / 4. - 1)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -math.pi / 2. - 1)

    if state_t > state_durations[current_state]:
        current_state += 1
        if current_state >= len(state_durations):
            current_state = 0
        state_t = 0

    rotate = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0., 0.0, math.pi / 2.])),
                      dtype=float).reshape(3, 3)
    # force = np.array(p.getJointState(pandaUid, 2)[2][0:3], dtype=float).reshape(3, 1)
    # force = np.dot(rotate, force)
    # print(force)
    contacts = p.getContactPoints(bodyA=pandaUid, bodyB=objectUid)
    for contact in contacts:
        print("Normal force(N) =", contact[9])
        print("lateralFriction1(N) =", contact[10])
        print("lateralFriction2(N) =", contact[12])



    p.stepSimulation()