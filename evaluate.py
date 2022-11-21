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
tableUid=p.loadURDF("table/table.urdf",basePosition=[0.5,0,-0.623])
Slot=p.loadURDF("FYPLONG/slotmodel1/URDF/slotmodel1.urdf", useFixedBase=True, basePosition=[0.72, 0, 0.014], baseOrientation=orientation_wall2)

objectUid=p.loadURDF("FYPLONG/boltmodel/urdf/boltmodel.urdf", basePosition=[0.72, 0, 0], baseOrientation=orientation_wall1)


# for i in range(1000):
#     p.stepSimulation()
#     time.sleep(0.1)

state_durations=[1,1,1,1,1,1,1,1,1,1,1,1,1,1]
control_dt=1./400.
p.setTimeStep=control_dt
state_t=0.
current_state=0

while True:

    # contacts = p.getContactPoints(bodyA=objectUid,bodyB=Slot)
    # for contact in contacts:
    #     print("Normal force(N) =",contact[9])
    #     print("lateralFriction(N) =",contact[10])

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
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, force=4)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, force=4)

    if current_state == 3:
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, force=2)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, force=2)
        p.setJointMotorControl2(pandaUid, 0, p.POSITION_CONTROL, -0.03768353675293228)
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL,  0.9354370671055676)
        p.setJointMotorControl2(pandaUid, 2, p.POSITION_CONTROL, -0.039335972449535586)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -1.4196412458642376)
        p.setJointMotorControl2(pandaUid, 4, p.POSITION_CONTROL,  -0.004408789461858773)
        p.setJointMotorControl2(pandaUid, 5, p.POSITION_CONTROL, 2.3574932065375562)
        p.setJointMotorControl2(pandaUid, 6, p.POSITION_CONTROL, -0.7853974286414079)
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, 7.57895048848115e-06)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, 3.958874649043139e-05)

    #
    if current_state == 4:
        p.setJointMotorControl2(pandaUid, 0, p.POSITION_CONTROL, -0.036911820711122596)
        p.setJointMotorControl2(pandaUid, 1, p.POSITION_CONTROL, 0.8814366471825353)
        p.setJointMotorControl2(pandaUid, 2, p.POSITION_CONTROL, -0.04026271852142396)
        p.setJointMotorControl2(pandaUid, 3, p.POSITION_CONTROL, -1.4378333357243476)
        p.setJointMotorControl2(pandaUid, 4, p.POSITION_CONTROL, -0.004480523040845715)
        p.setJointMotorControl2(pandaUid, 5, p.POSITION_CONTROL, 2.3291540993858453)
        p.setJointMotorControl2(pandaUid, 6, p.POSITION_CONTROL, -0.7853973334497285)
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, -4.099395010698602e-09)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, 3.4851039636359997e-06)
        p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, force=1)
        p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, force=1)

        # (-0.03768353675293228, 0.9354370671055676, -0.039335972449535586, -1.4196412458642376, -0.004408789461858773,
        #  2.3574932065375562, -0.7853974286414079, 7.57895048848115e-06, 3.958874649043139e-05)
        # numActiveThreads = 0num_joints = p.getNumJoints(pandaUid)
        # current_pos = p.getLinkState(pandaUid, num_joints - 1)[4]
        #
        #
        # new_robot_pos = [
        #     0.72, -0.054,
        #     0.03
        # ]
        #
        # robot_joint_positions = p.calculateInverseKinematics(
        #     bodyIndex=pandaUid,
        #     endEffectorLinkIndex=num_joints - 1,
        #     targetPosition=[
        #         new_robot_pos[0], new_robot_pos[1],
        #         new_robot_pos[2]
        #     ]
        # )
        #
        # print(robot_joint_positions)
        #
        # for i in range(9):
        #     p.resetJointState(
        #         bodyUniqueId=pandaUid,
        #         jointIndex=i,
        #         targetValue=robot_joint_positions[i],
        #             )

    if state_t > state_durations[current_state]:
        current_state += 1
        if current_state > len(state_durations):
            current_state = 0
        state_t = 0

    rotate = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0., 0.0, math.pi / 2.])),
                      dtype=float).reshape(3, 3)
    # force = np.array(p.getJointState(pandaUid, 2)[2][0:3], dtype=float).reshape(3, 1)
    # force = np.dot(rotate, force)
    # print(force)
    # contacts = p.getContactPoints(bodyA=pandaUid, bodyB=objectUid)
    # for contact in contacts:
    #     print("Normal force(N) =", contact[9])
    #     print("lateralFriction1(N) =", contact[10])
    #     print("lateralFriction2(N) =", contact[12])



    p.stepSimulation()