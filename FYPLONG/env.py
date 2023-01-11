#--coding:utf-8--
import numpy as np
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
import math
import cv2
import torch
import os


def random_crop(imgs, out):

    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


class KukaReachVisualEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    kMaxEpisodeSteps = 700
    kImageSize = {'width': 96, 'height': 96}
    kFinalImageSize = {'width': 84, 'height': 84}

    action_bound = [-1, 1]
    state_dim = 3
    action_dim = 6


    def __init__(self, is_render=True, is_good_view=False):

        #render part
        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)



        #edge definition
        self.x_low_obs = 0.65
        self.x_high_obs = 0.8
        self.y_low_obs = -0.1
        self.y_high_obs = 0.1
        self.z_low_obs = 0.01
        self.z_high_obs = 0.15

        #
        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = 0.01
        self.z_high_action = 0.05

        self.step_counter = 0

        #define the document path
        p.setAdditionalSearchPath(pd.getDataPath())
        self.urdf_root_path = pd.getDataPath()




        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05,
                             -3,-3,-3,-3,-3]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05,
                             3,3,3,3,3]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6,
                             6,6,6,6,6]
        # restposes for null space
        self.rest_poses = [0, math.pi / 4. + .15, 0, -math.pi / 2. + .15, 0, 3 * math.pi / 4,
               -math.pi / 4., 0, 0, 0, 0, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]
        # self.init_joint_positions = [
        #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
        #     -0.006539, 0.005,0,0,0,0
        # ]

        self.init_joint_positions = [
            0, math.pi / 4. + .15, 0, -math.pi / 2. + .15, 0, 3 * math.pi / 4,
               -math.pi / 4., 0, 0, 0.08, 0.08, 0
        ]
        #the end of effector is always downward
        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])



        #camera definition
        self.camera_parameters = {
            'width': 960.,
            'height': 720,
            'fov': 60,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.55, 0, 0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2)

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])


        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action]),
            high=np.array([
                self.x_high_action,
                self.y_high_action,
                self.z_high_action
            ]),
            dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(1, self.kFinalImageSize['width'], self.kFinalImageSize['height']))


        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_counter = 0

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, -10)

        # edge to minimize the operation frame
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        #Input the floor,table,manipulator,slot,object
        orientation_wall1 = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        orientation_wall2 = p.getQuaternionFromEuler([math.pi, 0, 0])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])

        self.kuka_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        table_uid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.62])

        self.slot = p.loadURDF("slotmodel1/urdf/slotmodel1.urdf", useFixedBase=True, basePosition=[0.723, 0, 0.02],
                               baseOrientation=orientation_wall2)
        self.object_id = p.loadURDF("boltmodel/urdf/boltmodel.urdf", basePosition=[0.723, 0, 0.01],
                                    baseOrientation=orientation_wall1)

        self.object_id2 = p.loadURDF("boltmodel/urdf/boltmodel.urdf", basePosition=[0.723, -0.08, 0.09],
                                     baseOrientation=orientation_wall1, useFixedBase=True)
        #change the color of the table
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])


        self.num_joints = p.getNumJoints(self.kuka_id)



        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )

        p.setJointMotorControl2(self.kuka_id, 9, p.POSITION_CONTROL, force=2)
        p.setJointMotorControl2(self.kuka_id, 10, p.POSITION_CONTROL, force=2)

        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]

        # p.applyExternalForce(objectUid, 0, (math.pi, 0, 0), (0, 0, 0), p.WORLD_FRAME)

        # for joint_index in range(joint_num):
        #     info_tuple = p.getJointInfo(robot_id, joint_index)







        rotate = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0., 0.0, math.pi / 2.])),
                          dtype=float).reshape(3, 3)
        # force = np.array(p.getJointState(pandaUid, 2)[2][0:3], dtype=float).reshape(3, 1)
        # force = np.dot(rotate, force)
        # print(force)
        # contacts = p.getContactPoints(bodyA=self.slot, bodyB=self.object_id)
        # for contact in contacts:
        #     print("Normal force(N) =", contact[9])
        #     print("lateralFriction1(N) =", contact[10])
        #     print("lateralFriction2(N) =", contact[12])
        # print("--------------------")
        # contacts = p.getContactPoints(bodyA=self.kuka_id, bodyB=self.object_id)
        # for contact in contacts:
        #     print("Normala force(N) =", contact[9])
        #     print("lateralFriction1a(N) =", contact[10])
        #     print("lateralFriction2a(N) =", contact[12])

        p.stepSimulation()


        (_, _, px, _,
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px

        p.enableJointForceTorqueSensor(bodyUniqueId=self.kuka_id,
                                       jointIndex=self.num_joints - 1,
                                       enableSensor=True)


        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        self.images = self.images[:, :, :
                                        3]  # the 4th channel is alpha channel, we do not need it.


        return self._process_image(self.images)

    def _process_image(self, image):
        """Convert the RGB pic to gray pic and add a channel 1

        Args:
            image ([type]): [description]
        """

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (self.kImageSize['width'], self.kImageSize['height']))[None, :, :] / 255.
            return image
        else:
            return np.zeros((1, self.kImageSize['width'], self.kImageSize['height']))


    def step(self, action):
        dv = 0.001
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.new_robot_pos = [
            self.current_pos[0] + dx, self.current_pos[1] + dy,
            self.current_pos[2] + dz
        ]
        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyIndex=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ],
            targetOrientation=self.orientation,        #the end of effector is always downward
            jointDamping=self.joint_damping,
            # lowerlimits=self.lower_limits,
            # upperLimits=self.upper_limits,
            restPoses=self.rest_poses,
            jointRanges=self.joint_ranges,
        )

        # print(self.robot_joint_positions)

        for i in range(9):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )

        p.stepSimulation()

        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1

        # contacts = p.getContactPoints(bodyA=self.slot, bodyB=self.object_id)
        # for contact in contacts:
        #     print("Normal force(N) =", contact[9])
        #     print("lateralFriction1(N) =", contact[10])
        #     print("lateralFriction2(N) =", contact[12])
        #
        # print("--------------------")
        #
        # contacts = p.getContactPoints(bodyA=self.kuka_id, bodyB=self.object_id)
        # for contact in contacts:
        #     print("Normala force(N) =", contact[9])
        #     print("lateralFriction1a(N) =", contact[10])
        #     print("lateralFriction2a(N) =", contact[12])

        return self._reward()

    def _reward(self):
        reward = 0

        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]

        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id2)[0]).astype(
            np.float32)

        square_dx = (self.robot_state[0] - self.object_state[0]) ** 2
        square_dy = (self.robot_state[1] - self.object_state[1]) ** 2
        square_dz = (self.robot_state[2] - self.object_state[2]) ** 2

        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy + square_dz)
        # print(self.distance)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        contacts1 = p.getContactPoints(bodyA=self.slot, bodyB=self.object_id)
        for contact in contacts1:
            if contact[9]>2 or abs(contact[10])>0.5 :
                reward = reward-0.1
                self.terminated = True

        contacts2 = p.getContactPoints(bodyA=self.kuka_id, bodyB=self.object_id)
        for contact in contacts2:
            if contact[9] ==0 :
                reward = reward - 0.5
                self.terminated = True
        # 如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)

        if terminated:
            reward = reward -0.1
            self.terminated = True

        # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.step_counter > self.kMaxEpisodeSteps:
            reward = reward -0.1
            self.terminated = True

        elif self.distance < 0.001:
            reward = reward + 1
            self.terminated = True
        else:
            reward = reward+ 0
            self.terminated = False

        info = {'distance:', self.distance}
        (_, _, px, _,
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px
        self.processed_image = self._process_image(self.images)
        # self.observation=self.robot_state
        self.observation = self.object_state
        return self.processed_image, reward, self.terminated, info

    def close(self):
        p.disconnect()


    def _get_force_sensor_value(self):
        force_sensor_value = p.getJointState(bodyUniqueId=self.kuka_id,
                                             jointIndex=self.num_joints -
                                                        1)[2][2]
        # the first 2 stands for jointReactionForces, the second 2 stands for Fz,
        # the pybullet methods' return is a tuple,so can not
        # index it with str like dict. I think it can be improved
        # that return value is a dict rather than tuple.
        return force_sensor_value


class CustomSkipFrame(gym.Wrapper):
    """ Make a 4 frame skip, so the observation space will change to (4,84,84) from (1,84,84)

    Args:
        gym ([type]): [description]
    """

    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(skip, self.kFinalImageSize['width'], self.kFinalImageSize['height']))
        self.skip = skip


    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return random_crop(states.astype(np.float32), self.kFinalImageSize['width']), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)],
                                0)[None, :, :, :]
        return random_crop(states.astype(np.float32), self.kFinalImageSize['width'])


if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    import matplotlib.pyplot as plt

    env = KukaReachVisualEnv(is_render=True)
    env = CustomSkipFrame(env)

    state = env.reset()
    print(state.shape)

    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    # print(env.action_space.n)
    for _ in range(200):
        action=env.action_space.sample()
        print(action)
        _,_,done,_= env.step(action)
        if done:
            env.reset()


    # img = state[0][0]
    # plt.imshow(img, cmap='gray')
    # plt.show()


