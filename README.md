# Self-learning robotics research on Deep reinforcement learning
A final year project of a student in university of birmingham 2021

Environmental set-up: 
VMware + Linux Ubuntu20.04 + Mujoco 2.0 + Python3.9 + Anaconda3 + Pycharm + Mujoco_py + Robosuite + Pybullet + Gym +Open3D + H5py + Pygame


The specific steps are shwon below with the solution to the common problems:

1. The installation of VMware and Linux Ubuntu20.04 can be seen from https://blog.csdn.net/qq_45642410/article/details/113756950 and the point is that the storage should be more than 20GB at least .

2. The rest of the steps on other softwares are shown in https://shirkozlovsky1.medium.com/setup-robosouite-pycharm-enviroment-on-linux-5dd0d3b8e4f0.
But there are lot of problems to be mentioned:

For the Mujoco, at last, please enter in the terminal ./simulate ../model/humanoid.xml on the path /.mujoco/mujoco200/bin

For the Python3.9 and Anaconda3, it is extreme significant to create a new environment (the edition of the python should be larger than 3.8) in its navigator and install the rest of softewares in the terminal like this one,where Robotics is the name of the new environment.

![image](https://user-images.githubusercontent.com/82950147/147616678-cbd72c1f-96b9-40b2-8d47-1fa34eb2d432.png)

For the pycharm, the point is the setting of the environmental variables. The specific steps are shown here.

![image](https://user-images.githubusercontent.com/82950147/147616864-c655ccaf-1e45-468f-b156-00903b441692.png)
![image](https://user-images.githubusercontent.com/82950147/147616884-243546e3-3857-41af-a353-ecf96984d5f7.png)
![image](https://user-images.githubusercontent.com/82950147/147616911-82df1331-630f-4e7e-a8c7-8d16b0825f57.png)

However, the libgrew.so name here can be different for various computers. It can be the libGLEW.so or libGLEW2.so. And the path of these two in the picture should be the actual path in your computer.

For the Mujoco_py, it is also necessary to install the gcc module by:
~sudo apt update
~sudo apt install build-essential
~sudo apt-get install manpages-dev
~gcc --version
To validate the gcc.


For the Robosuite, it is highlighted that the requirements-extra.txt is necessary to be used to install the Gym +Open3D + H5py + Pygame by:
~pip3 install -r requirements-extra.txt
By the way, for non-Mac OSX, put in a comment thehidap line is important.

For the run of the demos (.py) in Robosuite documents in Pycharm, there are many other library should be added if the system ask:
1. patchelf
~sudo add-apt-repository ppa:jamesh/snap-support
~sudo apt-get update
~sudo apt install patchelf

.......

Then, all the preparation is done. The next step is to modify the environment in Robosuite.


