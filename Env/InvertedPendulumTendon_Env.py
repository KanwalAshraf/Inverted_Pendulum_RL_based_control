import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

cubeStartPos = [-2.15,0,.75]
cubeStartPos2 = [0,0,1.4]
cubeStartPos3 = [2.15,0,.75]
PulleyStartOrientation = p.getQuaternionFromEuler([1.570796, 0, 0]) 
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0]) 
cubeStartOrientation2 = p.getQuaternionFromEuler([0,-1.570796,0])
class InvertedPendulumTendonEnv(gym.Env):
    def __init__(self, render=True):
        self._render = render
        self.noise_level=10
        
        # Set up the simulation environment
        p.connect(p.GUI if self._render else p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Load the plane
        p.loadURDF("plane.urdf")

        # Load the pendulum with strings
        
        self._base_1 = p.loadURDF("Base_1.urdf",cubeStartPos3, cubeStartOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
        self._base_2 = p.loadURDF("Base_2.urdf",cubeStartPos, cubeStartOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
        self._pendulum = p.loadURDF("Pendulum_Tendon_1_Cart_Rail.urdf",cubeStartPos2, cubeStartOrientation2, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
        
        # Set the action and observation spaces
        action_bound = 1.0
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=(1,), dtype=np.float32)
        obs_bound = np.inf
        self.observation_space = spaces.Box(low=-obs_bound, high=obs_bound, shape=(2,), dtype=np.float32)

        self._num_joints = 1  # Define the number of joints
        nJoints = p.getNumJoints(self._base_1)  #Base 1: magenta base and tendon
        jointNameToId = {}
        for i in range(nJoints):
            jointInfo = p.getJointInfo(self._base_1, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
        Base_pulley_1 = jointNameToId['Base_pulley1']
        nJoints = p.getNumJoints(self._pendulum)
        jointNameToId = {}
        for i in range(nJoints):
            jointInfo = p.getJointInfo(self._pendulum, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
        last_tendon_link_1 = jointNameToId['tendon1_13_tendon1_14']
        cart_pendulumAxis = jointNameToId['cart_pendulumAxis']
        cart = jointNameToId['slider_cart']
        nJoints = p.getNumJoints(self._base_2)  #Base 2: white base and tendon
        jointNameToId = {}
        for i in range(nJoints):
            jointInfo = p.getJointInfo(self._base_2, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
        last_tendon_link_2 = jointNameToId['tendon1_13_tendon1_14']
        Base_pulley_2 = jointNameToId['Base_pulley1']
        pulley_1_tendon_magenta = p.createConstraint(self._base_1, Base_pulley_1, self._pendulum, last_tendon_link_1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0], [-.56, 0, 0])
        tendon_white_cart = p.createConstraint(self._base_2, last_tendon_link_2, self._pendulum, cart, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0], [0,-.55, 0])
        p.setJointMotorControl2(self._pendulum, cart_pendulumAxis, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self._base_1, Base_pulley_1, p.VELOCITY_CONTROL, targetVelocity=10, force=1000) #Base 1: magenta base and tendon
        p.setJointMotorControl2(self._base_2, Base_pulley_2, p.VELOCITY_CONTROL, targetVelocity=10, force=-1000)#Base 2: white base and tendon


        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def custom_reward(self, state, action, next_state):
        # Calculate the deviation from the target angle
        deviation = np.abs(next_state[0])

        # Calculate the deviation from the target velocity
        velocity_deviation = np.abs(next_state[1])

        # Calculate the reward
        if deviation < 0.1 and velocity_deviation < 0.1:
            reward = 0.5*((1-math.cos(deviation))-(velocity_deviation))
        else:
            reward = -0.1

        return reward



    def step(self, action):
        # Apply the action torques to the joint
        # Start recording the simulation

        #p.setJointMotorControl2(self._pendulum, 0, p.TORQUE_CONTROL, force=action[0])# for every case
        p.setJointMotorControl2(self._pendulum, 0, p.TORQUE_CONTROL, force=action.item())# for buffer case
        #p.setJointMotorControl2(self._base_1, 0, p.VELOCITY_CONTROL, force=action[0]) #Base 1: magenta base and tendon
        #p.setJointMotorControl2(self._base_2, 0, p.VELOCITY_CONTROL, force=action[0])
        

        # Step the simulation forward
        p.stepSimulation()

        # Get the joint angle and velocity as the observation
        joint_state = p.getJointState(self._pendulum, 0)
        obs = np.array([joint_state[0], joint_state[1]])

        # Calculate the reward
        #reward = -obs[0]**2 - 0.1*obs[1]**2
        reward =self.custom_reward(obs,action,obs)

        # Check if the episode is done
        done = False

        # Return the step information
        return obs, reward, done, {}

       

    def reset(self):
    # Reset the simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.noise_level=0

    # Load the plane
        p.loadURDF("plane.urdf")

    # Load the cart and rail
        self._cart_rail = p.loadURDF("Pendulum_Tendon_1_Cart_Rail.urdf",cubeStartPos2, cubeStartOrientation2, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
        self._base_1 = p.loadURDF("Base_1.urdf", cubeStartPos3, cubeStartOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)
        self._base_2 = p.loadURDF("Base_2.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)

        # Randomize the joint positions and velocities
        for i in range(self._num_joints):
            pos = self.np_random.uniform(low=-0.1, high=0.1)
            vel = self.np_random.uniform(low=-0.1, high=0.1)
            p.resetJointState(self._cart_rail, i, pos, vel)

        # Get the initial observation
        obs = []
        for i in range(self._num_joints):
            joint_state = p.getJointState(self._cart_rail, i)
            obs.append(joint_state[0])
            obs.append(joint_state[1])

        obs = np.array(obs)

        return obs

import gym
from gym.envs.registration import register
# Register the InvertedPendulumEnv environment
register(
    id='InvertedPendulumTendonEnv-v78',
    entry_point='InvertedPendulumTendon_Env:InvertedPendulumTendonEnv',
)
