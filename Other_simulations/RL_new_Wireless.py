import gym
import pybullet_envs
import numpy as np
import pybullet as p
import torch
import imageio
import torch.nn as nn
import torch.optim as optim
from InvertedPendulumTendon_Env import InvertedPendulumTendonEnv
from network import simulate_channel_loss
import time
import random
# Define the neural network architecture
import cv2
import csv
import gym
from datetime import datetime
# Define the codec and create a VideoWriter object
def modify_action_with_delay(action, delay):
    # Wait for the specified delay
        time.sleep(delay)
    
    # Return the modified action
        return action

def wireless_channel(env, action):
    # Simulate delay
        delay_prob = random.uniform(0, 1)
        if delay_prob < 0.2:
        # Simulate packet loss
            loss_prob = random.uniform(0, 1)
            if loss_prob < 0.2:
                # Packet lost
                now=str(datetime.now())
                print(f"Packet lost at time {now}")
                # Return original action
                return action
            else:
                # Packet delayed but not lost
                now=str(datetime.now())
                delay = 0
                print(f"Packet delayed by {delay} seconds at time {now}")
                # Return modified action with delay
                return modify_action_with_delay(action, delay)
        else:
        # No delay or loss
            return action
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
#writer = imageio.get_writer('output.mp4', fps=60)
# Define the agent
class Agent:
    def __init__(self, env):
        self.env = env
        self.net = Net(env.observation_space.shape[0], 64, env.action_space.shape[0])
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0.0001)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.net(state)
        #noise = np.random.normal(scale=0.1, size=env.action_space.shape[0])
        action = np.clip(action.numpy(), env.action_space.low, env.action_space.high)
        return action
    

    


    def train(self, max_episodes=100, max_steps=1000):
        epis=[]
        rewarrd=[]
        with open('mydata_maxreward_no_loss.csv', mode='w', newline='') as csv_file:
            fieldnames = ['angle','velocity','reward'] # define the field names for the CSV file
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
            # write the header row in the CSV file
            writer.writeheader()
            for i_episode in range(max_episodes):
                state = self.env.reset()
                total_reward = 0.0
                for t in range(max_steps):
                    
                    action = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    packet_loss_rate=0.5
                    #wireless_channel(env, state)

                    self.optimizer.zero_grad()
                    loss = -torch.mean(self.net(torch.tensor(state, dtype=torch.float32)) * torch.tensor(action, dtype=torch.float32))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    state = next_state
                    total_reward += reward
                    writer.writerow({'angle': state[0],'velocity':state[1],'reward':total_reward})
                    if done:
                        break
                print("Episode {}: Total reward = {:.2f}".format(i_episode+1, total_reward))
                epis.append(i_episode+1)
                rewarrd.append(total_reward)
            return epis,rewarrd
    def test(self, num_episodes=100):
        epis=[]
        rewarrd=[]
        for i_episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break
            print("Episode {}: Total reward = {:.2f}".format(i_episode+1, total_reward))
            epis.append(i_episode+1)
            rewarrd.append(total_reward)
        return epis,rewarrd
'''
class Agent:
    def __init__(self, env):
        self.env = env
        self.net = Net(env.observation_space.shape[0], 64, env.action_space.shape[0])
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.net(state)
        return action.numpy()

    def train(self, max_episodes=100, max_steps=1000):
        epis=[]
        rewarrd=[]
        for i_episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0.0
            for t in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Set up video writer
                frame = p.getCameraImage(800, 600)[2]
                writer.append_data(frame)
                
                self.optimizer.zero_grad()
                loss = -torch.mean(self.net(torch.tensor(state, dtype=torch.float32)) * torch.tensor(action, dtype=torch.float32))
                loss.backward()
                self.optimizer.step()
                state = next_state
                total_reward += reward
                if done:
                    break
            #writer.close()
            print("Episode {}: Total reward = {:.2f}".format(i_episode+1, total_reward))
            epis.append(i_episode+1)
            rewarrd.append(total_reward)
        return epis,rewarrd
'''

# Create the environment and the agent
#env = gym.make("InvertedPendulumKanwalltenGUI2-v78")
env = gym.make('InvertedPendulumTendonEnv-v78')
#env.render()
agent = Agent(env)
# Reset the environment
obs = env.reset()

# Run the simulation for 1000 time steps and render the environment at each step
'''
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break

# Close the environment
env.close()
'''
import matplotlib.pyplot as plt
ep,re=agent.train()
# Train the agent
torch.save(agent.net.state_dict(), 'model_noisy.pt')

import csv


# Combine the arrays into a list of tuples
data = list(zip(ep, re))

# Define the file path and name
filename = 'maxreward_RL.csv'

# Write the data to a CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'reward'])
    for row in data:
        writer.writerow(row)
# Load the saved model

#saved_model = torch.load('model_new.pt')
#agent.net.load_state_dict(saved_model)

maximum=min(re)
new_reward=re/abs(maximum)
plt.figure()
plt.plot(ep,new_reward)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()

# Test the agent
#epi,rre=agent.test()
#plt.figure()
#plt.plot(epi,rre)
#plt.show()
