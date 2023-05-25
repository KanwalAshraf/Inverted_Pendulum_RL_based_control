import gym
import pybullet_envs
import numpy as np
import pybullet as p
import torch
import imageio
import torch.nn as nn
import torch.optim as optim
from InvertedPendulumTendon_Env import InvertedPendulumTendonEnv
# Define the neural network architecture
import cv2
import gym
# Define the codec and create a VideoWriter object
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
        state = state.unsqueeze(0) # unsqueeze the tensor to add a batch dimension
        with torch.no_grad():
            action = self.net(state)
        return action.numpy()[0] # remove the batch dimension before returning the action
        '''
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.net(state)
        action=action.cpu().numpy()[0]
        return action
        '''
    def get_available_action(self, state):
        actions=[]
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.net(state)
            actions.append(action.cpu().numpy()[0])
        return actions

    '''
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = self.net(state)
        return action.numpy()
    '''
    

    def train(self, max_episodes=100, max_steps=1000):
        epis=[]
        rewarrd=[]
        for i_episode in range(max_episodes):
            state = self.env.reset()
            total_reward = 0.0
            prev_reward=0
            last_three_entries = [None, None, None]
            for t in range(max_steps):
                
                prev_action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(prev_action)
                '''
                if reward<0:
                    reward=reward/100
                else:
                    reward=reward
                '''

                '''
                # Set up video writer
                frame = p.getCameraImage(800, 600)[2]
                writer.append_data(frame)
                '''

                self.optimizer.zero_grad()
                loss = -torch.mean(self.net(torch.tensor(state, dtype=torch.float32)) * torch.tensor(prev_action, dtype=torch.float32))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                '''
                if prev_reward>=reward:
                    state=state
                else:
                    state = next_state
                    '''
                state = next_state
                prev_reward=reward*0.1
                total_reward += reward
                # add the current entry to the list

    # remove the oldest entry from the list
    # check if all entries are equal
                
                if done:
                    break
            #writer.close()
            print("Episode {}: Total reward = {:.2f}".format(i_episode+1, total_reward))

            epis.append(i_episode+1)
            rewarrd.append(total_reward)
        return epis,rewarrd
# Test the agent
    def test(self, n_episodes=100):
        rewards = []
        episodes = []
        state = self.env.reset()
        max_steps=1000
        for i in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            for t in range(max_steps):
                # Randomly select an action from available actions
                available_actions = self.get_available_action(state)
                available_actions = np.ravel(available_actions)
                action = np.random.choice(available_actions)
                action=action.flatten()
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
            episodes.append(i)
        return episodes, rewards






# Create the environment and the agent
#env = gym.make("InvertedPendulumKanwalltenGUI2-v78")
env = gym.make('InvertedPendulumTendonEnv-v78')
#env.render()
agent = Agent(env)
# Reset the environment
obs = env.reset()
import matplotlib.pyplot as plt
# Train the agent
ep,re=agent.train()
# Save the model
torch.save(agent.net.state_dict(), 'model_new.pt')
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
'''
# Test the agent
epi,rre=agent.test()
plt.figure()
plt.plot(epi,rre)
plt.show()
'''
