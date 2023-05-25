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
import gym
from datetime import datetime
from torch.distributions import Categorical

# Define the codec and create a VideoWriter object
def modify_action_with_delay(action, delay):
    # Wait for the specified delay
        time.sleep(delay)
    
    # Return the modified action
        return action
import time
def calculate_snr(power, noise_level):
    # Assume a fixed signal-to-noise ratio (SNR) of 10 dB
    snr = 10
    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr/10)
    # Calculate noise power based on noise level
    noise_power = 10**(noise_level/10)
    # Calculate signal power based on signal power
    signal_power = snr_linear * noise_power
    # Return SNR based on signal and noise power
    return 10*np.log10(signal_power/noise_power)

import csv


import numpy as np
import time
from datetime import datetime

def wireless_channel(env, action, angle,velocity, power):
    
    # Calculate SNR based on power of the signal and noise level
    snr = calculate_snr(power, env.noise_level)
    delay = 0
    flagg=0
    #prev_action=0
    # Simulate delay
    delay_probs = [0.2, 0.4, 0.6, 0.8] # Define delay probability for each state
    #current_state = int(state) # Get current state from the environment
    #delay_prob = delay_probs[current_state-1]

    # Simulate transmission gap based on SNR and packet loss probability
    gap_probs = [0.2, 0.4, 0.6, 0.8] # Define gap probability for each state
    current_state_0 = int(angle) # Get current state from the environment
    current_state_1 = int(velocity)
    gap_prob_0 = gap_probs[current_state_0-1]
    gap_prob_1 = gap_probs[current_state_1-1]
    
    if (np.random.uniform() < gap_prob_0) or (np.random.uniform() < gap_prob_1):
        # Transmission gap occurred
        now = str(datetime.now())
        print(f"Transmission gap occurred at time {now}")
        
        # Return original action and packet loss flag

        if flagg>=1:
            return prev_action, True, delay, snr, now
        else:
            flagg+=1
            prev_action=action
            return action, True, delay, snr, now
        
    else:
        # No transmission gap
        prev_state_0=angle
        prev_state_1=velocity
        return action, False, delay, snr, False


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
        self.buffer=[]
        self.reward_buf=[]

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.net(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.view(1)


    def train(self, max_episodes=100, max_steps=1000):
        epis=[]
        rewarrd=[]
        powerr=[]
        loss_pack=[]
        loss_generic=[]
        inter_gap=[]
        power_average=[]
        gap_inter=[]
        pak_loss=[]
        
        with open('mydata_maxreward_1_packet_new.csv', mode='w', newline='') as csv_file:
            fieldnames = ['angle','velocity', 'power','SNR','gap','delay','loss','reward'] # define the field names for the CSV file
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            done=False
            # write the header row in the CSV file
            writer.writeheader()
            for i_episode in range(max_episodes):
                state = self.env.reset()
                total_reward = 0.0
                
                
                for t in range(max_steps):
                    # Select an action
                    prev_reward=0
                    if len(self.buffer) > 0:
                        # Use the last stored state from the buffer
                        state = self.buffer[-1]
                        
                    action = self.select_action(state)
                    prev_state=state
                    # Pass the action through the wireless channel
                    

                    flag=0
                    
                    #else:
                        #gap=False
                        #delay=0

                    # Store the current state in the buffer
                    
                    if t%1==0:
                        power=10
                        action, packet_lost,delay,snr,gap = wireless_channel(self.env, action,state[0],state[1],power)
                        next_state=prev_state
                        reward=-1
                        self.reward_buf.append(-1)

                        # Update the current state and reward
                        state = next_state
                        loss_pack.append(packet_lost)
                        inter_gap.append(gap)
                        powerr.append(power)
                        
                    # If packet is lost, store the last state again
                    else:
                        packet_lost=False
                        gap=False
                        delay=0
                        power=20
                        next_state, reward, done, _ = self.env.step(action)
                        prev_state=next_state
                        self.reward_buf.append(reward)
                        loss_pack.append(packet_lost)
                        inter_gap.append(gap)
                        powerr.append(power)
                    if done:
                        break
                    state = next_state
                    self.buffer.append(state)
                    writer.writerow({'angle': state[0],'velocity':state[1], 'power':power,'SNR':snr,'gap':gap,'delay':delay,'loss': packet_lost,'reward':total_reward})
                        # Update the current state and reward

                        #packet_lost=False
                    # Take the action in the environment
                    
                    if packet_lost or gap:
                        max_reward_index = np.argmax(self.reward_buf)
                        action_with_max_reward = self.buffer[max_reward_index]
                        state=action_with_max_reward
                        #self.buffer[-1] = state
                        #self.reward_buf[-1]=reward
                        print("Packet is lost or a gap is introduced")

                    # Update the total reward
                    if not (packet_lost) and not (gap):
                        total_reward += reward
                        prev_reward=reward
                    if packet_lost or gap:
                        total_reward += prev_reward



                    # Check if buffer is full
                    if len(self.buffer) > 10:
                        self.buffer.pop(0)
                        self.reward_buf.pop(0)

                    # Update the network
                    self.optimizer.zero_grad()
                    loss = -torch.mean(self.net(torch.tensor(state, dtype=torch.float32)) * torch.tensor(action, dtype=torch.float32))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()



                # Add the total reward for the episode
                print("Episode {}: Total reward = {:.2f}".format(i_episode+1, total_reward))
                most_common_packet_lost = min(set(loss_pack), key=loss_pack.count)
                most_common_gap = min(set(inter_gap), key=inter_gap.count)
                # Calculate the average power value
                average_power = sum(powerr) / len(powerr)

                epis.append(i_episode+1)
                rewarrd.append(total_reward)
                pak_loss.append(most_common_packet_lost)
                gap_inter.append(most_common_gap)
                power_average.append(average_power)
        

        return epis, rewarrd, pak_loss, gap_inter, power_average




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
ep,re,pak_loss, gap_inter, power_average=agent.train()
# Train the agent
torch.save(agent.net.state_dict(), 'model_noisy_v1.pt')
# Load the saved model

#saved_model = torch.load('model_new.pt')
#agent.net.load_state_dict(saved_model)
import csv


# Combine the arrays into a list of tuples
data = list(zip(ep, re,pak_loss, gap_inter, power_average))

# Define the file path and name
filename = 'maxreward_1%_packet_new_run_episodes_state.csv'

# Write the data to a CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'reward','Packet_Loss','InterPacket_gap','Average_Power_consumed'])
    for row in data:
        writer.writerow(row)

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
