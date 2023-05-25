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
# Define the codec and create a VideoWriter object
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
def modify_action_with_delay(action, delay):
    # Wait for the specified delay
        time.sleep(delay)
    
    # Return the modified action
        return action
'''
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

            '''
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
        self.reward_buf=[]
        self.buffer=[]

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
        powerr=[]
        loss_pack=[]
        loss_generic=[]
        inter_gap=[]
        power_average=[]
        gap_inter=[]
        pak_loss=[]
        
        with open('mydata_maxreward_500_packet_new_action_q_learn.csv', mode='w', newline='') as csv_file:
            fieldnames = ['angle','velocity', 'power','SNR','gap','delay','loss','reward'] # define the field names for the CSV file
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            self.gamma=0.1

            buffer = []
            buffer_size = 10 # choose a size appropriate for your application
            # write the header row in the CSV file
            writer.writeheader()
            for i_episode in range(max_episodes):
                state = self.env.reset()
                total_reward = 0.0
                prev_reward=0
                last_three_entries = [None, None, None]
                for t in range(max_steps):
                    action = self.select_action(state)
                    
                    #next_state, reward, done, _ = self.env.step(action)
                    #packet_loss_rate=0.5
                    #wireless_channel(env, state)
                    
                    # Add current experience to buffer
                    
                        
                    # Sample experiences from buffer and train on them
                    if len(buffer) == buffer_size:
                        self.optimizer.zero_grad()
                        batch = random.sample(buffer, 1)
                        batch_states = torch.tensor([x[0] for x in batch], dtype=torch.float32)
                        batch_actions = torch.tensor([x[1] for x in batch], dtype=torch.float32)
                        batch_rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                        batch_next_states = torch.tensor([x[3] for x in batch], dtype=torch.float32)
                        batch_dones = torch.tensor([x[4] for x in batch], dtype=torch.float32)
                        q_values = self.net(batch_states)
                        q_values_actions = torch.sum(q_values * batch_actions, dim=1)
                        next_q_values = self.net(batch_next_states)
                        max_next_q_values = torch.max(next_q_values, dim=1)[0]
                        expected_q_values = batch_rewards + (1 - batch_dones) * self.gamma * max_next_q_values
                        loss = torch.mean((q_values_actions - expected_q_values.detach()) ** 2)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                        if t%500==0:
                            power=10
                            action, packet_lost,delay,snr,gap = wireless_channel(self.env, action,state[0],state[1],power)
                            next_state, reward, done, _ = self.env.step(action)
                            self.reward_buf.append(reward)
                            print(self.reward_buf)

                            # Update the current state and reward
                            state = next_state
                            loss_pack.append(packet_lost)
                            inter_gap.append(gap)
                            powerr.append(power)
                        else:
                            packet_lost=False
                            gap=False
                            delay=0
                            power=20
                            next_state, reward, done, _ = self.env.step(action)
                            self.reward_buf.append(reward)
                            print(self.reward_buf)
                            loss_pack.append(packet_lost)
                            inter_gap.append(gap)
                            powerr.append(power)
                        state = next_state
                        writer.writerow({'angle': state[0],'velocity':state[1], 'power':power,'SNR':snr,'gap':gap,'delay':delay,'loss': packet_lost,'reward':total_reward})
                        buffer.append((state, action, reward, next_state, done))


                        # Update the total reward
                        if not (packet_lost) and not (gap):
                            total_reward += reward
                            prev_reward=reward
                        if packet_lost or gap:
                            total_reward += prev_reward
                        if len(buffer) > buffer_size:
                            # Remove oldest experience from buffer
                            buffer.pop(0)
                        if len(self.buffer) > 10:
                            self.buffer.pop(0)
                            self.reward_buf.pop(0)
                    

                    #state = next_state
                    #total_reward += reward
                        if packet_lost or gap:
                                max_reward_index = np.argmax(self.reward_buf)
                                action_with_max_reward = self.buffer[max_reward_index]
                                state=action_with_max_reward
                                #self.buffer[-1] = state
                                #self.reward_buf[-1]=reward
                                print("Packet is lost or a gap is introduced")

                        if done:
                            break


                        most_common_packet_lost = min(set(loss_pack), key=loss_pack.count)
                        most_common_gap = min(set(inter_gap), key=inter_gap.count)
                        # Calculate the average power value
                        average_power = sum(powerr) / len(powerr)
                        pak_loss.append(most_common_packet_lost)
                        gap_inter.append(most_common_gap)
                        power_average.append(average_power)
                    print("Episode {}: Total reward = {:.2f}".format(i_episode+1, total_reward))

                    epis.append(i_episode+1)
                    rewarrd.append(total_reward)


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
import csv
ep,re,pak_loss, gap_inter, power_average=agent.train()

# Combine the arrays into a list of tuples
data = list(zip(ep, re,pak_loss, gap_inter, power_average))

# Define the file path and name
filename = 'maxreward_500%_packet_new_run_episodes_action_q.csv'

# Write the data to a CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'reward','Packet_Loss','InterPacket_gap','Average_Power_consumed'])
    for row in data:
        writer.writerow(row)
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

# Train the agent
torch.save(agent.net.state_dict(), 'model_noisy_q.pt')
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
