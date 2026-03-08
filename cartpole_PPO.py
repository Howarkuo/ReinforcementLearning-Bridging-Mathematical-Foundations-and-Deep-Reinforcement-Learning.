#  REINFORCE-> Actor Critic Architecture
# actor: fc_pi
# critic: fc_v

# states - 
# W relu- O - 
# softmax - 
# action probs - 
# loss (action / rewards )

# Architecture
# 1. Environment learning interaction: s,r done, info = env.step(a)
# 2. dl framework : torch.n and torch.optim.adam
# 3. function approximation: neural network to map states to value 
# 4. temporal discouting : gamma to prioritize immediate reward over future rewards

# output: probability distribution : handles both discrete and continious actiosn

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Patch for modern numpy versions
np.bool8 = bool

# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

from gym.wrappers import RecordVideo

def save_trained_video(model):
    print("\nRecording video of the trained agent...")
    
    # 1. Create the environment with rgb_array mode for video capture
    video_folder = './results/videos'
    os.makedirs(video_folder, exist_ok=True)
    
    # We use 'rgb_array' so the computer captures the frames internally
    temp_env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    # 2. Wrap the environment with RecordVideo
    # This will save a video of the first episode played in this env
    env = RecordVideo(temp_env, video_folder, episode_trigger=lambda x: True)
    
    s, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Use your trained model to pick the best action
        prob = model.pi(torch.from_numpy(s).float())
        action = torch.argmax(prob).item() 
        
        s_prime, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        s = s_prime
        total_reward += r
        
    print(f"Video saved! Final score: {total_reward}")
    env.close()

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20
    history = []

    print("Starting CPU training (PPO)...")
    for n_epi in range(601): # Capped for quick demo
        s, _ = env.reset()
        done = False
        curr_score = 0
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime
                score += r
                curr_score += r
                if done: break
            model.train_net()
        
        history.append(curr_score)

        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = score/print_interval
            print(f"# Episode: {n_epi}, Avg Score: {avg_score:.1f}")
            score = 0.0
            if avg_score > 490: # Consider solved
                print("Environment Solved!")
                break

    env.close()

    # --- SAVE PERFORMANCE GRAPH ---
    os.makedirs('./results', exist_ok=True)
    plt.plot(history)
    plt.title('PPO CartPole-v1 Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('./results/ppo_performance.png')
    print("Graph saved to ./results/ppo_performance.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    save_trained_video(model) 

    # --- RENDER TRAINED AGENT (LIVE) ---
    print("\nVisualizing trained agent (Live Window)...")
    # ... [Rest of your live visualization code] ...

    # --- RENDER TRAINED AGENT ---
    print("\nVisualizing trained agent...")
    render_env = gym.make('CartPole-v1', render_mode="human")
    s, _ = render_env.reset()
    for _ in range(500):
        render_env.render()
        time.sleep(0.02)
        prob = model.pi(torch.from_numpy(s).float())
        a = torch.argmax(prob).item() # Pick best action
        s, r, terminated, truncated, info = render_env.step(a)
        if terminated or truncated: break
    render_env.close()

if __name__ == '__main__':
    main()

# Episode: 20, Avg Score: 27.2
# Episode: 40, Avg Score: 26.4
# Episode: 60, Avg Score: 30.6
# Episode: 80, Avg Score: 52.2
# Episode: 100, Avg Score: 47.6
# Episode: 120, Avg Score: 94.8
# Episode: 140, Avg Score: 138.8
# Episode: 160, Avg Score: 209.0
# Episode: 180, Avg Score: 334.3
# Episode: 200, Avg Score: 284.6
# Episode: 220, Avg Score: 124.7
# Episode: 240, Avg Score: 192.5
# Episode: 260, Avg Score: 242.7
# Episode: 280, Avg Score: 152.3
# Episode: 300, Avg Score: 131.3
# Episode: 320, Avg Score: 141.0
# Episode: 340, Avg Score: 188.2
# Episode: 360, Avg Score: 291.5
# Episode: 380, Avg Score: 224.6
# Episode: 400, Avg Score: 274.4
# Episode: 420, Avg Score: 287.1
# Episode: 440, Avg Score: 284.4
# Episode: 460, Avg Score: 219.3
# Episode: 480, Avg Score: 325.3
# Episode: 500, Avg Score: 281.2
# Episode: 520, Avg Score: 418.4
# Episode: 540, Avg Score: 425.1
# Episode: 560, Avg Score: 379.2
# Episode: 580, Avg Score: 350.4
# Episode: 600, Avg Score: 193.8
