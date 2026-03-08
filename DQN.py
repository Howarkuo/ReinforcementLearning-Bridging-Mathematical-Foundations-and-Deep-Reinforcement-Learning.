# Reference: https://medium.com/aiii-ai/%E7%B5%90%E5%90%88%E5%BC%B7%E5%8C%96%E5%AD%B8%E7%BF%92dqn%E8%88%87%E9%81%8A%E6%88%B2%E9%96%8B%E7%99%BC-%E4%BD%BF%E7%94%A8python%E5%AF%A6%E7%8F%BE%E8%B2%AA%E9%A3%9F%E8%9B%87%E4%B8%A6%E8%A8%93%E7%B7%B4ai%E5%AD%B8%E7%BF%92%E9%81%8A%E6%88%B2%E7%AD%96%E7%95%A5-77685e794b72
# ref for papaer : https://hackmd.io/@YungHuiHsu/BJgnMHbUH6

# from Q-Table to Q-Network: low-dimention to high dimentional input

import gymnasium as gym
import os
import collections
import random
import matplotlib.pyplot as plt # For PNG saving
from gym.wrappers import RecordVideo # For video recording
import numpy as np
np.bool8 = bool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32
#class replayb-uffer()-put(): append transition , sample(): 
#class : Qnet(): Storage: nn.Module / State Input: Raw Continious Floats (0.123 , -0.45..) / Update Rule: optimizer.step() /
# Stability: ReplayBuffer / Target Net
# Module: [(W1 * x) + b1--> [ReLU] --> [(W2 * x) + b2] ]
# 

#function: put()- transition 
#function: sample()- pick random 32 

#function: train()
#function: main(): 

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
# 
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# --- NEW: VIDEO RECORDING FUNCTION ---
def save_video(q_model):
    print("\nRecording video...")
    video_path = './results/dqn_video'
    # Use rgb_array mode so it records internally without needing a monitor
    render_env = gym.make('CartPole-v1', render_mode="rgb_array")
    render_env = RecordVideo(render_env, video_folder=video_path, episode_trigger=lambda x: True)
    
    s, _ = render_env.reset()
    done = False
    while not done:
        # Use epsilon=0 to only use the best learned actions
        a = q_model.sample_action(torch.from_numpy(s).float(), epsilon=0)
        s, r, terminated, truncated, _ = render_env.step(a)
        done = terminated or truncated
    render_env.close()
    print(f"Video saved to {video_path}")

def main():
    os.makedirs('./results', exist_ok=True) # Create results folder
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    score_history = [] # For PNG performance graph
    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(601): # Reduced for demonstration, change back to 10000 if needed
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s, _ = env.reset()
        done = False
        epi_score = 0

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            epi_score += r
            if done:
                break
        
        score_history.append(epi_score) # Record every episode score
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print(f"n_episode :{n_epi}, score : {score/print_interval:.1f}, n_buffer : {memory.size()}, eps : {epsilon*100:.1f}%")
            
            # Early stop if solved
            if score/print_interval > 490:
                print("Environment Solved!")
                break
            score = 0.0

    env.close()

    # --- SAVE PERFORMANCE PNG ---
    plt.plot(score_history)
    plt.title('DQN CartPole-v1 Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('./results/dqn_performance.png')
    print("Graph saved to ./results/dqn_performance.png")
    
    # --- SAVE VIDEO ---
    save_video(q)

if __name__ == '__main__':
    main()


# n_episode :20, score : 9.8, n_buffer : 197, eps : 7.9%
# n_episode :40, score : 9.5, n_buffer : 387, eps : 7.8%
# n_episode :60, score : 9.4, n_buffer : 576, eps : 7.7%
# n_episode :80, score : 9.3, n_buffer : 762, eps : 7.6%
# n_episode :100, score : 9.4, n_buffer : 951, eps : 7.5%
# n_episode :120, score : 9.8, n_buffer : 1148, eps : 7.4%
# n_episode :140, score : 9.7, n_buffer : 1342, eps : 7.3%
# n_episode :160, score : 9.7, n_buffer : 1536, eps : 7.2%
# n_episode :180, score : 9.8, n_buffer : 1731, eps : 7.1%
# n_episode :200, score : 9.7, n_buffer : 1924, eps : 7.0%
# C:\Users\howar\Desktop\Opengym_ReinforcementLearning\DQN.py:52: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:256.)
#   return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
# n_episode :220, score : 11.8, n_buffer : 2161, eps : 6.9%
# n_episode :240, score : 22.1, n_buffer : 2602, eps : 6.8%
# n_episode :260, score : 49.7, n_buffer : 3596, eps : 6.7%
# n_episode :280, score : 69.0, n_buffer : 4976, eps : 6.6%
# n_episode :300, score : 63.6, n_buffer : 6249, eps : 6.5%
# n_episode :320, score : 217.1, n_buffer : 10591, eps : 6.4%
# n_episode :340, score : 357.4, n_buffer : 17738, eps : 6.3%
# n_episode :360, score : 314.0, n_buffer : 24018, eps : 6.2%
# n_episode :380, score : 308.6, n_buffer : 30189, eps : 6.1%
# n_episode :400, score : 327.1, n_buffer : 36732, eps : 6.0%
# n_episode :420, score : 315.9, n_buffer : 43051, eps : 5.9%
# n_episode :440, score : 347.4, n_buffer : 50000, eps : 5.8%
# n_episode :460, score : 302.9, n_buffer : 50000, eps : 5.7%
# n_episode :480, score : 263.5, n_buffer : 50000, eps : 5.6%
# n_episode :500, score : 329.1, n_buffer : 50000, eps : 5.5%
# n_episode :520, score : 381.1, n_buffer : 50000, eps : 5.4%
# n_episode :540, score : 330.9, n_buffer : 50000, eps : 5.3%
# n_episode :560, score : 375.4, n_buffer : 50000, eps : 5.2%
# n_episode :580, score : 299.0, n_buffer : 50000, eps : 5.1%
# n_episode :600, score : 392.7, n_buffer : 50000, eps : 5.0%
# Graph saved to ./results/dqn_performance.png

# n_episode :520, score : 381.1, n_buffer : 50000, eps : 5.4%
# n_episode :540, score : 330.9, n_buffer : 50000, eps : 5.3%
# n_episode :560, score : 375.4, n_buffer : 50000, eps : 5.2%
# n_episode :580, score : 299.0, n_buffer : 50000, eps : 5.1%
# n_episode :600, score : 392.7, n_buffer : 50000, eps : 5.0%
# Graph saved to ./results/dqn_performance.png

# n_episode :540, score : 330.9, n_buffer : 50000, eps : 5.3%
# n_episode :560, score : 375.4, n_buffer : 50000, eps : 5.2%
# n_episode :580, score : 299.0, n_buffer : 50000, eps : 5.1%
# n_episode :600, score : 392.7, n_buffer : 50000, eps : 5.0%
# Graph saved to ./results/dqn_performance.png

# n_episode :560, score : 375.4, n_buffer : 50000, eps : 5.2%
# n_episode :580, score : 299.0, n_buffer : 50000, eps : 5.1%
# n_episode :600, score : 392.7, n_buffer : 50000, eps : 5.0%
# Graph saved to ./results/dqn_performance.png

# n_episode :580, score : 299.0, n_buffer : 50000, eps : 5.1%
# n_episode :600, score : 392.7, n_buffer : 50000, eps : 5.0%
# Graph saved to ./results/dqn_performance.png

# n_episode :600, score : 392.7, n_buffer : 50000, eps : 5.0%