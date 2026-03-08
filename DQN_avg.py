# Example of adding a moving average (window of 50) to your reward_history
import gymnasium as gym
import os
import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import RecordVideo

# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

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

        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
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
        return self.fc3(x)
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0,1)
        else: 
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def save_video(q_model):
    print("\nRecording demonstration video...")
    video_path = './results/dqn_video_avg'
    
    # 1. Create the base environment
    # Note: Using "gym" vs "gymnasium" might require slightly different render_modes
    render_env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    # 2. Wrap it for recording (Removed 'disable_logger')
    render_env = RecordVideo(
        render_env, 
        video_folder=video_path, 
        episode_trigger=lambda x: True
    )
    
    s, _ = render_env.reset()
    done = False
    
    while not done:
        # Ensure we convert s to a tensor correctly
        obs_tensor = torch.from_numpy(s).float()
        a = q_model.sample_action(obs_tensor, epsilon=0)
        
        s, r, terminated, truncated, _ = render_env.step(a)
        done = terminated or truncated
    
    # 3. Explicitly close to ensure the video file is "flushed" and saved
    render_env.close()
    print(f"Video saved to {video_path}")
def plot_results(score_history):
    window = 50
    # Calculate moving average
    moving_avg = [np.mean(score_history[max(0, i-window):i+1]) for i in range(len(score_history))]
    
    plt.figure(figsize=(10,5))
    plt.plot(score_history, alpha=0.3, label='Raw Score', color='blue') 
    plt.plot(moving_avg, label=f'Moving Average ({window})', color='red', linewidth=2)
    plt.title("DQN Learning Curve - CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('./results/training_plot.png')
    plt.show()

def main():
    os.makedirs('./results', exist_ok=True)
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    score_history = []
    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(601):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
        s, _ = env.reset()
        done = False
        epi_score = 0

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0
            
            # Normalize reward for stability
            memory.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
            epi_score += r

        score_history.append(epi_score)
            
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            avg_score = score / print_interval
            print(f"n_episode :{n_epi}, avg score : {avg_score:.1f}, n_buffer : {memory.size()}, eps : {epsilon*100:.2f}%")
            
            if avg_score > 490:
                print("Environment Solved!")
                break
            score = 0.0

    env.close()
    plot_results(score_history)
    save_video(q)

if __name__ == "__main__":
    main()

# n_episode :20, avg score : 10.1, n_buffer : 202, eps : 7.90%
# n_episode :40, avg score : 9.5, n_buffer : 392, eps : 7.80%
# n_episode :60, avg score : 10.0, n_buffer : 592, eps : 7.70%
# n_episode :80, avg score : 10.2, n_buffer : 796, eps : 7.60%
# n_episode :100, avg score : 9.5, n_buffer : 986, eps : 7.50%
# n_episode :120, avg score : 9.9, n_buffer : 1184, eps : 7.40%
# n_episode :140, avg score : 9.4, n_buffer : 1373, eps : 7.30%
# n_episode :160, avg score : 10.2, n_buffer : 1578, eps : 7.20%
# n_episode :180, avg score : 9.9, n_buffer : 1776, eps : 7.10%
# n_episode :200, avg score : 9.9, n_buffer : 1975, eps : 7.00%
# n_episode :220, avg score : 9.7, n_buffer : 2168, eps : 6.90%
# n_episode :240, avg score : 10.1, n_buffer : 2370, eps : 6.80%
# n_episode :260, avg score : 11.1, n_buffer : 2591, eps : 6.70%
# n_episode :280, avg score : 12.2, n_buffer : 2836, eps : 6.60%
# n_episode :80, avg score : 10.2, n_buffer : 796, eps : 7.60%
# n_episode :100, avg score : 9.5, n_buffer : 986, eps : 7.50%
# n_episode :120, avg score : 9.9, n_buffer : 1184, eps : 7.40%
# n_episode :140, avg score : 9.4, n_buffer : 1373, eps : 7.30%
# n_episode :160, avg score : 10.2, n_buffer : 1578, eps : 7.20%
# n_episode :180, avg score : 9.9, n_buffer : 1776, eps : 7.10%
# n_episode :200, avg score : 9.9, n_buffer : 1975, eps : 7.00%
# n_episode :220, avg score : 9.7, n_buffer : 2168, eps : 6.90%
# n_episode :240, avg score : 10.1, n_buffer : 2370, eps : 6.80%
# n_episode :260, avg score : 11.1, n_buffer : 2591, eps : 6.70%
# n_episode :280, avg score : 12.2, n_buffer : 2836, eps : 6.60%
# n_episode :200, avg score : 9.9, n_buffer : 1975, eps : 7.00%
# n_episode :220, avg score : 9.7, n_buffer : 2168, eps : 6.90%
# n_episode :240, avg score : 10.1, n_buffer : 2370, eps : 6.80%
# n_episode :260, avg score : 11.1, n_buffer : 2591, eps : 6.70%
# n_episode :280, avg score : 12.2, n_buffer : 2836, eps : 6.60%
# n_episode :220, avg score : 9.7, n_buffer : 2168, eps : 6.90%
# n_episode :240, avg score : 10.1, n_buffer : 2370, eps : 6.80%
# n_episode :260, avg score : 11.1, n_buffer : 2591, eps : 6.70%
# n_episode :280, avg score : 12.2, n_buffer : 2836, eps : 6.60%
# n_episode :280, avg score : 12.2, n_buffer : 2836, eps : 6.60%
# n_episode :300, avg score : 13.9, n_buffer : 3114, eps : 6.50%
# n_episode :320, avg score : 33.2, n_buffer : 3779, eps : 6.40%
# n_episode :340, avg score : 118.0, n_buffer : 6138, eps : 6.30%
# n_episode :360, avg score : 318.6, n_buffer : 12509, eps : 6.20%
# n_episode :380, avg score : 356.7, n_buffer : 19643, eps : 6.10%
# n_episode :400, avg score : 407.8, n_buffer : 27799, eps : 6.00%
# n_episode :420, avg score : 407.6, n_buffer : 35952, eps : 5.90%
# n_episode :440, avg score : 295.2, n_buffer : 41857, eps : 5.80%
# n_episode :460, avg score : 255.1, n_buffer : 46959, eps : 5.70%
# n_episode :480, avg score : 200.2, n_buffer : 50000, eps : 5.60%
# n_episode :500, avg score : 255.1, n_buffer : 50000, eps : 5.50%
# n_episode :520, avg score : 197.8, n_buffer : 50000, eps : 5.40%
# n_episode :540, avg score : 182.2, n_buffer : 50000, eps : 5.30%
# n_episode :560, avg score : 251.4, n_buffer : 50000, eps : 5.20%
# n_episode :580, avg score : 235.6, n_buffer : 50000, eps : 5.10%
# n_episode :600, avg score : 266.6, n_buffer : 50000, eps : 5.00%