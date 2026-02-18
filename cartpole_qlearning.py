# model free learning 
# functions: select_action(), stimulate(), get_exploration_rate, get_learning_rate, state_to_bucket

# https://blog.techbridge.cc/2017/11/04/openai-gym-intro-and-q-learning/
import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os
import time

# Patch for modern numpy versions
np.bool8 = bool

## Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v1')

## Defining the environment related constants
NUM_BUCKETS = (1, 1, 6, 3) # (x, x', theta, theta'z ) (position, velocity, pole angle, pole angular velocity)
# why buckets are needed? The table cannot have an infinite number of rows for the the raw state from gym, so we round them into integer index (buckets)
#  total 1 *1 *6*3 = 18 scenarios
NUM_ACTIONS = env.action_space.n # (left, right)
print(NUM_ACTIONS)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5] # velocity limit
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)] # radian limit
print(STATE_BOUNDS)
# [
#   (array(-4.8, dtype=float32), array(4.8, dtype=float32)),  # Cart Position
#   [-0.5, 0.5],                                               # Cart Velocity (Your override)
#   (array(-0.41887903, dtype=float32), array(0.41887903, dtype=float32)), # Pole Angle
#   [-0.8726646259971648, 0.8726646259971648]                  # Angular Velocity (Your override)
# ]

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
print(q_table)
# q_tabel: (1,1,6,3,2)
# state coord: [0, 0, 3, 1].
# tabular q table: 18 states, 2 acitons -> total 36 q values
# q_table[state]: Dimensions Provided 4 of 5, Return an array of all actions  -> e.g: q_table[0][0][3][1] result in [-0.12, 0.45]# A 1D array of size 2, q-value for moving left and right
# q_table[state]: Returns an array of all actions (e.g., [2.5, 4.1]). This is used with np.amax() to find the "best possible future" ($V(n)$)


## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

#Term,In your Code,Duration
# Step, One iteration of the for t in range(MAX_T) loop.,A fraction of a second.
# Episode / Game,One full execution of the for episode in range(NUM_EPISODES) loop.,Between 1 and 500 steps.
# Training,The entire simulate() function.,"Up to 1,000 episodes."

## Defining the simulation related constants
NUM_EPISODES = 1000 # Global Limit,	Total attempts allowed for training.
MAX_T = 500       # Increased for CartPole-v1, An Episode is a full sequence of steps, Episode Limit, The point where a single game is forced to stop.
SOLVED_T = 499    # CartPole-v1 max score, Target Score, The minimum score needed to count as a "win".
STREAK_TO_END = 120 # Finish Line, 	Number of wins in a row needed to stop training early.
# 120 consecutive games

# select_action(): exploration - exploitation tradeoff
# random.random(): if current exploration rate > random decimal -> enter exploration mode - random sample in action space (left,right)
# explore rate - epsilon and learning rate - alpha

def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action


# q-table update formula: 
# reward + (discount f* best_q) : reward+ the best score you expect to get from the new state you just landed in
#  target: what we found out - target = reward + (discount factor* np.amax(q_table[state]))
#  old value: q_table[state_0 + (action,)] , what we already known in a single number return
# q_table[state_0 + (action,)]: By adding the (action,) tuple to the end, you are picking one specific "slot" to update with the learning formula.
# q_table[state + (action,)]: Dimensions Provided 5 of 5, so it returns A Scalar (One specific action value, comparing to q_table[state]) 
# Temporal error difference error: td_error = target - old_value
# updating : q_table[state_0 + (action, )] += learning_rate * td_error

# get_explore_rate(t), get_learning_rate(t)
# reduce randomness overtime:
# exploration vs. exploitation trade-off : Move randomly vs use value stored in q_table
# if random.random() < explore_rate (0.01),the agent enters Exploration mode.
# current episode:t 
# t+1 to prevent division by zero
# /25 scaling factor 
#  base 10 logarithm to create a curve that increase fast at first and then grow slowly 
#  finally 1 - log (t+1/25) so  the values drop fast at first and then level off slowly, allowing the agent to "fine-tune" for a long time.
# min (1, ) : ceiling of probabilty of random action never exceed 100%
# max (0.01,):  Always take random 1 % action of the time, to prevent stuck in a local loop
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
        # linear scaling to determine the slice it belongs if the value is within normal value bounds
        # e.g : radian from -0.87 to +0.87, Number of Buckets: 6 (Indices will be 0, 1, 2, 3, 4, 5).
        # scaling: for every 1 radian of movement, the index moves roughly how many buckets
        # bound_width: 0.87 - (-0.87) = 1.74
        # offset : (5 * -0.87) / 1.74 = -2.5 => the zero point is actually located $2.5$ steps away from the "array zero".
        # -0.87: This is the STATE_BOUNDS[i][0] (the minimum physical value allowed).
        # Think of a ruler that starts at -0.87 and ends at 0.87. To use it with a computer, you must slide it so it starts at 0
        # state_i: Raw Physics, Continuous Decimal (e.g., -0.43)
        # scaling * state_i: "Table Units"	Scaled Decimal (e.g., -1.23)
        # ... - offset	Positioning	Shifted to start at zero (e.g., 1.27)
        # int(round(...))	Discretization	Final Index (e.g., 1)
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def simulate():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99 
    
    num_streaks = 0
    
    # 1. Create a list to track our scores for the visualization graph
    reward_history = [] 

    for episode in range(NUM_EPISODES):
        #outer loop: total games the agent allow to play : 1000
        # Modern gym reset() returns a tuple
        obv, info = env.reset()
        state_0 = state_to_bucket(obv)
        #eg.: ( 0,0,3,1) + (1,)
        for t in range(MAX_T):
        #innter loop: This is the Timeline of a single game. It can last anywhere from 1 step (if the agent is bad) to 500 step
            # Select and execute action
            action = select_action(state_0, explore_rate)
            
            # Modern gym step() returns 5 values
            obv, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Observe the result
            state = state_to_bucket(obv)
            
            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            
            state_0 = state
            
            if done:
                # Track the score (timesteps survived) for this episode
                reward_history.append(t + 1)
                
                # Print clean progression every 50 episodes
                if (episode + 1) % 50 == 0:
                    print(f"Episode: {episode + 1:4d} | Score: {t + 1:3d} | Explore Rate: {explore_rate:.3f} | Streaks: {num_streaks}")

                if t >= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

        if num_streaks > STREAK_TO_END:
            print(f"\nEnvironment solved accurately after {episode} episodes!")
            break


        # Update parameters for the next episode
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

    env.close()

    # 2. Visualize the training performance when finished
    print("\nTraining finished. Generating performance graph...")
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Score (Timesteps Survived)')
    plt.title('CartPole-v1 Q-Learning Performance')
    plt.grid(True)
    plt.show()






if __name__ == "__main__":
    simulate()

# def simulate():
#     learning_rate = get_learning_rate(0)
#     explore_rate = get_explore_rate(0)
#     discount_factor = 0.99 
    
#     num_streaks = 0
#     reward_history = [] 

#     print("Starting silent training... Please wait.")
    
#     for episode in range(NUM_EPISODES):
#         obv, info = env.reset()
#         state_0 = state_to_bucket(obv)
        
#         for t in range(MAX_T):
#             action = select_action(state_0, explore_rate)
#             obv, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
            
#             state = state_to_bucket(obv)
#             best_q = np.amax(q_table[state])
#             q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            
#             state_0 = state
            
#             if done:
#                 reward_history.append(t + 1)
                
#                 if (episode + 1) % 50 == 0:
#                     print(f"Episode: {episode + 1:4d} | Score: {t + 1:3d} | Explore Rate: {explore_rate:.3f} | Streaks: {num_streaks}")

#                 if t >= SOLVED_T:
#                     num_streaks += 1
#                 else:
#                     num_streaks = 0
#                 break

#         if num_streaks > STREAK_TO_END:
#             print(f"\nEnvironment solved accurately after {episode} episodes!")
#             break

#         explore_rate = get_explore_rate(episode)
#         learning_rate = get_learning_rate(episode)

#     env.close() # Close the training environment

#     # --- NEW FEATURE 1: Save the Graph ---
#     print("\nTraining finished. Saving performance graph...")
    
#     # Create a directory named 'results' if it doesn't already exist
#     os.makedirs('./results', exist_ok=True) 
    
#     plt.plot(reward_history)
#     plt.xlabel('Episode')
#     plt.ylabel('Score (Timesteps Survived)')
#     plt.title('CartPole-v1 Q-Learning Performance')
#     plt.grid(True)
    
#     # Save the graph to the folder, then show it
#     save_path = './results/training_performance.png'
#     plt.savefig(save_path)
#     print(f"Graph successfully saved to: {save_path}")
    
#     # Note: plt.show() pauses the script until you close the graph window!
#     plt.show() 


#     # --- NEW FEATURE 2: Render & Visualize the Trained Agent ---
#     print("\nNow rendering the fully trained agent...")
    
#     # Open a NEW environment with the human popup window turned on
#     render_env = gym.make('CartPole-v1', render_mode="human")
#     obv, info = render_env.reset()
#     state_0 = state_to_bucket(obv)
    
#     for t in range(MAX_T):
#         render_env.render()
#         time.sleep(0.05) # Pause so human eyes can watch the animation
        
#         # We set explore_rate to 0.0 because training is over! 
#         # The agent must only use what it learned.
#         action = select_action(state_0, explore_rate=0.0) 
        
#         obv, reward, terminated, truncated, info = render_env.step(action)
#         state_0 = state_to_bucket(obv)
        
#         if terminated or truncated:
#             print(f"Visualization finished. The trained agent survived for {t + 1} timesteps!")
#             time.sleep(2) # Leave the window open for 2 seconds before closing
#             break
            
#     render_env.close()


# def select_action(state, explore_rate):
#     if random.random() < explore_rate:
#         action = env.action_space.sample()
#     else:
#         action = np.argmax(q_table[state])
#     return action

# def get_explore_rate(t):
#     return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

# def get_learning_rate(t):
#     return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

# def state_to_bucket(state):
#     bucket_indice = []
#     for i in range(len(state)):
#         if state[i] <= STATE_BOUNDS[i][0]:
#             bucket_index = 0
#         elif state[i] >= STATE_BOUNDS[i][1]:
#             bucket_index = NUM_BUCKETS[i] - 1
#         else:
#             bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
#             offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
#             scaling = (NUM_BUCKETS[i]-1)/bound_width
#             bucket_index = int(round(scaling*state[i] - offset))
#         bucket_indice.append(bucket_index)
#     return tuple(bucket_indice)

# if __name__ == "__main__":
#     simulate()



# Episode:   50 | Score:  15 | Explore Rate: 0.708 | Streaks: 0
# Episode:  100 | Score:  14 | Explore Rate: 0.402 | Streaks: 0
# Episode:  150 | Score:  11 | Explore Rate: 0.225 | Streaks: 0
# Episode:  200 | Score: 500 | Explore Rate: 0.099 | Streaks: 6
# Episode:  250 | Score: 500 | Explore Rate: 0.010 | Streaks: 37
# Episode:  300 | Score: 500 | Explore Rate: 0.010 | Streaks: 87

# Environment solved accurately after 332 episodes! 
# early exit logic: if num_streaks > STREAK_TO_END:.agent survived for 500 steps in 120 consecutive games. (87+32 =119)