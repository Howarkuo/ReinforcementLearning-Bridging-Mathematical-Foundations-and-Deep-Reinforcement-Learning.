#Practice: grid world city

# finding the sweet spot of the street parking 
# using bellman equation


# to evaluate the current pricing scheme,
# we need to update the value of a state s by looking at all possible acction a and policy pi might take 
# and then summing the discounted rewards and future values across all possibel next state s'
import numpy as np
import tools
import grader


#### 0. set up - value-function array, action space array, the policy array (2D), transitions array

# Environment parameter
# s: num space {0,1,2,3}, A: price point  {0,1,2 }

# template: tools.ParkingWorld 
# Encapsulation: The internal logic (how a price of $2 affects a garage with 1 car) is hidden inside the class
# State (Attributes): When you call env = tools.ParkingWorld(num_spaces, num_prices), you are instantiating an object. This specific object now "remembers" that it has 3 spaces and 3 price points.
# method: env.transitions(s, a)- f I'm in state s and I set price a, what could happen?"
num_spaces = 3
num_prices = 3
env = tools.ParkingWorld(num_spaces, num_prices)
V = np.zeros(num_spaces + 1)
pi = np.ones((num_spaces + 1, num_prices)) / num_prices

# value function array: 1 D array
# V: array([0., 0., 0., 0.])

# pi: 2 D array: probability of taking action j in state i 
# array([[0.33333333, 0.33333333, 0.33333333],
    #    [0.33333333, 0.33333333, 0.33333333],
    #    [0.33333333, 0.33333333, 0.33333333],
    #    [0.33333333, 0.33333333, 0.33333333]])

# 0.1 global policy-- 
# stochastic policy : a|s in probability distribution to map 
pi[0] = [0.75, 0.11, 0.14]


# enumerate(): built in to build loop with index
for s, pi_s in enumerate(pi):
    for a, p in enumerate(pi_s):
        print pi[0] = [0.75, 0.11, 0.14]

for s, pi_s in enumerate(pi):
    for a, p in enumerate(pi_s):
        print(f'pi(A={a}|S={s}) = {p.round(2)}    ', end='')
    print()

# the policy
# pi(A=0|S=0) = 0.75    pi(A=1|S=0) = 0.11    pi(A=2|S=0) = 0.14   
# When the street is completely empty (S=0), the City Council's current strategy is to set the price to Cheap (A=0) 75% of the time, Normal (A=1) 11% of the time, and Expensive (A=2) 14% of the time.
 
# pi(A=0|S=1) = 0.33    pi(A=1|S=1) = 0.33    pi(A=2|S=1) = 0.33    
# pi(A=0|S=2) = 0.33    pi(A=1|S=2) = 0.33    pi(A=2|S=2) = 0.33    
# pi(A=0|S=3) = 0.33    pi(A=1|S=3) = 0.33    pi(A=2|S=3) = 0.33  
#"If I am in this state, what action should I choose?"



#env.S : [0,1,2,3]
# env.A: [0,1,2]

# 0.2: The transition level
state = 3
action = 1
transitions = env.transitions(state, action)
#transitions
# defined by tools.py
# always keep the street full but always leave exactly one spot open
# If the street is completely full (state=3), and I decide to charge the Normal price (action=1), what will happen?"
#        reward  , probability 
# state: 3-> 0 , 1, ,2, 3, ,4 
# array([[1.        , 0.12390437],
#        [2.        , 0.15133714],
#        [3.        , 0.1848436 ],
#        [2.        , 0.53991488]])

for sp, (r, p) in enumerate(transitions):
    print(f'p(S\'={sp}, R={r} | S={state}, A={action}) = {p.round(2)}')

# p(S'=0, R=1.0 | S=3, A=1) = 0.12
# p(S'=1, R=2.0 | S=3, A=1) = 0.15
# p(S'=2, R=3.0 | S=3, A=1) = 0.18
# p(S'=3, R=2.0 | S=3, A=1) = 0.54

# immediate reward = $$(0.12 \times 1) + (0.15 \times 2) + (0.18 \times 3) + (0.54 \times 2) = \text{Expected Immediate Reward}$$

### 1. policy evaluation
# $$\large v(s) \leftarrow \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$

def evaluate_policy(env, V, pi, gamma, theta):
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
    # the max here has nothing to do with choosing actions. It is simply a mathematical bookkeeping trick to figure out when to stop the while loop. 
    #The Goal: We want to keep updating our V array until the numbers stop changing.
    # Assume my policy is fixed. Average the outcomes ($\sum$) based on my policy's probabilities until delta is tiny."       
    return V
#delta: tracks the latest change to the state's value



# delta 

# update V[s] -> v_new
# -----------
# Graded Cell
# -----------
def bellman_update(env, V, pi, s, gamma):
    """Mutate ``V`` according to the Bellman update equation."""
    # YOUR CODE HERE
    v_new = 0 
    for a, prob_a in enumerate(pi[s]):
        transitions = env.transitions(s, a)
    #: It took a specific policy array (pi), evaluated it, and averaged the outcomes based on how often the policy chose each action ($\sum_a \pi(a | s)$). It looped over the states dozens of times until the values completely converged.

        for next_state, (reward, prob_transition) in enumerate(transitions):
            v_new += prob_a * prob_transition * (reward + gamma * V[next_state])
    # Update the value function in-place
    V[s] = v_new





# set up test environment
num_spaces = 10
# state: from empty to 10
num_prices = 4
env = tools.ParkingWorld(num_spaces, num_prices)

# build test policy
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1

gamma = 0.9
theta = 0.1

V = np.zeros(num_spaces + 1)
V = evaluate_policy(env, V, city_policy, gamma, theta)

print(V)

# [80.04173399 81.65532303 83.37394007 85.12975566 86.87174913 88.55589131
#  90.14020422 91.58180605 92.81929841 93.78915889 87.77792991]

# set up test environment
num_spaces = 10
num_prices = 4
env = tools.ParkingWorld(num_spaces, num_prices)

# build test policy
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1

gamma = 0.9
theta = 0.1

V = np.zeros(num_spaces + 1)
V = evaluate_policy(env, V, city_policy, gamma, theta)

# test the value function
answer = [80.04, 81.65, 83.37, 85.12, 86.87, 88.55, 90.14, 91.58, 92.81, 93.78, 87.77]

# make sure the value function is within 2 decimal places of the correct answer
assert grader.near(V, answer, 1e-2)

tools.plot(V, city_policy)

# observation: Observe that the value function qualitatively resembles the city council's preferences — 
# it monotonically increases as more parking is used, 
# until there is no parking left, 
# in which case the value is lower.
#  Because of the relatively simple reward function 
# (more reward is accrued when many but not all parking spots are taken and less reward is accrued when few or all parking spots are taken) 
# and the highly stochastic dynamics function
#  (each state has positive probability of being reached each time step) 
# the value functions of most policies will qualitatively resemble this graph.


# move on: 
# good policies are policies that spend more time in desirable states and less time in undesirable states. 
# ] such a steady state distribution is achieved by setting the price to be low in low occupancy states 
# (so that the occupancy will increase)
#  and setting the price high when occupancy is high 
# (so that full occupancy will be avoided).

# now we established a policy, let's try another new

###2.  Policy Iteration- summation of probability of  transitioning to the next state $s'$ and receiving reward $r$ given the current state $s$ and action $a$ times (the immediate reward+discount factor*value of the next state under the current policy )
# Policy iteration works by alternating between evaluating the existing policy and making the policy greedy with respect to the existing value function


# V: $V_\pi(s')$ is the value of the next state under the current policy 
# [80.04173399 81.65532303 83.37394007 85.12975566 86.87174913 88.55589131
#  90.14020422 91.58180605 92.81929841 93.78915889 87.77792991]

# pi: 2 D array: probability of taking action j in state i 
# array([[0.33333333, 0.33333333, 0.33333333],
    #    [0.33333333, 0.33333333, 0.33333333],
    #    [0.33333333, 0.33333333, 0.33333333],
    #    [0.33333333, 0.33333333, 0.33333333]])


def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        # make a copy of the current action probability for a state
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        
        if not np.array_equal(pi[s], old):
            policy_stable = False
            
    return pi, policy_stable

# value: the expectation of the future state as reward
# state: row, emv/a action - numb of columns
# two-step dance of Policy Iteration (Evaluate an average policy $\rightarrow$ Improve it $\rightarrow$ Evaluate the new average policy)
# 
def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
        
    return V, pi

def q_greedify_policy(env, V, pi, s, gamma):
    """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
    q_value = np.zeros(len(env.A))

    for a in env.A:
        transitions = env.transitions(s, a)
        for next_state, (reward, prob_transition) in enumerate(transitions):
            # No 'done' flag needed here based on how ParkingWorld works
            q_value[a] += prob_transition * (reward + gamma * V[next_state])

    best_action = np.argmax(q_value)
# "I don't care which action caused this high score right now, I just want to write the highest possible number on the scoreboard for this state so my future estimates are better."
    pi[s] = np.zeros(len(env.A))
    pi[s][best_action] = 1.0

# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

gamma = 0.9
theta = 0.1
env = tools.ParkingWorld(num_spaces=6, num_prices=4)

V = np.array([7, 6, 5, 4, 3, 2, 1])
pi = np.ones((7, 4)) / 4

new_pi, stable = improve_policy(env, V, pi, gamma)

# expect first call to greedify policy
expected_pi = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])
assert np.all(new_pi == expected_pi)
assert stable == False

# the value function has not changed, so the greedy policy should not change
new_pi, stable = improve_policy(env, V, new_pi, gamma)

assert np.all(new_pi == expected_pi)
assert stable == True


# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.
# policy_iteration: evaluation -> improvement -> Evaluation -> Improvement


gamma = 0.9
theta = 0.1
env = tools.ParkingWorld(num_spaces=10, num_prices=4)

V, pi = policy_iteration(env, gamma, theta)

V_answer = [81.60, 83.28, 85.03, 86.79, 88.51, 90.16, 91.70, 93.08, 94.25, 95.25, 89.45]
pi_answer = [
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]

# make sure value function is within 2 decimal places of answer
assert grader.near(V, V_answer, 1e-2)
# make sure policy is exactly correct
assert np.all(pi == pi_answer)



env = tools.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = policy_iteration(env, gamma, theta)

# State    Value    Action
# 0    81.6    0
# 1    83.3    0
# 2    85.0    0
# 3    86.8    0
# 4    88.5    0
# 5    90.2    0
# 6    91.7    0
# 7    93.1    0
# 8    94.3    0
# 9    95.3    3
# 10    89.5    3

#### optimal policy translation: 
# "When there are 9 cars parked, set the price to the most expensive tier to stop the 10th car from parking!"


## Section 3: Value Iteration
## $$\large v(s) \leftarrow \max_a \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$

# the maximization of the policy will be the short cut to find the optimal policy, which 
# is different from policy iteration, which adds up the pi policy distribution



def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi

def bellman_optimality_update(env, V, s, gamma):
    """Mutate ``V`` according to the Bellman optimality update equation."""
    
    # 1. Create an array to hold the expected return (Q-value) for every action
    q_values = np.zeros(len(env.A))
    
    # 2. Loop through every possible action
    for a in env.A:
        transitions = env.transitions(s, a)
        
        # 3. Calculate the expected return for this specific action (the sum part)
        for next_state, (reward, prob_transition) in enumerate(transitions):
            q_values[a] += prob_transition * (reward + gamma * V[next_state])
            
    # 4. The Magic Step: Instead of averaging based on a policy, 
    # we just grab the absolute highest score!
    V[s] = np.max(q_values)

#Difference in q_greedy_policy() and bellman_optimality_update()
# what they do with that list of scores at the very end.
# update policy pi[s] v.s. update the value function V[s]


# --------------
# Debugging Cell
# --------------
# Feel free to make any changes to this cell to debug your code

gamma = 0.9
env = tools.ParkingWorld(num_spaces=6, num_prices=4)

V = np.array([7, 6, 5, 4, 3, 2, 1])

# only state 0 updated
bellman_optimality_update(env, V, 0, gamma)
assert list(V) == [5, 6, 5, 4, 3, 2, 1]

# only state 2 updated
bellman_optimality_update(env, V, 2, gamma)
assert list(V) == [5, 6, 7, 4, 3, 2, 1]


# You can check your value function (rounded to one decimal place) and policy against the answer below:
# State    Value    Action
# 0    81.6    0
# 1    83.3    0
# 2    85.0    0
# 3    86.8    0
# 4    88.5    0
# 5    90.2    0
# 6    91.7    0
# 7    93.1    0
# 8    94.3    0
# 9    95.3    3
# # 10    89.5    3


# These are the exact same graphs you generated at the end of Policy Iteration. This highlights one of the most beautiful mathematical guarantees in Reinforcement Learning: Whether you take the slow, steady path (Policy Iteration) or the aggressive shortcut (Value Iteration), you are guaranteed to arrive at the exact same optimal solution.

# Value Iteration is just Policy Iteration where you get impatient.

# def value_iteration2(env, gamma, theta):
    # V = np.zeros(len(env.S))
    # pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    # while True:
    #     delta = 0
    #     for s in env.S:
    #         v = V[s]
    #         q_greedify_policy(env, V, pi, s, gamma)
    #         bellman_update(env, V, pi, s, gamma)
    #         delta = max(delta, abs(v - V[s]))
    #     if delta < theta:
    #         break
    # return V, pi