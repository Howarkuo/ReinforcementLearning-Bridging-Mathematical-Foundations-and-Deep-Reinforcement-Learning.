

import gym
import numpy as np
import time
np.bool8 = bool
game = gym.make('CartPole-v1', render_mode="human")
## Check dimension of spaces ##
print(env.action_space)
#Discrete(2) : 0 left, 1 right 
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
# Box([-4.8000002e+00 -3.4028235e+38 (infinity) -4.1887903e-01 -3.4028235e+38 (infinity)], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
# cart position, Cart Velocity , Pole Anglelimits, Pole Angular Velocity 
## Check range of spaces ##
"""
print(env.action_space.high)-
You'll get error if you run this, because 'Discrete' object has no attribute 'high'
"""
print(env.observation_space.high)
print(env.observation_space.low)
# 1. Update to v1 and specify "human" so the popup window appears
  
for i_episode in range(3): #how many episodes you want to run
	observation, info = game.reset() #reset() returns initial observation
	# reset set 4 variables in tiny random number between -0.05 and 0.05.
  
	for steps in range(100):
		# take 100 actions
		game.render()
		time.sleep(0.05)
		print(observation)
		# action =np.random.randint(0,4)
		action = game.action_space.sample()
		#"Give me one random, legal button press" (which will strictly be a 0 or a 1).
		next_state, reward, terminated, truncated, info = game.step(action)
		# if done:
		# 	print("Episode finished after {} timesteps".format(steps+1))
		# 	break
		# 5. Combine terminated and truncated to check if the episode is over
		if terminated or truncated:
			print("Episode finished after {} timesteps".format(steps+1))
			time.sleep(1)
			break
		observation = next_state
		
# output: observation state: cart pos, cart velocity, pole angle, pole angular velocity 
# terminate : 
# The Angle: The pole leans more than 12 degrees from perfectly vertical. (It fell over).
# The Position: The cart moves more than 2.4 units from the center. (It drove off the edge of the screen).


        
# game.close()



# The Position: The cart moves more than 2.4 units from the center. (It drove off the edge of the screen).
# See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
# [ 0.01103121 -0.00484239  0.00030742  0.03298664]
# [ 0.01093437 -0.19996874  0.00096715  0.32576653]
# [ 0.00693499 -0.00486058  0.00748248  0.03338877]
# [ 0.00683778  0.19015327  0.00815026 -0.256924  ]
# [ 0.01064084  0.3851579   0.00301178 -0.5470251 ]
# [ 0.018344    0.18999378 -0.00792872 -0.25339475]
# [ 0.02214388  0.38522804 -0.01299662 -0.54856795]
# [ 0.02984844  0.58053017 -0.02396798 -0.84531724]
# [ 0.04145904  0.77597076 -0.04087432 -1.14544   ]
# [ 0.05697846  0.5814058  -0.06378312 -0.86585   ]
# [ 0.06860658  0.77733517 -0.08110012 -1.1778859 ]
# [ 0.08415328  0.5833547  -0.10465784 -0.9116889 ]
# [ 0.09582037  0.7797252  -0.12289162 -1.2353461 ]
# [ 0.11141488  0.58637804 -0.14759853 -0.983552  ]
# [ 0.12314244  0.783136   -0.16726959 -1.3187165 ]
# [ 0.13880515  0.97993076 -0.19364391 -1.6587368 ]
# Episode finished after 16 timesteps
# [-0.0054155  -0.0448286   0.01750582 -0.02619064]
# [-0.00631207  0.15003799  0.01698201 -0.3132993 ]
# [-0.00331131  0.34491396  0.01071603 -0.6005786 ]
# [ 0.00358697  0.5398844  -0.00129555 -0.889867  ]
# [ 0.01438465  0.34478    -0.01909289 -0.5975916 ]
# [ 0.02128025  0.14993036 -0.03104472 -0.31098336]
# [ 0.02427886  0.34548053 -0.03726438 -0.613293  ]
# [ 0.03118847  0.5411029  -0.04953025 -0.91747594]
# [ 0.04201053  0.3466843  -0.06787977 -0.64076173]
# [ 0.04894422  0.5426836  -0.080695   -0.9540249 ]
# [ 0.05979789  0.738793   -0.0997755  -1.2709304 ]
# [ 0.07457375  0.93503696 -0.1251941  -1.5931177 ]
# [ 0.09327449  1.1314025  -0.15705647 -1.9220716 ]
# [ 0.11590254  1.3278246  -0.19549789 -2.2590635 ]
# Episode finished after 14 timesteps