'''
MIT License

Copyright (c) 2020 Junyoeb Baek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# Implementation of DDPG(Deep Deterministic Policy Gradient) 
# on WalkYTo-rl framwork


import roboschool
import gym

from agent import ddpgAgent

NUM_EPISODES_ = 1000

def main():
	# Create Environments
	model_ = 'RoboschoolAnt-v1'
	env = gym.make(model_)
	
	# Create Agent model
	agent = ddpgAgent(env)


	# Initialize Environments
	steps = env._max_episode_steps # steps per episode
	num_act_ = env.action_space.shape[0]
	num_obs_ = env.observation_space.shape[0]
	print("============ENVIRONMENT===============")
	print("num_of_action_spaces : %d"%num_act_)
	print("num_of_observation_spaces: %d"%num_obs_)	
	print("max_steps_per_episode: %d"%steps)
	print("======================================")

	for epi in range(NUM_EPISODES_):
		print("=========EPISODE # %d =========="%epi)
		observation = env.reset()
		epi_reward = 0
		for t in range(steps):
			# environment rendering on Graphics
			env.render()
			
			#your agent goes here
			action = env.action_space.sample()#agent.make_action()

			observation, reward, done, info = env.step(action) 
			
			# train the agent's brain
			agent.train()

			epi_reward = epi_reward + reward

			# check if the episode is finished
			if done or (t == steps-1):
				print("Episode#%d, steps:%d, rewards:%f"%(epi,t,epi_reward))
				break;
				
	env.close()

if __name__ == '__main__':
	main()