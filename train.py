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


import roboschool, gym
import numpy as np, time
from tqdm import tqdm

from agent.ddpg import ddpgAgent

NUM_EPISODES_ = 500
BATCH_SIZE = 64

def main():
	# Create Environments
	model_ = 'RoboschoolAnt-v1'
	env = gym.make(model_)
	
	# Create Agent model
	agent = ddpgAgent(env)


	# Initialize Environments
	steps = 500#env._max_episode_steps # steps per episode
	num_act_ = env.action_space.shape[0]
	num_obs_ = env.observation_space.shape[0]
	print("============ENVIRONMENT===============")
	print("num_of_action_spaces : %d"%num_act_)
	print("num_of_observation_spaces: %d"%num_obs_)	
	print("max_steps_per_episode: %d"%steps)
	print("======================================")


	act_range = env.action_space.high
	for epi in range(NUM_EPISODES_):
		print("=========EPISODE # %d =========="%epi)
		obs = env.reset()
		actions, states, rewards, dones, new_states = [],[],[],[],[]

		epi_reward = 0
		for t in tqdm(range(steps)):
			# environment rendering on Graphics
			env.render()
			
			# Make action from the current policy
			action_ = agent.make_action(obs)#env.action_space.sample()#
			action = np.clip(action_ + agent.noise.generate(t), -act_range, act_range)

			# do step on gym at t-time
			new_obs, reward, done, info = env.step(action) 
			
			# store the results to buffer
			agent.memorize(obs, action, reward, done, new_obs)
			# sample from buffer
			states, actions, rewards, dones, new_states, _ = agent.sample_batch(BATCH_SIZE)

			# get target q-value using target network
			q_vals = agent.critic.target_predict([new_states,agent.actor.target_predict(new_states)])

			# bellman iteration for target critic value
			critic_target = agent.critic.bellman(rewards, q_vals, dones)

			# train(or update) the actor & critic and target networks
			agent.update_networks(states, actions, critic_target)

			# grace finish and go to t+1 time
			obs = new_obs
			epi_reward = epi_reward + reward

			# check if the episode is finished
			if done or (t == steps-1):
				print("Episode#%d, steps:%d, rewards:%f"%(epi,t,epi_reward))
				break;
				
	env.close()

if __name__ == '__main__':
	main()