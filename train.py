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
# on OpenAI gym framwork


import roboschool, gym
import numpy as np, time, os
from tqdm import tqdm

from agent.ddpg import ddpgAgent

NUM_EPISODES_ = 500

def main():
	# Create Environments
	models = {	'cheetah':"RoboschoolHalfCheetah-v1",
				'ant':'RoboschoolAnt-v1',
				'pong':"RoboschoolPong-v1",
				'walker':"RoboschoolWalker2d-v1",
				'hopper':"RoboschoolHopper-v1",
				'humanoid':"RoboschoolHumanoid-v1",
				'humanoidflag':"RoboschoolHumanoidFlagrun-v1"}
	
	env = gym.make(models['cheetah'])
	
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


	try:
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

				# grace finish and go to t+1 time
				obs = new_obs
				epi_reward = epi_reward + reward

				# check if the episode is finished
				if done or (t == steps-1):
					# Replay
					agent.replay(10)
					print("Episode#%d, steps:%d, rewards:%f"%(epi,t,epi_reward))
					if epi%10 == 1:
						dir_path = "%s/weights"%os.getcwd()
						if not os.path.isdir(dir_path):
							os.mkdir(dir_path)
						path = dir_path+'/'+'gym_ddpg_'
						agent.save_weights(path + 'ep%d'%epi)
					break;

	except KeyboardInterrupt as e:
		print(e)
	finally:
		dir_path = "%s/weights"%os.getcwd()
		if not os.path.isdir(dir_path):
			os.mkdir(dir_path)
		path = dir_path+'/'+'gym_ddpg_'
		agent.save_weights(path +'_temp_')
		env.close()

if __name__ == '__main__':
	main()