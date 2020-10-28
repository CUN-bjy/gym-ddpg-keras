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

import argparse

from agent.ddpg import ddpgAgent

NUM_EPISODES_ = 1000

def model_play(pretrained_):
	# Create Environments
	models = {	'cheetah':"RoboschoolHalfCheetah-v1",
				'ant':'RoboschoolAnt-v1',
				'pong':"RoboschoolPong-v1",
				'walker':"RoboschoolWalker2d-v1",
				'hopper':"RoboschoolHopper-v1",
				'humanoid':"RoboschoolHumanoid-v1",
				'humanoidflag':"RoboschoolHumanoidFlagrun-v1"}
	
	env = gym.make(models['ant'])
	
	# Create Agent model
	agent = ddpgAgent(env)

	if not pretrained_ == None:
		agent.load_weights(pretrained_)

	# Initialize Environments
	steps = env._max_episode_steps # steps per episode
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
			obs = env.reset()
			actions, states, rewards, dones, new_states = [],[],[],[],[]

			epi_reward = 0
			while True:
				# environment rendering on Graphics
				env.render()
				
				# Make action from the current policy
				action = agent.make_action(obs)#env.action_space.sample()#

				# do step on gym at t-time
				new_obs, reward, done, info = env.step(action) 

				# grace finish and go to t+1 time
				obs = new_obs
				epi_reward = epi_reward + reward

				if done: break


	except KeyboardInterrupt as e:
		print(e)
	finally:
		env.close()


argparser = argparse.ArgumentParser(
	description='Train DDPG Agent on the openai gym')

argparser.add_argument(
	'-w',	'--weights',help='path to pretrained weights')


if __name__ == '__main__':
	#################################
	#   Parse Configurations
	#################################

	args = argparser.parse_args()
	weights_path = args.weights
	
	model_play(pretrained_=weights_path)
