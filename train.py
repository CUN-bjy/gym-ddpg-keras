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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import argparse

from agent.ddpg import ddpgAgent

NUM_EPISODES_ = 20000

def model_train(pretrained_):
	# Create Environments
	models = {	'cartpole':"CartPole-v1",
				'pendulum':"RoboschoolInvertedPendulum-v1",
				'cheetah':"RoboschoolHalfCheetah-v1",
				'walker':"RoboschoolWalker2d-v1",
				'hopper':"RoboschoolHopper-v1"}
	
	env = gym.make(models['hopper'])
	
	try:
		# Ensure action bound is symmetric
		assert (np.all(env.action_space.high+env.action_space.low) == 0)
		is_discrete = False
		print('Continuous Action Space')
	except AttributeError:
		is_discrete = True
		print('Discrete Action Space')

	# Create Agent model
	agent = ddpgAgent(env, batch_size=128, w_per=False, is_discrete=is_discrete)

	if not pretrained_ == None:
		agent.load_weights(pretrained_)

	# Initialize Environments
	steps = 500#env._max_episode_steps # steps per episode
	num_act_ = env.action_space.n if is_discrete else env.action_space.shape[0]
	num_obs_ = env.observation_space.shape[0]
	print("============ENVIRONMENT===============")
	print("num_of_action_spaces : %d"%num_act_)
	print("num_of_observation_spaces: %d"%num_obs_)	
	print("max_steps_per_episode: %d"%steps)
	print("======================================")


	logger = dict(episode=[],reward=[],critic_loss=[])
	plt.ion()
	fig1 = plt.figure(1);	fig2 = plt.figure(2)
	ax1 = fig1.add_subplot(111)
	ax2 = fig2.add_subplot(111)


	try:
		act_range = (env.action_space.high - env.action_space.low) / 2 if not is_discrete else 1.
		rewards = []; critic_losses = []
		max_reward = 0
		for epi in range(NUM_EPISODES_):
			print("=========EPISODE # %d =========="%epi)
			obs = env.reset()

			epi_reward = 0
			for t in tqdm(range(steps)):
				plt.pause(0.01)
				# environment rendering on Graphics
				env.render()
				
				# Make action from the current policy
				a = agent.make_action(obs, t)#env.action_space.sample()#
				action = np.argmax(a) if is_discrete else a

				# do step on gym at t-time
				new_obs, reward, done, info = env.step(action) 

				# store the results to buffer	
				agent.memorize(obs, a, reward, done, new_obs)

				# grace finish and go to t+1 time
				obs = new_obs
				epi_reward = epi_reward + reward

				agent.replay(1)

				# check if the episode is finished
				if done or (t == steps-1):
					print("Episode#%d, steps:%d, rewards:%f"%(epi,t,epi_reward))
					# agent.replay(1)

					# save weights at the new records performance
					if epi_reward >= max_reward:
						max_reward = epi_reward
						dir_path = "%s/weights"%os.getcwd()
						if not os.path.isdir(dir_path):
							os.mkdir(dir_path)
						path = dir_path+'/'+'gym_ddpg_'
						agent.save_weights(path + 'ep%d_%f'%(epi,max_reward))


					# save reward logs
					ax1.cla(); ax2.cla();
					logger['episode'] = range(epi+1)
					logger['reward'].append(epi_reward)
					logger['critic_loss'].append(agent.critic.critic_loss)

					df = pd.DataFrame(logger)
					sns.lineplot(ax=ax1,x='episode',y='reward', data=df)
					sns.lineplot(ax=ax2,x='episode',y='critic_loss', data=df)
					break;

	except KeyboardInterrupt as e: print(e)
	finally:
		# weight saver
		dir_path = "%s/weights"%os.getcwd()
		if not os.path.isdir(dir_path):
			os.mkdir(dir_path)
		path = dir_path+'/'+'gym_ddpg_'
		agent.save_weights(path +'lastest')
		env.close()

		# log saver
		import pickle
		pickle.dump(logger,open(path+'%s.log'%time.time(),'wb'))


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
	
	model_train(pretrained_=weights_path)
