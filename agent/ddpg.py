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

import numpy as np

from .actor import ActorNet
from .critic import CriticNet

from utils.memory_buffer import MemoryBuffer
from utils.noise_process import OrnsteinUhlenbeckProcess

BATCH_SIZE = 5000

class ddpgAgent():
	"""Deep Deterministic Policy Gradient(DDPG) Agent
	"""
	def __init__(self, env_, buffer_size = 20000):
		# gym environments
		self.env = env_
		self.obs_dim = self.env.observation_space.shape[0]
		self.act_dim = self.env.action_space.shape[0]

		# initialize actor & critic and its targets
		self.actor = ActorNet(self.obs_dim, self.act_dim, self.env.action_space.high, lr_=10e-4,tau_=10e-3)
		self.critic = CriticNet(self.obs_dim, self.act_dim, lr_=10e-3,tau_=10e-3,discount_factor=0.99)

		# Experience Buffer
		self.buffer = MemoryBuffer(buffer_size, with_per=False)
		# OU-Noise-Process
		self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

	###################################################
	# Network Related
	###################################################
	def make_action(self, obs):
		""" predict next action from Actor's Policy
		"""
		return self.actor.predict(obs)[0]

	def update_networks(self, obs, acts, critic_target):
		""" Train actor & critic from sampled experience
		"""
		# update critic
		self.critic.train(obs, acts, critic_target)

		# get next action and Q-value Gradient
		n_actions = self.actor.network.predict(obs)
		q_grads = None#self.critic.Qgradient(obs, n_actions)

		# update actor
		self.actor.train(obs,self.critic.network,q_grads)

		# update target networks
		self.actor.target_update()
		self.critic.target_update()

	def replay(self, replay_num_):
		for _ in range(replay_num_):
			# sample from buffer
			states, actions, rewards, dones, new_states, _ = self.sample_batch(BATCH_SIZE)

			# get target q-value using target network
			q_vals = self.critic.target_predict([new_states,self.actor.target_predict(new_states)])

			# bellman iteration for target critic value
			critic_target = self.critic.bellman(rewards, q_vals, dones)

			# train(or update) the actor & critic and target networks
			self.update_networks(states, actions, critic_target)


	####################################################
	# Buffer Related
	####################################################

	def memorize(self,obs,act,reward,done,new_obs):
		"""store experience in the buffer
		"""
		self.buffer.memorize(obs,act,reward,done,new_obs)

	def sample_batch(self, batch_size):
		""" Sampling from the batch
		"""
		return self.buffer.sample_batch(batch_size)

	###################################################
	# Save & Load Networks
	###################################################
	def save_weights(self,path):
		""" Agent's Weights Saver
		"""
		self.actor.save_network(path)
		self.critic.save_network(path)

	def load_weights(self, actor_path, critic_path):
		""" Agent's Weights Loader
		"""
		self.actor.load_network(actor_path)
		self.critic.load_network(critic_path)