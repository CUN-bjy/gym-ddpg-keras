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

BUFFER_SIZE = 20000
class ddpgAgent():
	"""Deep Deterministic Policy Gradient(DDPG) Agent
	"""
	def __init__(self, env_, is_discrete=False, batch_size=100, w_per=True):
		# gym environments
		self.env = env_

		self.discrete = is_discrete
		self.obs_dim = env_.observation_space.shape[0]
		self.act_dim = env_.action_space.n if is_discrete else env_.action_space.shape[0]

		self.action_bound = (env_.action_space.high - env_.action_space.low) / 2 if not is_discrete else 1.
		self.action_shift = (env_.action_space.high + env_.action_space.low) / 2 if not is_discrete else 0.

		# initialize actor & critic and its targets
		self.discount_factor = 0.99
		self.actor = ActorNet(self.obs_dim, self.act_dim, self.action_bound, lr_=10e-4,tau_=10e-3)
		self.critic = CriticNet(self.obs_dim, self.act_dim, lr_=10e-3,tau_=10e-3,discount_factor=self.discount_factor)

		# Experience Buffer
		self.buffer = MemoryBuffer(BUFFER_SIZE, with_per=w_per)
		self.with_per = w_per
		self.batch_size = batch_size
		# OU-Noise-Process
		self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

	###################################################
	# Network Related
	###################################################
	def make_action(self, obs, t):
		""" predict next action from Actor's Policy
		"""
		# obs = np.expand_dims(obs, axis=0).astype(np.float32)
		action_ = self.actor.predict(obs)[0]
		a = np.clip(action_ + self.noise.generate(t), -self.action_bound, self.action_bound)
		return np.argmax(a) if self.discrete else a

	def update_networks(self, obs, acts, critic_target):
		""" Train actor & critic from sampled experience
		"""
		# update critic
		self.critic.train(obs, acts, critic_target)

		# get next action and Q-value Gradient
		n_actions = self.actor.network.predict(obs)
		q_grads = self.critic.Qgradient(obs, n_actions)

		# update actor
		self.actor.train(obs,self.critic.network,q_grads)

		# update target networks
		self.actor.target_update()
		self.critic.target_update()

	def replay(self, replay_num_):
		if self.with_per and (self.buffer.size() <= self.batch_size): return

		for _ in range(replay_num_):
			# sample from buffer
			states, actions, rewards, dones, new_states, idx = self.sample_batch(self.batch_size)

			# get target q-value using target network
			q_vals = self.critic.target_predict([new_states,self.actor.target_predict(new_states)])

			# bellman iteration for target critic value
			critic_target = np.asarray(q_vals)
			for i in range(q_vals.shape[0]):
				if dones[i]:
					critic_target[i] = rewards[i]
				else:
					critic_target[i] = self.discount_factor * q_vals[i] + rewards[i]

				if self.with_per:
					self.buffer.update(idx[i], abs(q_vals[i]-critic_target[i]))

			# train(or update) the actor & critic and target networks
			self.update_networks(states, actions, critic_target)


	####################################################
	# Buffer Related
	####################################################

	def memorize(self,obs,act,reward,done,new_obs):
		"""store experience in the buffer
		"""
		if self.with_per:
			q_val = self.critic.network([np.expand_dims(obs,axis=0),self.actor.predict(obs)])
			next_action = self.actor.target_network.predict(np.expand_dims(new_obs, axis=0))
			q_val_t = self.critic.target_predict([np.expand_dims(new_obs,axis=0), next_action])
			new_val = reward + self.discount_factor * q_val_t
			td_error = abs(new_val - q_val)[0]
		else:
			td_error = 0			

		self.buffer.memorize(obs,act,reward,done,new_obs,td_error)

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

	def load_weights(self, pretrained):
		""" Agent's Weights Loader
		"""
		self.actor.load_network(pretrained)
		self.critic.load_network(pretrained)