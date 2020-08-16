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
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten


class CriticNet():
	""" Critic Network for DDPG
	"""
	def __init__(self, in_dim, out_dim, lr_, tau_):
		self.obs_dim = in_dim
		self.act_dim = out_dim
		self.lr = lr_; self.tau = tau_

		# initialize critic network and target
		self.network = create_network()
		self.target_network = create_network()

	def create_network(self):
		""" Create a Critic Network Model using Keras
			as a Q-value approximator function
		"""
		pass

	def gradients(self):
		"""
		"""
		pass


	def train_on_batch(self):
		"""
		"""
		pass

	def target_predict(self):
		"""
		"""
		pass

	def target_update(self):
		""" soft target update for training target critic network
		"""
		weights, weights_t = self.network.get_weights(), self.target_network.get_weights()
		for i in range(len(weights)):
			weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
		self.target_network.set_weights(weights_t)

	def save_network(self, path):
		self.target_network.save_weights(path + '_critic.h5')

	def load_network(self, path):
		self.target_network.load_weights(path)