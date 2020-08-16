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
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten


class ActorNet():
	""" Actor Network for DDPG
	"""
	def __init__(self, in_dim, out_dim, lr_, tau):
		self.obs_dim = in_dim
		self.act_dim = out_dim
		self.lr = lr_

		# initialize actor network and target
		self.network = self.create_network()
		self.target_network = self.create_network()

		# initialize optimizer
		self.optimizer = self.create_optimizer()


	def create_network(self):
		""" Create a Actor Network Model using Keras
		"""

		# input layer(observations)
		input_ = Input(shape=(self.obs_dim,))

		# hidden layer 1
		h1 = Dense(300, activation='relu')(input_)
		
		# hidden_layer 2
		h2 = Dense(400, activation='relu')(h1)
		
		# output layer(actions)
		output_ = Dense(self.act_dim, activation='tanh')(h2)

		return Model(input_,output_)

	def create_optimizer(self):
		""" Create a optimizer for updating network
			to the optimal direction 
		""" 
		action_gdts = K.placeholder(shape=(None,self.act_dim))
		params_grad = tf.gradients(self.network.output, self.network.trainable_weights, -action_gdts)
		grads = zip(params_grad, self.network.trainable_weights)
		return K.funciton([self.network.input, action_gdts],[tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])


	def train(self, obs, grads):
		""" training Actor's Weights
		"""
		self.optimizer([obs,grads])

	def soft_update(self):
		""" soft target update for training target actor network
		"""
		pass

	def predict(self):
		""" predict function for Actor Network
		"""
		pass

	def target_predict(self):
		"""  predict function for Target Actor Network
		"""
		pass

	def save_network(self, path):
		self.target_network.save_weights(path + '_actor.h5')
		
	def load_network(self, path):
		self.target_network.load_weights(path)


# for test
if __name__ == '__main__':
	actor = ActorNet()