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

from keras.initializers import GlorotNormal
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation


class ActorNet():
	""" Actor Network for DDPG
	"""
	def __init__(self, in_dim, out_dim, lr_, tau_):
		self.obs_dim = in_dim
		self.act_dim = out_dim
		self.lr = lr_; self.tau = tau_

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
		h1_ = Dense(300, kernel_initializer=GlorotNormal())(input_)
		h1_b = BatchNormalization()(h1_)
		h1 = Activation('relu')(h1_b)

		# hidden_layer 2
		h2_ = Dense(400, kernel_initializer=GlorotNormal())(h1)
		h2_b = BatchNormalization()(h2_)
		h2 = Activation('relu')(h2_b)

		# output layer(actions)
		output_ = Dense(self.act_dim, kernel_initializer=GlorotNormal())(h2)
		output_b = BatchNormalization()(output_)
		output = Activation('tanh')(output_b)

		return Model(input_,output)

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

	def target_update(self):
		""" soft target update for training target actor network
		"""
		weights, weights_t = self.network.get_weights(), self.target_network.get_weights()
		for i in range(len(weights)):
			weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
		self.target_network.set_weights(weights_t)

	def predict(self, obs):
		""" predict function for Actor Network
		"""
		return self.network.predict(np.expand_dims(obs, axis=0))

	def target_predict(self, new_obs):
		"""  predict function for Target Actor Network
		"""
		return self.target_network.predict(new_obs)

	def save_network(self, path):
		self.target_network.save_weights(path + '_actor.h5')

	def load_network(self, path):
		self.target_network.load_weights(path)


# for test
if __name__ == '__main__':
	actor = ActorNet()