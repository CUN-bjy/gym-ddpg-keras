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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ActorNet():
	def __init__(self, num_obs, num_act):

		self.model = keras.Sequential(
			[
				layers.Input(shape=(num_obs,))
				layers.Dense(400, activation="softplus", name='layer1'),
				layers.Dense(300, activation="tanh", name='layer2')
				layers.Dense(num_act, name="actions")
			])

		self.model.compile(
			optimizer='adam', #loss='',
			metrics=['accuracy']
			)

	def __del__(self):
		pass


# for test
if __name__ == '__main__':
	actor = ActorNet()