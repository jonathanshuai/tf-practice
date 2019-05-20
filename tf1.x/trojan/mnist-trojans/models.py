import math
import tensorflow as tf
from torch import nn
import numpy as np
from myutil import Parameters

params = Parameters('parameters.json')

class BasicCNN(nn.Module):
	def __init__(self, n_layers=2, filters=8, kernel_size=3, 
		strides=1, use_batch_norm=True, activation='relu', 
		height=28, width=28, channels=1):

		super(BasicCNN, self).__init__()

		self.n_layers = n_layers
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.use_batch_norm = use_batch_norm
		self.activation = activation

		self.height = height
		self.width = width
		self.channels = channels

		self.conv_layers = []
		pad_size = math.ceil(kernel_size / 2) - 1
		
		filter_sizes = [channels] + [filters] * n_layers

		for i in range(n_layers):
			conv_layer_ops = []
			conv_layer_ops.append(nn.Conv2d(filter_sizes[i], filter_sizes[i + 1],
											kernel_size=kernel_size, stride=strides,
											padding=pad_size))
			if use_batch_norm:
				conv_layer_ops.append(nn.BatchNorm2d(filter_sizes[i + 1]))

			if activation == 'relu':
				conv_layer_ops.append(nn.ReLU())
			elif activation == 'tanh':
				conv_layer_ops.append(nn.Tanh())

			conv = nn.Sequential(*conv_layer_ops)
			self.conv_layers.append(conv)

		self.conv = nn.Sequential(*self.conv_layers)
		self.fc = nn.Linear(self.height * self.width * self.filters, 10)

	def forward(self, x):
		x = self.conv(x)
		x = x.view(-1, self.height * self.width * self.filters)
		x = self.fc(x)
		return x

def basic_cnn(x, name,
				n_layers=2, 
				filters=8, 
				kernel_size=3, 
				strides=1, 
				use_batch_norm=True, 
				activation='relu',
				height=28, width=28):
	with tf.variable_scope(name):
		for _ in range(n_layers):
			x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, padding="SAME")
			if use_batch_norm:
				x = tf.layers.batch_normalization(x)
			if activation == 'relu':
				x = tf.nn.relu(x)
			elif activation == 'tanh':
				x = tf.nn.tanh(x)


	x = tf.reshape(x, [-1, height * width * filters])
	x = tf.layers.dense(x, 10)
	return x