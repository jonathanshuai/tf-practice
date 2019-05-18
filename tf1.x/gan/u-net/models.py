import math
import tensorflow as tf
import numpy as np
from myutil import Parameters

params = Parameters('parameters.json')


def truncated_normal(mean=0, stddev=0.02):
	return tf.initializers.truncated_normal(mean=mean, stddev=stddev)


def contracting_block(x, filters=64, kernel_size=3, 
						strides=1, pool_size=2, pool_strides=2,
						use_pad=False, padding="VALID", name="contracting",
						use_pool=True, use_batch_norm=True):
	
	with tf.variable_scope(name):
		pad_size = math.ceil(kernel_size / 2) - 1

		if use_pool:
			# Pool
			x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=pool_strides)

		# Convolution
		for i in range(2):
			if use_pad:
				x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
			
			x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, 
									strides=strides, padding=padding, name=f"b{i}",
									kernel_initializer=truncated_normal(0, params.STDDEV))

			if use_batch_norm:
				x = tf.contrib.layers.instance_norm(x)

			x = tf.nn.relu(x)
			

		return x

def expansive_block(x, crops, filters=64, 
						kernel_size=3, deconv_kernel_size=2,
						strides=1, deconv_strides=2, 
						use_pad=False, padding="VALID", deconv_padding="SAME",
						name="expansive", use_batch_norm=True):

	with tf.variable_scope(name):
		pad_size = math.ceil(kernel_size / 2) - 1
	
		# Deconv
		x = tf.layers.conv2d_transpose(x, filters, kernel_size=deconv_kernel_size,
							padding=deconv_padding, strides=deconv_strides, name="deconv0",
							kernel_initializer=truncated_normal(0, params.STDDEV))

		if use_batch_norm:
			x = tf.contrib.layers.instance_norm(x)

		x = tf.nn.relu(x)

		# Concat
		x = tf.concat([x, crops], axis=3)
		
		# Convolution
		for i in range(2):
			if use_pad:
				x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
			
			x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, 
									strides=strides, padding=padding, name=f"b{i}",
									kernel_initializer=truncated_normal(0, params.STDDEV))

			if use_batch_norm:
				x = tf.contrib.layers.instance_norm(x)

			x = tf.nn.relu(x)
			
	
		return x
	

def center_crop(inputs, y, x):
	inputs_shape = inputs.get_shape().as_list()
	y_margin = (inputs_shape[1] - y) // 2
	x_margin = (inputs_shape[2] - x) // 2

	return tf.slice(inputs, [0, y_margin, x_margin, 0],
							[inputs.shape[0], y, x, inputs.shape[3]])


class Unet:
	def __init__(self, n_layers=4, use_pad=False, 
					conv_kernel_size=3, deconv_kernel_size=2,
					name="unet", use_batch_norm=True):

		self.n_layers = n_layers
		self.conv_kernel_size = conv_kernel_size
		self.deconv_kernel_size = deconv_kernel_size
		self.name = name
		self.use_batch_norm = use_batch_norm
		self.use_pad = use_pad

	def run(self, x):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			contracting_inputs = []

			# Initial layers
			x = contracting_block(x, params.FILTERS, 
					kernel_size=self.conv_kernel_size,
					use_pad=self.use_pad, 
					name=f"contracting_0", 
					use_pool=False, use_batch_norm=self.use_batch_norm)

			# Maxpool contractions
			for i in range(self.n_layers):
				contracting_inputs.append(x)
				mult = 2 ** (i + 2)
				x = contracting_block(x, params.FILTERS * mult, 
					kernel_size=self.conv_kernel_size,
					use_pad=self.use_pad, 
					name=f"contracting_{i + 1}", use_batch_norm=self.use_batch_norm)

			for i in range(self.n_layers - 1, -1, -1):
				x_shape = x.get_shape().as_list()
				crop_size_y = x_shape[1] * 2
				crop_size_x = x_shape[2] * 2
				crop = center_crop(contracting_inputs[i], crop_size_y, crop_size_x)
				x = expansive_block(x, crop, kernel_size=self.conv_kernel_size, 
					deconv_kernel_size=self.deconv_kernel_size, 
					use_pad=self.use_pad, 
					name=f"expansive_{i}", use_batch_norm=self.use_batch_norm)

		x = tf.layers.conv2d(x, 1, kernel_size=1, strides=1, 
								padding="VALID", name="final_conv", 
								kernel_initializer=truncated_normal(0, params.STDDEV))
		return x

	def get_input_size(self, output_size):
		conv_reduction = 2 * (self.conv_kernel_size // 2)

		# Initial layers
		input_size = output_size

		for i in range(self.n_layers):
			input_size = input_size + 2 * conv_reduction
			input_size = input_size // 2

		for i in range(self.n_layers):
			input_size = input_size + 2 * conv_reduction
			input_size = input_size * 2


		input_size = input_size + 2 * conv_reduction
		return input_size

	def get_output_size(self, input_size):
		conv_reduction = 2 * (self.conv_kernel_size // 2)

		# Initial layers
		output_size = input_size - 2 * conv_reduction
		for i in range(self.n_layers):
			output_size = output_size // 2
			output_size = output_size - 2 * conv_reduction


		for i in range(self.n_layers):
			output_size = output_size * 2
			output_size = output_size - 2 * conv_reduction

		return output_size