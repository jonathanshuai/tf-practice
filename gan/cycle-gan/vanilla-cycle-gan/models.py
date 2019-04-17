import math
import tensorflow as tf
import numpy as np
from myutil import Parameters

params = Parameters('parameters.json')


def build_residual_block(inputs, filters, name="residual_block", kernel_size=3, 
							strides=1, padding="VALID", use_batch_norm=True):
	with tf.variable_scope(name):
		pad_size = math.ceil(kernel_size / 2) - 1
		
		padded_inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
		b0 = tf.layers.conv2d(padded_inputs, filters, kernel_size=kernel_size, 
								strides=strides, padding=padding, name="b0",
								kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=params.STDDEV))
		if use_batch_norm:
			b0 = tf.contrib.layers.instance_norm(b0)

		b0 = tf.nn.relu(b0)


		b0 = tf.pad(b0, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
		b1 = tf.layers.conv2d(b0, filters, kernel_size=kernel_size, 
								strides=strides, padding=padding, name="b1",
								kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=params.STDDEV))
		
		if use_batch_norm:
			b1 = tf.contrib.layers.instance_norm(b1)

	return tf.nn.relu(b1 + inputs)



# 1. This network contains two stride-2 c onvolutions, 
# 2. several residual blocks, 
# 3. and two fractionally-strided convolutions with stride 1/2

def generator_6blocks(x, n_downsample=2, residual_blocks=6, name="generator",
							 reuse=False, use_batch_norm=True):
	with tf.variable_scope(name, reuse=reuse):

		# c7s1-64
		x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		x = tf.layers.conv2d(x, params.G_FILTERS, kernel_size=7, strides=1,
									padding="VALID", name="conv0")

		x = tf.nn.relu(x)

		# d128, d256
		for i in range(n_downsample):
			mult = 2 ** (i + 1)
			x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
			x = tf.layers.conv2d(x, params.G_FILTERS * mult, kernel_size=3, strides=2, 
										padding="VALID", name=f"downsample_{i}")
			if use_batch_norm:
				x = tf.contrib.layers.instance_norm(x)

			x = tf.nn.relu(x)


		# R256 x 6
		mult = 2 ** (n_downsample)
		for i in range(residual_blocks):
			x = build_residual_block(x, params.G_FILTERS * mult, name=f'residual_block_{i}')

		# u128
		# x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
		x = tf.layers.conv2d_transpose(x, 128, kernel_size=3, strides=2,
											padding="SAME", name="deconv0")

		# u64
		x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=2,
											padding="SAME", name="deconv1")

		# c7s1-3
		x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
		x = tf.layers.conv2d(x, 3, kernel_size=7, strides=1,
									padding="VALID", name="final_conv")

		return x
        
    
def discriminator(x, n_layers=4, name="discriminator", 
                          reuse=False, use_batch_norm=True):
    
    with tf.variable_scope(name, reuse=reuse):
        x = tf.layers.conv2d(x, params.D_FILTERS, kernel_size=4, padding="SAME", 
                                                 name="conv0")
        x = tf.nn.leaky_relu(x, 0.2)

        for i in range(n_layers):
            mult = 2 ** (i + 1)
            x = tf.layers.conv2d(x, params.D_FILTERS * mult, kernel_size=4, 
                                         padding="SAME", name=f"conv{i + 1}")
            if use_batch_norm:
                x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x, 0.2)
            
        x = tf.layers.conv2d(x, 1, kernel_size=4,
                                     padding="SAME", name="final_conv")

        return x

def fake_image_pool(n_fakes, fake, fake_pool):
    if n_fakes < params.POOL_SIZE:
        fake_pool[n_fakes] = fake
        return fake
    
    else:
        p = np.random.random()
        if p > 0.5:
            random_id = np.random.randint(0, params.POOL_SIZE - 1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp[None, :, :, :]
        else:
            return fake
            