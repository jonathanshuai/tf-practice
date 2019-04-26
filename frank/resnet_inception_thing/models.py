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


def build_inception_block(inputs, reduce_filters=[32, 48, 8], output_filters=[32, 64, 16],  
                          kernel_sizes=[1, 3, 5], strides=1, 
                          padding="VALID", name="inception_block", use_batch_norm=True):
    with tf.variable_scope(name):
        outputs = []
        
        for kernel_size, reduce_filter, output_filter in zip(kernel_sizes, reduce_filters, output_filters):
            if kernel_size == 1:
                output = tf.layers.conv2d(inputs, output_filter, kernel_size=kernel_size, strides=strides,
                              kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=params.STDDEV))
            else:
                pad_size = math.ceil(kernel_size / 2) - 1
                
                reduce = tf.layers.conv2d(inputs, reduce_filter, kernel_size=1,
                                          strides=1, padding="SAME")
                                          
                padded_reduce = tf.pad(reduce, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
                
                output = tf.layers.conv2d(padded_reduce, output_filter, kernel_size=kernel_size, strides=strides,
                              kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=params.STDDEV))
            if use_batch_norm:
                output = tf.contrib.layers.instance_norm(output)
            outputs.append(output)

        return tf.nn.relu(tf.concat(outputs, axis=3)) 
                
def build_hybrid_block(inputs, reduce_filters=[32, 48, 16, 16], output_filters=[32, 64, 32, 32], 
                       kernel_sizes=[1, 3, 5, 7], strides=1, 
                       padding="VALID", name="hybrid_block", use_batch_norm=True):
    with tf.variable_scope(name):
     
        b0 = build_inception_block(inputs, reduce_filters=reduce_filters, output_filters=output_filters,
                                   kernel_sizes=kernel_sizes, strides=strides, 
                                   padding=padding, name="b0", use_batch_norm=use_batch_norm)
        b1 = build_inception_block(b0, reduce_filters=reduce_filters, output_filters=output_filters,
                                   kernel_sizes=kernel_sizes, strides=1, 
                                   padding=padding, name="b1", use_batch_norm=use_batch_norm)
        
        if strides != 1:
            inputs = tf.layers.conv2d(inputs, inputs.shape[-1], kernel_size=1, strides=strides,
                      kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=params.STDDEV))
        if use_batch_norm:
            inputs = tf.contrib.layers.instance_norm(inputs)
                
                 
        return tf.nn.relu(b1 + inputs)

    
def hybrid_network(x, n_classes, n_layers=4, name="segmentation_network",
                                 reuse=False, use_batch_norm=True):
    with tf.variable_scope(name, reuse=reuse):

        x = tf.layers.conv2d(x, params.FILTERS, kernel_size=4, padding="SAME", name="conv0",
                            kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=params.STDDEV))
        x = tf.nn.leaky_relu(x, 0.2)
        
        
        x = build_hybrid_block(x, strides=1, name=f"hybrid_block_0", use_batch_norm=use_batch_norm)
        
        for i in range(n_layers - 1):
            x = build_hybrid_block(x, strides=2, name=f"hybrid_block_{i + 1}", use_batch_norm=use_batch_norm)
            
        x = tf.layers.conv2d(x, params.FILTERS, kernel_size=4, padding="SAME", name="final_conv",
                             kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=params.STDDEV))
        
        
        # Average pool reduces to Bx1x1x512
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = tf.squeeze(x, axis=[1, 2])
        x = tf.layers.dense(x, n_classes)
        
        return x
        
        
        
        
        
        
        
        