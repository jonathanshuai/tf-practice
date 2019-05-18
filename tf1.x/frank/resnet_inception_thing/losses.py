import tensorflow as tf

def segmentation_loss(output_probs, labels):
    return -tf.reduce_mean(
                          labels * tf.log(tf.clip_by_value(output_probs, 1e-10, 1.0))
                           + (1 - labels) * tf.log(tf.clip_by_value(1 - output_probs, 1e-10, 1.0))
                          )

