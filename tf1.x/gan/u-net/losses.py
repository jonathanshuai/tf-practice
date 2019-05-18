import tensorflow as tf

def cycle_consistency_loss(real_images, generated_images):
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake):
    return tf.reduce_mean(tf.squared_difference(prob_fake, 1))

def lsgan_loss_discriminator(prob_real, prob_fake):
    return (tf.reduce_mean(tf.squared_difference(prob_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake, 0))) * 0.5