import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('n_classes', 'AUTO', 'Number of classes. Set to AUTO for extrapolation.')
tf.app.flags.DEFINE_string('image_size', 'AUTO', 'Size of input images expected by base model. Set to AUTO for extrapolation.')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Size of each batch.')
tf.app.flags.DEFINE_integer('n_epochs', 50, 'Number of epochs to train for.')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for gradient updates.')
tf.app.flags.DEFINE_integer('step_size', 8, 'Number of epochs before one step for exponential decay.')
tf.app.flags.DEFINE_float('gamma', 0.9, 'Amount to scale learning rate by.')
tf.app.flags.DEFINE_boolean('debug', False, 'Set to TRUE for debug.')
tf.app.flags.DEFINE_string('run_name', 'first_run', 'Name of the run. Used for tensorboard logging.')

print(FLAGS.learning_rate)
print(FLAGS.debug)
