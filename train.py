import tensorflow as tf
import numpy as np

from models.MoDS import MoDS

# Globals ---------------------------------------------------------------------
# Number of time steps being looked at
n_steps = 3
# Number of phoneme categories being used
n_cats = 4
# Number of mixture components to use
N = 4

# Learning rate
lr = 1e-4
# Max iterations
max_iters = 20000
# Print every
print_iter = 10
# Validate every
val_iter = 100

# Load the data ---------------------------------------------------------------
# data file
data = np.load('./data/place.npy')
data += 1e-6
data_sum = np.sum(data, axis=2, keepdims=True)
data /= data_sum

train_batch = data

# train_batch = np.ones((5000, n_steps, n_cats)) / (n_steps*n_cats)
val_batch = np.ones((500, n_steps, n_cats)) / (n_steps*n_cats)

# TODO: reweighting the data
# TODO: early stopping


# Start the session -----------------------------------------------------------
with tf.Session() as sess:
	# The data placeholder
	x = tf.placeholder(tf.float32, shape=[None, n_steps, n_cats])

	# Build the model
	model = MoDS(N, n_cats, n_steps)

	# The loss function will be negative log likelihood
	NLL = -model.log_prob(x)

	# Adam optimizer
	train_step = tf.train.AdamOptimizer(lr).minimize(NLL)

	sess.run(tf.global_variables_initializer())
	for iter_num in xrange(max_iters):
		train_step.run(feed_dict={x: train_batch})

		if iter_num % val_iter == 0:
			temp = NLL.eval(feed_dict={x: val_batch})
			print('\n' + '-'*80)
			print('Validation NLL: %g' % (temp))
			print('-'*80 + '\n')

		if iter_num % print_iter == 0:
			temp = NLL.eval(feed_dict={x: train_batch})
			print('NLL @ %d: %g' % (iter_num, temp))

	model.__str__(sess)