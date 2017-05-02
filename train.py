import tensorflow as tf
import numpy as np
import pickle
from plot_tools import plot_model

from models.MoDS import MoDS

# Globals ---------------------------------------------------------------------
# Number of time steps being looked at
n_steps = 3
# Number of phoneme categories being used
n_cats = 3
# Number of mixture components to use
N = 4

# Learning rate
lr = 1e-4
# Max iterations
max_iters = 5000
# Print every
print_iter = 10
# Validate every
val_iter = 100

# Train set proportion
train_prop = 0.95

# Load the data ---------------------------------------------------------------
# data files
# data = np.load('./data/place.npy')
# ids = np.load('./data/place_ids.npy')
# child_counts = pickle.load(open('./data/place_counts', 'r'))
# 
data = np.load('./data/CV.npy')
ids = np.load('./data/CV_ids.npy')
child_counts = pickle.load(open('./data/CV_counts', 'r'))
# 
# data = np.load('./data/v_pos.npy')
# ids = np.load('./data/v_pos_ids.npy')
# child_counts = pickle.load(open('./data/v_pos_counts', 'r'))

# To avoid NaN errors
data += 1e-6
data_sum = np.sum(data, axis=2, keepdims=True)
data /= data_sum

# Reweight the data
num_data_points = data.shape[0]
data_weights = np.zeros(num_data_points)
child_weights = {}
num_data_points = float(num_data_points)

inv_weight_sum = 0
for child in child_counts:
	if child_counts[child] > 0:
		inv_weight_sum += num_data_points/child_counts[child]

for child in child_counts:
	if child_counts[child] > 0:
		child_weights[child] = num_data_points/(child_counts[child]*inv_weight_sum)

for i in xrange(data.shape[0]):
	data_weights[i] = child_weights[ids[i]]
print(data_weights.shape)

# Building the train and validation batches
train_batch = data

# train_batch = np.ones((5000, n_steps, n_cats)) / (n_steps*n_cats)
val_batch = np.ones((500, n_steps, n_cats)) / (n_steps*n_cats)

# TODO: early stopping

# TODO: Run on gpu
# Automatic Latex generation
# Posterior probabilities
# writeup


# Start the session -----------------------------------------------------------
with tf.Session() as sess:
	# The data placeholder
	x = tf.placeholder(tf.float32, shape=[None, n_steps, n_cats])
	x_weights = tf.placeholder(tf.float32, shape=[None])

	# Build the model
	model = MoDS(N, n_cats, n_steps)

	# The loss function will be negative log likelihood
	NLL = -model.log_prob(x, x_weights)

	# Adam optimizer
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(lr, global_step,
	                                           2500, 1, staircase=True)
	# Passing global_step to minimize() will increment it at each step.
	train_step = (
		tf.train.AdamOptimizer(lr).minimize(NLL, global_step=global_step)
	)

	sess.run(tf.global_variables_initializer())
	for iter_num in xrange(max_iters):
		train_step.run(
			feed_dict={
				x: train_batch,
				x_weights: data_weights
			}
		)

		# if iter_num % val_iter == 0:
		# 	temp = NLL.eval(feed_dict={x: val_batch})
		# 	print('\n' + '-'*80)
		# 	print('Validation NLL: %g' % (temp))
		# 	print('-'*80 + '\n')

		if iter_num % print_iter == 0:
			temp = NLL.eval(
				feed_dict={
					x: train_batch,
					x_weights: data_weights
				}
			)
			print('NLL @ %d: %g' % (iter_num, temp))

	model.__str__(sess)

	# Plot stats from trained model
	plot_name = 'Evolution of Consonant-Vowel Distributions'
	cat_names = ['vowels', 'voiced cons.', 'voiceless cons.']

	means, stds = model.get_mean_and_std(sess)
	plot_model(N, n_cats, n_steps, means, stds, cat_names, plot_name, save=True, save_fname='CV', show=False)