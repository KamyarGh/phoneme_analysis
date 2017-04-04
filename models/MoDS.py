import tensorflow as tf
import numpy as np

class Dirichlet():
	def __init__(self, K):
		self.K = K
		self.alphas = tf.Variable(
			tf.truncated_normal(
				[K],
				mean = 1,
				stddev = 0.1
			)
		)

	def log_prob(self, x):
		l_p = tf.reduce_sum(
			tf.multiply(
				self.alphas - 1,
				tf.log(x)
			),
			axis=1,
			keep_dims=True
		)

		return -tf.lbeta(self.alphas) + l_p


class DirichletSequence():
	def __init__(self, K, n_steps=3):
		self.K = K
		self.n_steps = n_steps

		self.dirs = [Dirichlet(K) for i in xrange(n_steps)]

	def log_prob(self, batch):
		# Input is a list of data points, one per step
		x_list = tf.unstack(batch, axis=1)
		assert(len(x_list) == self.n_steps, 'Need one datapoint per step!')

		return tf.reduce_sum(
			map(
				lambda (x, d): d.log_prob(x),
				zip(x_list, self.dirs)
			),
			axis=1,
			keep_dims=True
		)


class MoDS():
	def __init__(self, N, K, n_steps=3):
		# N: Number of mixtures
		# K: Number of categories in Dirichlet

		self.N = N
		self.K = K
		self.n_steps = n_steps

		# Initialize mixing weights
		self.un_norm_mix_weights = tf.truncated_normal(
			[N],
			mean = 1,
			stddev = 0.1
		)
		self.un_norm_mix_weights = tf.Variable(self.un_norm_mix_weights)

		# Build the mixture components
		self.components = [DirichletSequence(K, n_steps) for i in xrange(N)]

	def log_prob(self, batch):
		# Compute the log probabilities from each component distribution
		comp_log_probs = tf.concat(
			map(
				lambda comp: comp.log_prob(batch),
				self.components
			),
			axis=1
		)

		norm_mix_weights = self.un_norm_mix_weights / tf.reduce_sum(self.un_norm_mix_weights)
		log_norm_mix_weights = tf.log(norm_mix_weights)

		return tf.reduce_sum(
			tf.reduce_logsumexp(
				tf.add(comp_log_probs, log_norm_mix_weights),
				axis=1
			)
		)

	def __str__(self, sess):
		print('\nMoDS Parameters ({} Comp, {} Steps):'.format(self.N, self.n_steps))
		print(len(self.components))
		print(len(self.components[0].dirs))
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		temp = self.un_norm_mix_weights.eval(session=sess)
		temp = temp / sum(temp)
		print('Mixing Weights: {}'.format(temp))
		print('\n')
		for i in xrange(self.N):
			print('C%d:' % i)
			for j in xrange(self.n_steps):
				print('Step {}: {}'.format(j, np.exp(self.components[i].dirs[j].alphas.eval(session=sess))))