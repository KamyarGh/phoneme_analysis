import numpy as np
import pickle

intervals = []

def build_sub_iterator(first_iter, second_iter):
	def sub_iterator():
		for i1 in first_iter:
			for i2 in second_iter:
				if isinstance(i2, list):
					yield [i1] + i2
				else:
					yield [i1, i2]


def interval_idx(age):
	for i, interval in enumerate(intervals):
		if interval[0] <= age <= interval[1]:
			return i
	raise Exception


def load_data(data_fname):
	assert(len(intervals) > 0)

	num_intervals = len(intervals)
	data = pickle.load(data_fname)

	batch_ids = []
	batch_vectors = []

	for child in data:
		int_sets = [set() for i in xrange(num_intervals)]

		for age in data[child]:
			idx = interval_idx(age)
			int_sets[idx].add(age)

		cur_iterator = iter(int_sets[-1])
		for i in xrange(num_intervals-2, -1, -1):
			cur_iterator = build_sub_iterator(iter(int_sets[i]), cur_iterator)

		for element in cur_iterator:
			batch_ids.append(child)
			batch_vectors.append(np.array(element))

	batch_vectors = np.array(batch_vectors)

	return batch_ids, batch_vectors