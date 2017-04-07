import numpy as np
import pickle

intervals = []

def build_sub_iterator(first_iter_factory, second_iter_factory=None):
	if second_iter_factory is None:
		def sub_iterator():
			# The first iterator's elements will be turned into lists
			# 	we can append to in later iterator building
			for x in first_iter_factory():
				yield [x]
	else:
		def sub_iterator():
			for x in first_iter_factory():
				for y in second_iter_factory():
					yield x+[y]

	return sub_iterator


# Yes, yes, I know I can do binary search and whatnot
# 	but it has no practical advantage here cause I will
# 	only ever have a few intervals
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


		int_sets_iters = map(lambda p: lambda: iter(p), l)
		int_sets_iters[0] = build_sub_iterator(int_sets_iters[0])
		iter_factory = reduce(lambda i1, i2: build_sub_iterator(i1, i2), int_sets_iters)

		for element in iter_factory():
			batch_ids.append(child)
			batch_vectors.append(np.array(element))

	batch_vectors = np.array(batch_vectors)

	return batch_ids, batch_vectors