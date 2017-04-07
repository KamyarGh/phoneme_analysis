import numpy as np
import pickle
from format_data import format_data

intervals = [
	[8,12],
	[12, 18],
	[18,24]
]

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
# Returns -1 if not in any of the intervals
def interval_idx(age):
	for i, interval in enumerate(intervals):
		if interval[0] <= age <= interval[1]:
			return i
	return -1


def load_data(data, save_path=None):
	assert(len(intervals) > 0)

	num_intervals = len(intervals)

	batch_ids = []
	batch_vectors = []

	for child in data:
		int_sets = [set() for i in xrange(num_intervals)]

		for age in data[child]:
			idx = interval_idx(age)
			if idx != -1:
				int_sets[idx].add(age)

		# Build the cool iterator
		int_sets_iters = map(lambda p: lambda: iter(p), int_sets)
		int_sets_iters[0] = build_sub_iterator(int_sets_iters[0])
		iter_factory = reduce(lambda i1, i2: build_sub_iterator(i1, i2), int_sets_iters)

		for element in iter_factory():
			batch_ids.append(child)
			batch_vectors.append(
				[
					data[child][ei] for ei in element
				]
			)

	batch_vectors = np.array(batch_vectors)

	if save_path is not None:
		np.save(save_path, batch_vectors)
		np.save(save_path+'_ids', batch_ids)
	return batch_ids, batch_vectors


if __name__ == '__main__':
	data = format_data(['stops', 'affricates', 'fricatives', 'nasals'], './test_formatting')
	load_data(data, './data/manner')