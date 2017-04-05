import numpy as np
from agg_data import load_phone_data
import operator

intervals = [
	[8,12],
	[12, 18],
	[18,24]
]

# intervals = [
# 	[8,12],
# 	[12, 16],
# 	[16,20],
# 	[20,24]
# ]

phone_data = load_phone_data()

num_good_ids = 0
data_points = []

for child in phone_data:
	age_ints = [0 for i in xrange(len(intervals))]
	
	for age in phone_data[child]:
		for i, interval in enumerate(intervals):
			if interval[0] <= age <= interval[1]:
				age_ints[i] += 1

	if 0 not in age_ints:
		num_good_ids += 1
		print(age_ints)
		data_points.append(reduce(operator.mul, age_ints))

print('\nIntervals:')
print(intervals)
print('\nNum Good Ids: %d' % num_good_ids)
print('\nData points per id:')
print(data_points)
print('\nTotal Data Points: %d' % sum(data_points))