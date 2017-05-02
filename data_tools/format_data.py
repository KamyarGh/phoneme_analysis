import numpy as np
import os
import pickle
from collections import defaultdict
from catalog import get_catalog
from scipy.ndimage.filters import gaussian_filter1d


CORPUS_PATH = './davis_corpus'
CATEGORY_PATH = './category_path'
catalog = get_catalog()

# get map from file base name to age for all children
def get_fname_to_age_map():
	fname_to_age = {}

	for child_dir in os.listdir(CORPUS_PATH):
		child_dir = os.path.join(CORPUS_PATH, child_dir)

		age_set = set([])
		for data_file in os.listdir(child_dir):
			if '.cha' in data_file:
				# Read the age of the child
				with open(os.path.join(child_dir, data_file), 'r') as f:
					for line in f:
						if '@ID' in line:
							age = line.split('|')[3]
							age = age.split(';')
							if age[1] != '':
								age = float(age[0])*12 + float(age[1])
							else:
								age = float(age[0])*12
							break
				age = int(age*100 + 0.5)/100.0
				# duplicate age fixing
				if age in age_set:
					age += 0.01
				age_set.add(age)

				fname_to_age[data_file.split('.')[0]] = age

	return fname_to_age


def check_fname(fname):
	fname = fname.split('.')
	if 'query' not in fname[0] and fname[-1]=='xml':
		return [True, fname[0], fname[1]]
	return [False]

# Due to time limitations there was not ime for refactoring
# So we have this inefficient work to do
def smooth_data(data):
	for child in data:
		ages = []
		values = []

		for age in data[child]:
			ages.append(age)
			values.append(data[child][age])

		ages = np.array(ages)
		values = np.array(values)
		sort_inds = np.argsort(ages)
		ages = ages[sort_inds]
		values = values[sort_inds]

		values = gaussian_filter1d(values, 2, axis=0)

		# Rebuild the data dictionary
		new_dict = {}
		for i in xrange(ages.shape[0]):
			new_dict[ages[i]] = values[i]

		data[child] = new_dict

	return data



def format_data(categories, save_path=None):
	fname_to_age = get_fname_to_age_map()
	formatted_data = defaultdict(lambda: defaultdict(lambda: []))

	for cat in categories:
		cat = catalog[cat]
		cat_dir = os.path.join(CATEGORY_PATH, cat)

		for cat_data_file in os.listdir(cat_dir):
			check_result = check_fname(cat_data_file)
			if check_result[0]:
				child = check_result[1]
				child_record = check_result[2]

				total = 0
				with open(os.path.join(cat_dir, cat_data_file), 'r') as f:
					for line in f:
						if line.strip() == '</result>':
							total += 1
				
				formatted_data\
					[child]\
						[
							fname_to_age[
								child_record
							]
						].append(total)
				
				# print('{} --> {} --> {} --> {}'.format(
				# 		child,
				# 		child_record,
				# 		fname_to_age[child_record],
				# 		total
				# 	)
				# )

	for child in formatted_data:
		for age in formatted_data[child]:
			temp = formatted_data[child][age]
			temp_sum = float(sum(temp))
			if temp_sum != 0:
				formatted_data[child][age] = map(lambda p: p/temp_sum, temp)
				# print(formatted_data[child][age])

	# if save_path is not None:
	# 	pickle.dump(formatted_data, open(save_path, 'w'))

	formatted_data = smooth_data(formatted_data)

	return formatted_data


def print_formatted_data(formatted_data):
	for child in formatted_data:
		for age in formatted_data[child]:
			print('{} --> {} --> {}'.format(
					child,
					age,
					formatted_data[child][age]
				)
			)


if __name__ == '__main__':
	# print(get_fname_to_age_map())
	data = format_data(['vowels', 'c_voiced', 'c_voiceless'], './test_formatting')
	print_formatted_data(data)

	import matplotlib.pyplot as plt

	ages = []
	values = []

	# Nate, Paxton, Martin, Rowan, Hannah
	c = 'Paxton'
	for age in data[c]:
		ages.append(age)
		values.append(data[c][age])
	ages = np.array(ages)
	values = np.array(values)
	sort_inds = np.argsort(ages)
	ages = ages[sort_inds]
	values = values[sort_inds]

	# values = gaussian_filter1d(values, 2, axis=0)

	fig, ax = plt.subplots(1,1)
	ax.plot(ages, values[:, 0], label='vowels')
	ax.plot(ages, values[:, 1], label='cons. voiced')
	ax.plot(ages, values[:, 2], label='cons. voiceless')
	ax.set_xlabel('Months')
	ax.set_ylabel('Proportion')
	ax.set_title('Paxton\'s Records for Consonant-Vowel Distributions After Smoothing')
	ax.legend()
	ax.set_ylim([0, 0.8])
	plt.savefig('after.png')
	# plt.show()