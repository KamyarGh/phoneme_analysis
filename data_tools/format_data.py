import numpy as np
import os
import pickle
from collections import defaultdict
from catalog import get_catalog

CORPUS_PATH = './davis_corpus'
CATEGORY_PATH = './category_path'
catalog = get_catalog()

# get map from file base name to age for all children
def get_fname_to_age_map():
	fname_to_age = {}

	for child_dir in os.listdir(CORPUS_PATH):
		child_dir = os.path.join(CORPUS_PATH, child_dir)

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
				fname_to_age[data_file.split('.')[0]] = age

	return fname_to_age


def check_fname(fname):
	fname = fname.split('.')
	if 'query' not in fname[0] and fname[-1]=='xml':
		return [True, fname[0], fname[1]]
	return [False]


def format_data(categories, save_path):
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
			temp = np.array(temp).astype(float)
			temp_sum = np.sum(temp)
			if temp_sum != 0:
				formatted_data[child][age] = temp/temp_sum
				# print(formatted_data[child][age])

	# pickle.dump(formatted_data, save_path)
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
	data = format_data(['consonants', 'vowels'], './test_formatting')
	print_formatted_data(data)