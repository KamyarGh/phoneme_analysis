import os
import pickle
from collections import defaultdict

CORPUS_PATH = './davis_corpus'
CATEGORY_PATH = '/path/to/categories'

# get map from file base name to age for all children
def get_fname_to_age_map():
	fname_to_age = {}

	for child_dir in os.listdir(CORPUS_PATH):
		child_dir = os.path.join(CORPUS_PATH, child_dir)

		for data_file in os.listdir(child_dir):
			if '.cha' in data_file:
				# Read the age of the child
				with open(os.path.join(child_dir, data_file)) as f:
					for line in f:
						if '@ID' in line:
							age = line.split('|')[3]
							age = age.split(';')
							if age[1] != '':
								age = float(age[0])*12 + float(age[1])
							else:
								age = float(age[0])*12
							break
				fname_to_age[data_file.split('.')[0]] = age


def format_data(categories, save_path):
	fname_to_age = get_fname_to_age_map()
	formatted_data = defaultdict(lambda: defaultdict(lambda: []))

	for cat in categories:
		cat_dir = os.path.join(CATEGORY_PATH, cat)

		for child in os.listdir(cat_dir):
			cat_child_path = os.path.join(cat_dir, child)
			
			for cat_data_file in os.listdir(cat_child_path):
				total = 0
				with open(os.path.join(cat_child_path, cat_data_file)) as f:
					for line in f:
						if line.strip() == '</result>':
							total += 1
				formatted_data\
					[child]\
						[
							fname_to_age[
								cat_data_file.split('.')[0]
							]
						].append(total)
	for child in formatted_data:
		for age in formatted_data[child]:
			temp = formatted_data[child][age]
			temp = np.array(temp)
			temp_sum = np.sum(temp)
			if temp_sum != 0:
				formatted_data[child][age] = temp/temp_sum

	pickle.dump(formatted_data, save_path)