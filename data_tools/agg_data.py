# Aggregate the phoneme data
import os

DATA_PATH = './davis_corpus'

def load_phone_data(print_stats=False):
	phone_data = {}

	for child_dir in os.listdir(DATA_PATH):
		child_dir = os.path.join(DATA_PATH, child_dir)
		child_dict = {}

		for data_file in os.listdir(child_dir):
			if '.cha' in data_file:
				phone_dict = {}
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

				# Read the phoneme frequencies
				data_file = data_file.split('.')[0] + '.S.cex'
				with open(os.path.join(child_dir, data_file)) as f:
					counter = 0
					for line in f:
						if counter != 7:
							counter += 1
						else:
							if line != '':
								line = line.split()
								phone_dict[line[1]] = int(line[0])

				child_dict[age] = phone_dict
		phone_data[child_dir] = child_dict

	if print_stats:
		for child in phone_data:
			print('~~~~~~~~~~')
			print(child)
			for age in phone_data[child]:
				print('\n\t%f' % age)
				for ph in phone_data[child][age]:
					print('\t\t{}\t{}'.format(ph, phone_data[child][age][ph]))
			break

	return phone_data


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	phone_data = load_phone_data()
	ids, ages = [], []

	for i, child in enumerate(phone_data):
		for age in phone_data[child]:
			ids.append(i)
			ages.append(age)

	plt.scatter(ages, ids)
	plt.savefig('./figs/phone_data_summary.png')
	plt.show()