import numpy as np
import matplotlib.pyplot as plt

def plot_model(N, n_cats, n_steps, means, stds, cat_names, plot_name, save=False, save_fname='', show=False):
	if save:
		assert(save_fname != '')

	y_max = np.amax(means+stds) + 0.05

	width = 0.75
	inds = np.arange(1, n_cats+1)-width/2.0
	colors = ['purple', 'cyan', 'yellow', 'r', 'g', 'b', 'pink', 'purple', 'cyan']

	for i in xrange(N):
		# plt.figure(i+1)
		fig, ax = plt.subplots(n_steps, 1, sharex=True)

		for j in xrange(n_steps):
			step = n_steps - j - 1
			# , capsize=7, error_kw={'linewidth':3}
			ax[j].bar(inds, means[i,step], width, color=colors[step], yerr=stds[i,step], ecolor='green', error_kw={'capsize':4, 'linewidth':3})
			ax[j].set_ylabel('Stage {}'.format(step+1))
			ax[j].set_xticks(inds + width / 2)
			ax[j].set_xticklabels(cat_names)
			# ax[j].set_yticks(np.arange(0,y_max,0.1))
			ax[j].set_xlim([width/2,n_cats+1-width/2])
			ax[j].set_ylim([0,y_max])
			
			if j == 0:
				ax[j].set_title(plot_name + ' in Group {}'.format(i+1))
		
		if show:
			plt.show()

		if save:
			plt.savefig(save_fname + '_{}.png'.format(i+1))


if __name__ == '__main__':
	N = 4
	n_cats = 4
	n_steps = 3

	plot_name = 'Evolution Distribution of Manners of Articulation'
	cat_names = ['stops', 'affricates', 'fricatives', 'nasals']

	means = np.ones([N, n_steps, n_cats])/n_cats
	stds = 0.02*np.ones([N, n_steps, n_cats])

	plot_model(N, n_cats, n_steps, means, stds, cat_names, plot_name, save=True, save_fname='test_plotting', show=False)