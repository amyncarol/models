import matplotlib.pyplot as plt
import numpy as np

def change_label(para_list):
	store_list = []
	for i in para_list:
		if i not in store_list:
			store_list.append(i)
	print(store_list)
	return [store_list.index(item) for item in para_list]

# a small test
# para_list = [1, 1, 2, 2, 4, 4, 5, 5]
# print(change_label(para_list))


def plot_gridsearch(grid):
	"""
	grid: the trained grid, only for 2d grid search
	"""
	params = grid.cv_results_['params']
	param_name = list(params[0].keys())
	para1 = [i[param_name[0]] for i in params]
	para2 = [i[param_name[1]] for i in params]
	para1_n = len(set(para1))
	para2_n = len(set(para2))

	
	if not isinstance(para1[0], int) and not isinstance(para1[0], float):
		para1 = change_label(para1)
	if not isinstance(para2[0], int) and not isinstance(para2[0], float):
		para2 = change_label(para2)

	if para1[0] == para1[1]:
		shape = (para1_n, para2_n)
	else:
		shape = (para2_n, para1_n)

	para1 = np.array(para1).reshape(shape)
	para2 = np.array(para2).reshape(shape)

	mean_test_score = grid.cv_results_['mean_test_score'].reshape(shape)
	std_test_score = grid.cv_results_['std_test_score'].reshape(shape)
	mean_train_score = grid.cv_results_['mean_train_score'].reshape(shape)
	std_train_score = grid.cv_results_['std_train_score'].reshape(shape)

	plt.subplot(121)
	if shape[0] == para1_n:
		for i in range(shape[1]):
			plt.errorbar(x = para1[:, i], y = mean_test_score[:, i], yerr = std_test_score[:, i], marker = 'o', label = 'test_'+str(para2[0, i]))
			plt.errorbar(x = para1[:, i], y = mean_train_score[:, i], yerr = std_train_score[:, i], label = 'train_'+str(para2[0, i]))
	else:
		for i in range(shape[0]):
			plt.errorbar(x = para1[i, :], y = mean_test_score[i, :], yerr = std_test_score[i, :], marker = 'o', label = 'test_'+str(para2[i, 0]))
			plt.errorbar(x = para1[i, :], y = mean_train_score[i, :], yerr = std_train_score[i, :], label = 'train_'+str(para2[i, 0]))
	plt.legend()

	plt.subplot(122)
	if shape[0] == para1_n:
		for i in range(shape[0]):
			plt.errorbar(x = para2[i, :], y = mean_test_score[i, :], yerr = std_test_score[i, :], marker = 'o', label = 'test_'+str(para1[i, 0]))
			plt.errorbar(x = para2[i, :], y = mean_train_score[i, :], yerr = std_train_score[i, :], label = 'train_'+str(para1[i, 0]))
	else:
		for i in range(shape[1]):
			plt.errorbar(x = para2[:, i], y = mean_test_score[:, i], yerr = std_test_score[:, i], marker = 'o', label = 'test_'+str(para1[0, i]))
			plt.errorbar(x = para2[:, i], y = mean_train_score[:, i], yerr = std_train_score[:, i], label = 'train_'+str(para1[0, i]))
	plt.legend()

	plt.show()

# def pca_project(X):
# 	"""to do"""
# 	_, _, V = np.linalg.svd(X)   
# 	V = V.T
# 	proj_matrix_3d = V[:, :n_componets]

# 	from mpl_toolkits.mplot3d import Axes3D

# 	X_demeaned = 
# 	X_proj3d = X_demeaned @ proj_matrix_3d

# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')

# 	xs = X_proj3d[:, 0]
# 	ys = X_proj3d[:, 1]
# 	zs = X_proj3d[:, 2]

# 	ax.set_xlabel('PC1')
# 	ax.set_ylabel('PC2')
# 	ax.set_zlabel('PC3')

# 	ax.scatter(xs, ys, zs)



