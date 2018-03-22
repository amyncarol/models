import pandas as pd 
import numpy as np 
from features import get_X_1, get_y, get_X_2
from plots import plot_gridsearch

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
import time

class NeuralNet(object):
	def __init__(self):
		self.model = Pipeline([
			('scaler', StandardScaler()),
			('regressor', MLPRegressor(max_iter = 200000))
		])

		self.scoring = 'neg_mean_absolute_error'
		#self.scoring = 'r2'

	def validate(self, X, y, alphas, hidden_layer_sizes):
		"""
		X and y are the training data
		hidden_layer_sizes is a list of hidden_layer_sizes
		alphas is a list of regularization paramters
		"""
		param_grid = {
			'regressor__hidden_layer_sizes' : hidden_layer_sizes, 
			'regressor__alpha' : alphas
		}

		grid = GridSearchCV(self.model, cv = 4, n_jobs = 4, param_grid = param_grid, scoring=self.scoring, verbose=10, return_train_score=True)
		grid.fit(X, y)

		return grid


if __name__ == '__main__':
	data = pd.read_csv('data/formation_energy/formation_standard_expanded_onehot')
	X = get_X_2(data)

	###the inplace change of the dataframe due to previous steps, reread data!!
	data = pd.read_csv('data/formation_energy/formation_standard_expanded_onehot')
	y = get_y(data).ravel()
	
	model = NeuralNet()
	grid = model.validate(X, y, [30, 10, 3, 1, 0.3, 0.1, 0.03], [(100, 100, ), (50, 50, 50, 50)])
	plot_gridsearch(grid)



