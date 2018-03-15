import pandas as pd 
import numpy as np 
from features import get_element_info, generate_features, get_X, get_y
from plots import plot_gridsearch

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
import time

class LinearModel(object):
	def __init__(self):
		self.model = Pipeline([
			('poly', PolynomialFeatures()),
			('scaler', StandardScaler()),
			('regressor', Ridge())
		])

		self.scoring = 'neg_mean_absolute_error'

	def validate(self, X, y, alphas, degrees):
		"""
		X and y are the training data
		degrees is a list of degrees of polynomial features
		alphas is a list of alphas of the ridge regression
		"""
		param_grid = {
		'poly__degree' : degrees, 
		'regressor__alpha' : alphas
		}

		grid = GridSearchCV(self.model, cv = 4, n_jobs = 4, param_grid = param_grid, scoring=self.scoring, verbose=10)
		grid.fit(X, y)

		return grid


class KRR(object):
	def __init__(self, kernel):
		self.model = Pipeline([
			('scaler', StandardScaler()),
			('regressor', KernelRidge())
		])

		self.scoring = 'neg_mean_absolute_error'
		self.kernel = kernel

	def validate(self, X, y, alphas, parameters):
		"""
		X and y are the training data
		alphas is a list of alphas of the ridge regression
		parameters is a list of gammas if rbf kernel
		parameters is a list of degrees if polynomial kernel
		"""
		if self.kernel == 'rbf':
			param_grid = {
				'regressor__alpha' : alphas, 
				'regressor__gamma' : parameters
			}
		elif self.kernel == 'polynomial':
			param_grid = {
				'regressor__alpha' : alphas,
				'regressor__degree' : parameters
			}

		grid = GridSearchCV(self.model, cv = 4, n_jobs = 4, param_grid = param_grid, scoring=self.scoring, verbose=10)
		grid.fit(X, y)

		return grid


if __name__ == '__main__':
	#generate_features('energy_result', 'energy_result_expanded')
	data = pd.read_csv('energy_result_expanded')
	X = get_X(data)
	y = get_y(data)
	#model = LinearModel()
	#grid = model.validate(X, y, [1000, 100, 10], [4, 5, 6])

	# model = KRR('rbf')
	# grid = model.validate(X, y, [1, 0.1, 0.01], [3, 1, 0.3, 0.1, 0.03])
	
	plot_gridsearch(grid)
	


