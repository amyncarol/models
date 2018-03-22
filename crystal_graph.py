import pandas as pd
import numpy as np

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.local_env import *
from pymatgen.core.periodic_table import Element

from features import get_atomic_structure

class CrystalGraph(object):
	"""
	a object representing the crystal graph object defined in:

	Xie, et al. 
	Crystal Graph Convolutional Neural Networks for Accurate and
	Interpretable Prediction of Material Properties
	"""
	def __init__(self, structure, nn_algorithm, table_file, toprint=False):
		"""
		Args:
			structure: a pymatgen structure object
			nn_algorithm: the algorithm used to find near neighbors. A NearNeighbors object.
						  if using MinimumDistanceNN, there is no guarantee that the bond is double-directed.
						  if using VoronoiNN, not clear if the bond is double-directed.

			table_file: the csv table that stores preprocessed element features
			toprint: whether print the elemental feature information or not
			
		Attributes:
			graph: a dict containing connectivity/bonding for each site
			feature_dict: a dict containing element features for each site			
		"""
		self.structure = structure
		self.nn_algorithm = nn_algorithm
		self.table_file = table_file
		self.toprint = toprint

		self.graph = self.get_graph()
		self.feature_dict = self.get_feature_dict()


	def get_graph(self):
		"""
		return a dict of list that contains the crystal graph.
						
		Returns:
			a dict of list.

			Perovskite ABC3 as a example:

			A_site, B_site, etc are all site indexes.

			{A_site: [C1_site, C1_site, C1_site, C1_site, C2_site, C2_site, C2_site, C2_site, C3_site, C3_site, C3_site, C3_site],
			 B_site: [C1_site, C1_site, C2_site, C2_site, C3_site, C3_site],
			 C1_site: [B_site, B_site, A_site, A_site, A_site, A_site],
			 C2_site: [B_site, B_site, A_site, A_site, A_site, A_site],
			 C3_site: [B_site, B_site, A_site, A_site, A_site, A_site]}
		"""
		graph_dict = {}
		sites = self.structure.sites
		for i in range(len(sites)):
			site_list = []
			for item in self.nn_algorithm.get_nn_info(self.structure, i):
				site_list.append(item['site_index'])
			graph_dict[i] = site_list
		return graph_dict

	def get_site_features(self, n):
		"""
		given the site index, return the feature vector of this site

		Args:
			n: site index, starting from 0
		

		Returns:
			a numpy array of size (n_features,), n_features is the number of features
		"""

		ele = self.structure.sites[n].specie
		return get_element_features(ele.symbol, self.table_file, self.toprint)

	def get_feature_dict(self):
		"""
		returns the a dict of feature vectors for each site

		Returns:
			{0: feature_vector, 1: feature_vector, ....}
		"""
		feature_dict = {}
		sites = self.structure.sites
		for i in range(len(sites)):
			feature_dict[i] = self.get_site_features(i)

		return feature_dict
	

############## below are more general functionalities for element feature generation ################################

def get_element_features(ele_symbol, table_file, toprint=False):
	"""
	given an element symbol, return its feature vectors including elemental information

	Args:
		ele_symbol: an element symbol
		table_file: the csv table that stores preprocessed element features
		print: whether print the elemental feature information or not

	Returns: 
		a numpy array of size (n,), n is the number of features

		if lack of data, return None
	"""
	df = pd.read_csv(table_file)
	series = df[df['element'] == ele_symbol].iloc[0, :-1]

	##handle the case where lack of data happens
	series = series.replace([np.inf, -np.inf], np.nan)

	if toprint:
		print(series)

	if pd.isna(series).any():
		print('the following feature for {} is not existing: '.format(ele_symbol))
		print(series[pd.isna(series)])

	else:
		feature_vector = series.as_matrix()
		return feature_vector


def _generate_element_feature_table(output_file):
	"""
	generate a panda table for elemental features, each row stores all features for an element.

	Stored as csv in output_file
	"""
	z_list = [i for i in range(1, 93)]
	element_list = []
	for z in z_list:
		element_list.append(Element.from_Z(z).symbol)
	df = pd.DataFrame({'element': element_list, 'Z': z_list})

	df = df.apply(_get_element_info, axis = 1)
	df.to_csv(output_file, index = False)


def _get_element_info(row):
	"""
	given a pandas row containing an element, returns an expanded row with more elemental infomation

	TODO: clean duplicate functionality in features.py : get_element_info
	"""
	###element object
	ele = Element(row['element'])

	###atomia mass
	row['mass'] = ele.atomic_mass

	###electronegativity
	row['electronegativity'] = ele.X

	###row and group in periodic table
	row['row'] = ele.row
	row['group'] = ele.group

	###Mendeleev number
	row['mendeleev_no'] = ele.mendeleev_no

	###block character
	row['character'] = ele.block

	###atomic radii
	row['atomic_radii'] = ele.atomic_radius

	###ionic_radii
	
	###electronic_structure
	orbitals, energies, occupancies = get_atomic_structure(ele, 3) ##3-outmost subshells
	for j in range(len(orbitals)):
		row[str(j)+'_subshell_symmetry'] = orbitals[j]
		row[str(j)+'_subshell_energy'] = energies[j]
		row[str(j)+'_subshell_occupancy'] = occupancies[j]
	
	return row

def _onehot(input_file, output_file):
	"""
	turn categorical columns into onehot indicator variables
	""" 
	df = pd.read_csv(input_file)
	element = df['element']
	df = df.drop(['element'], axis = 1)
	df = pd.get_dummies(df)
	df['element'] = element
	df.to_csv(output_file, index = False)



if __name__ == "__main__":
	###generate tables, do it just once, already generated
	#_generate_element_feature_table('data/elements/element_features.csv')
	#_onehot('data/elements/element_features.csv', 'data/elements/element_features_onehot.csv')
	
	##small test, test get_element_features
	#print(get_element_features('Si', 'data/elements/element_features_onehot.csv', False))


	###small test, test CrystalGraph
	structure = Poscar.from_file("test_files/POSCAR.mp-5827_CaTiO3.vasp").structure
	#nn_algorithm = MinimumDistanceNN(tol = 0.1, cutoff = 10.0)
	#nn_algorithm = MinimumDistanceNN(tol = 0.42, cutoff = 3.0)  tolerance too large has its problem
	nn_algorithm = VoronoiNN(tol = 0, cutoff = 10.0)
	cg = CrystalGraph(structure, nn_algorithm, 'data/elements/element_features_onehot.csv', False)
	
	print('\n')
	print(cg.structure)
	print('\n')
	print(cg.nn_algorithm)
	print('\n')
	print(cg.graph)
	print('\n')
	print(cg.feature_dict)
	print('\n')
	
	


