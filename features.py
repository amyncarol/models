import numpy as np
import pandas as pd 
import re
from pymatgen.core.periodic_table import Element

def get_element_info(row):
	"""
	given a pandas row containing 2116 compound, returns an expanded row with elemental infomation
	"""
	m = re.match("([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])", row['compound'])
	row['A'], row['B'], row['Bp'], row['X'] = m.group(1), m.group(3), m.group(5), m.group(7)
	row['A_oxidation'], row['B_oxidation'], row['Bp_oxidation'], row['X_oxidation'] = 1, 1, 3, -1
	for i in ['A', 'B', 'Bp', 'X']:
		###element object
		ele = Element(row[i])

		###atomic number
		row[i+'_Z'] = ele.Z

		###atomia mass
		row[i+'_mass'] = ele.atomic_mass

		###electronegativity
		row[i+'_electronegativity'] = ele.X

		###row and group in periodic table
		row[i+'_row'] = ele.row
		row[i+'_group'] = ele.group

		###Mendeleev number
		row[i+'_mendeleev_no'] = ele.mendeleev_no

		###block character
		row[i+'_character'] = ele.block

		###atomic radii
		row[i+'_atomic_radii'] = ele.atomic_radius

		###ionic_radii
		if row[i+'_oxidation'] in ele.ionic_radii:
			row[i+'_ionic_radii'] = ele.ionic_radii[row[i+'_oxidation']]
		elif row[i+'_oxidation']-1 in ele.ionic_radii:
			row[i+'_ionic_radii'] = ele.ionic_radii[row[i+'_oxidation']-1]
		elif row[i+'_oxidation']+1 in ele.ionic_radii:
			row[i+'_ionic_radii'] = ele.ionic_radii[row[i+'_oxidation']+1]
		else:
			print('Cannot found the ionic radii for oxidation state {} +(-1, 0, 1): {}: {}'.format(row[i+'_oxidation'], \
						ele, ele.ionic_radii))

		###electronic_structure
		orbitals, energies, occupancies = get_atomic_structure(ele, 3) ##3-outmost subshells
		for j in range(len(orbitals)):
			row[i+'_'+str(j)+'_subshell_symmetry'] = orbitals[j]
			row[i+'_'+str(j)+'_subshell_energy'] = energies[j]
			row[i+'_'+str(j)+'_subshell_occupancy'] = occupancies[j]
	return row

def get_atomic_structure(ele, number_of_subshells = 3):
	"""
	given an element, return three lists of its n-outmost subshells,
	1\ symmetries of orbitals(s, p, d, f, e) (e means empty orbitals)
	2\ energies of orbitals
	3\ occupancies of orbitals
	"""
	if len(ele.atomic_orbitals) < number_of_subshells:
		n = len(ele.atomic_orbitals)
	else:
		n = number_of_subshells

	energy_dict = {}
	for k, v in ele.atomic_orbitals.items():
		energy_dict[v] = k
	
	occupancy_dict = {}
	ref_dict = {'s': 2, 'p': 6, 'd': 10, 'f': 14}

	for i in ele.full_electronic_structure:
		key = str(i[0]) + i[1]
		value = i[2]/ref_dict[i[1]]
		occupancy_dict[key] = value

	energies = sorted(energy_dict.keys(), reverse = True)[:n]
	orbitals = [energy_dict[i] for i in energies]
	occupancies = [occupancy_dict[i] for i in orbitals]
	orbitals = [i[1] for i in orbitals]

	if len(energies) < number_of_subshells:
		for i in range(number_of_subshells-len(energies)):
			energies.append(0)
			orbitals.append('e')
			occupancies.append(0)

	return orbitals, energies, occupancies

def generate_features(input_file, output_file):
	"""
	format of input_file(no header):
	Cs2Ag1Al1F6 -47.45251713 eV
	Cs2Ag1Au1Br6 -26.57069408 eV
	Cs2Ag1Bi1Br6 -29.72289641 eV
	Cs2Ag1Bi1Cl6 -32.98683175 eV
	"""
	data = pd.read_csv(input_file, sep = ' ', header = None, names = ['compound', 'energy', 'dummy'])
	data = data.drop(['dummy'], axis = 1)
	data = data.apply(get_element_info, axis = 1)
	data.to_csv(output_file, index = False)

def onehot(input_file, output_file):
	"""
	turn categorical columns into onehot indicator variables
	""" 
	df = pd.read_csv(input_file)
	df = df.drop(['compound'], axis = 1)
	df = pd.get_dummies(df)
	df.to_csv(output_file, index = False)


def get_X_1(df):
	"""
	some ionic radii information, following the double perovskite tolerance factor paper
	"""
	X = df.as_matrix(columns = ['A_Z', 'B_Z', 'Bp_Z', 'X_Z', \
		'A_ionic_radii', 'B_ionic_radii', 'Bp_ionic_radii', 'X_ionic_radii'])
	n = X.shape[0]
	for i in range(4, 8):
		for j in range(i+1, 8):
			column = (X[:, i]/X[:, j]).reshape((n, 1))
			X = np.hstack((X, column))
	print(X.shape)
	return X

def get_X_2(df):
	"""
	more features, following the Meredig formation energy paper
	"""

	###ionic radii ratios
	X_radii = df.as_matrix(columns = ['A_ionic_radii', 'B_ionic_radii', 'Bp_ionic_radii', 'X_ionic_radii'])
	n = X_radii.shape[0]
	for i in range(4):
		for j in range(i+1, 4):
			column = (X_radii[:, i]/X_radii[:, j]).reshape((n, 1))
			X_radii = np.hstack((X_radii, column))

	###others
	df = df.drop(['A_ionic_radii', 'B_ionic_radii', 'Bp_ionic_radii', 'X_ionic_radii'], axis=1)
	df = df.drop(['energy'], axis = 1)
	X = df.as_matrix()

	###together
	X = np.hstack((X, X_radii))

	print(X.shape)
	return X

def get_y(df):
	y = df.as_matrix(columns = ['energy'])
	print(y.shape)
	return y

if __name__ == '__main__':
	#generate_features('energy_result', 'energy_result_expanded')
	#onehot('energy_result_expanded', 'energy_result_expanded_onehot')
	data = pd.read_csv('energy_result_expanded_onehot')
	X = get_X_2(data)

	###the inplace change of the dataframe due to previous steps, reread data!!
	data = pd.read_csv('energy_result_expanded_onehot')
	y = get_y(data)







