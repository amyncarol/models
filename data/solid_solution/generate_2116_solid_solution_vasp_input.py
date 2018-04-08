import sys
sys.path.insert(0, '/Users/yao/Google Drive/mmtools/')
import re
from math import sqrt
import numpy as np

from mmtools.structure.solid_solution import SolidSolutionMaker, SolidSolutionFileWriter
from mmtools.utils.vasprun_structure_pickle import load_structures

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

def compare():
	"""too slow, don't use"""
	structure_list = load_structures('./structure.pkl')[0:10]
	with open('mixing_pairs.csv', 'w') as f:
		for i in range(len(structure_list)):
			for j in range(i+1, len(structure_list)):
				struc1 = structure_list[i]
				struc2 = structure_list[j]
				ssm = SolidSolutionMaker(struc1, struc2, 0.1)
				if ssm.can_mix(radii_diff = 0.2):
					f.write(struc1.formula + ',' + struc2.formula+'\n')

def get_structure_dict(struc_list_file):
	"""given a pkl file containing a list of structure, return a dict with
	 key being the formula and value being the structure

	 Args: 
	 	struc_list_file: the pkl file

	 Returns:
	 	struc_dict
	"""
	struc_dict = {}
	structure_list = load_structures(struc_list_file)
	for struc in structure_list:
		struc_dict[struc.formula] = struc
	return struc_dict

def generate_element_pairs(el_list, radius_diff):
	"""
	given a el_list, produce all mixing pairs within a radius_diff 

	Args:
		el_list: a list of elements
		radius_diff: the radius different won't be larger than this threshold

	Returns:
		pair_list: a list of tuples (ele_A, ele_B)
	"""
	pair_list = []
	for i in range(len(el_list)):
		for j in range(i+1, len(el_list)):
			el1 = el_list[i]
			el2 = el_list[j]
			abs_diff = abs(el1.atomic_radius_calculated - el2.atomic_radius_calculated)
			if abs_diff < radius_diff:
				pair_list.append((el1, el2))
	return pair_list

def screen_pairs(comp_pairs, calculated_compounds):
	"""
	given a list of compound pairs, and a list of already calculated compounds, 
	return a list of compound pairs which both compounds in the pair have been 
	calculated

	Args:
		comp_pairs: a list of compounds pairs ('A', 'B')
		calculated_compounds: a list of calculated compounds

	Returns:
		comp_pairs_screened: a list of screened compounds pairs
	"""
	comp_pairs_screened = []
	for pair in comp_pairs:
		if (pair[0] in calculated_compounds) and (pair[1] in calculated_compounds):
			comp_pairs_screened.append(pair)
	return comp_pairs_screened

def get_pymatgen_formula(compound):
	"""
	given a string of 'A2B1Bp1X6' return its pymatgen standard formula

	Args:
		compound: a string of form 'A2B1Bp1X6'

	Returns:
		formula : a formula style defined in pymatgen
	"""
	m = re.match("([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])", compound)
	A, B, Bp, X = m.group(1), m.group(3), m.group(5), m.group(7)
	struc = IStructure(lattice=[[1, 0.000000, 0.000000], 
								[0.5, sqrt(3)/2, 0.000000],
								[0.5, 1/2.0/sqrt(3.0), sqrt(2.0/3.0)]], 
                          species=[A, A, B, Bp, X, X, X, X, X, X], 
                          coords = [[0.2500, 0.2500, 0.2500], 
                          			[0.7500, 0.7500, 0.7500],
                                    [0.5000, 0.5000, 0.5000],
                                    [0.0000, 0.0000, 0.0000],
                                    [0.2550, 0.7450, 0.2550],
                                    [0.7450, 0.7450, 0.2550], 
                                    [0.7450, 0.2550, 0.7450], 
                                    [0.7450, 0.2550, 0.2550], 
                                    [0.2550, 0.7450, 0.7450],
                                    [0.2550, 0.2550, 0.7450]])
	return struc.formula

if __name__ == "__main__":
	###the elements considered
	A = [Element(i) for i in ['Cs']]
	B = [Element(i) for i in ['Cu', 'Ag', 'Au', 'In', 'Tl']]
	Bp= [Element(i) for i in ['Ga', 'In', 'Tl', 'As', 'Sb', 'Bi', 'Sc']]
	X = [Element(i) for i in ['Cl', 'Br', 'I']]

	##the element pair within a certain raidus difference
	B_pair = generate_element_pairs(B, 0.25)
	Bp_pair = generate_element_pairs(Bp, 0.25)
	X_pair = generate_element_pairs(X, 0.25)

	##generate all compound pairs, the two compounds in the pair has only one element different
	comp_pairs = []

	for a in [i.symbol for i in A]:
		for b in [i.symbol for i in B]:
			for bp in [i.symbol for i in Bp]:
				for x in [(i[0].symbol, i[1].symbol) for i in X_pair]:
					comp_pairs.append((a+'2'+b+'1'+bp+'1'+x[0]+'6', a+'2'+b+'1'+bp+'1'+x[1]+'6'))


	for a in [i.symbol for i in A]:
		for b in [i.symbol for i in B]:
			for bp in [(i[0].symbol, i[1].symbol) for i in Bp_pair]:
				for x in [i.symbol for i in X]:
					comp_pairs.append((a+'2'+b+'1'+bp[0]+'1'+x+'6', a+'2'+b+'1'+bp[1]+'1'+x+'6'))

	for a in [i.symbol for i in A]:
		for b in [(i[0].symbol, i[1].symbol) for i in B_pair]:
			for bp in [i.symbol for i in Bp]:
				for x in [i.symbol for i in X]:
					comp_pairs.append((a+'2'+b[0]+'1'+bp+'1'+x+'6', a+'2'+b[1]+'1'+bp+'1'+x+'6'))

	
	###get all calculated compounds
	calculated_compounds = []
	with open('formation_standard', 'r') as f:
		for l in f:
			calculated_compounds.append(l.split(' ')[0])

	##get pairs with both compounds calculated
	screend_pairs = screen_pairs(comp_pairs, calculated_compounds)

	##get all calculated final structures of pure double perovskite
	struc_dict = get_structure_dict('./structure.pkl')
	
	##get all pairs of structures
	struc_pairs = []
	for pair in screend_pairs:
		comp1 = pair[0]
		comp2 = pair[1]
		comp1_struc = struc_dict[get_pymatgen_formula(comp1)]
		comp2_struc = struc_dict[get_pymatgen_formula(comp2)]
		struc_pairs.append({comp1: comp1_struc, comp2: comp2_struc})
	
	##write all vasp run files
	error_lines = ''
	for pair in struc_pairs:
		keys = list(pair.keys())
		struc1 = pair[keys[0]]
		struc2 = pair[keys[1]]
		wd = '/Users/yao/Google Drive/models/data/solid_solution/vasp_inputs/' + keys[0] + '_' + keys[1]
		
		try:
			ssw = SolidSolutionFileWriter(struc1, struc2, wd, percent_list=np.linspace(0.05, 0.95, 5))
			ssw.write_vasp_files()
		except KeyError:
			error_lines += (keys[0] + ',' + keys[1] + '\n')
	with open('/Users/yao/Google Drive/models/data/solid_solution/vasp_inputs/error_file', 'w') as f:
		f.write(error_lines)
