import re
from pymatgen.core.periodic_table import Element

def get_diff(input1, input2, output):
	"""
	given two files containing compounds per line, output the compounds that exist in input1, but not in input2

	input1:
	Cs2Ag1Al1F6
	Cs2Ag1As1F6
	Cs2Ag1Au1Br6
	Cs2Ag1Au1F6

	input2:
	Cs2Ag1Al1F6 -3.06382545971
	Cs2Ag1As1F6 -2.46506153321
	Cs2Ag1Au1Br6 -0.897941594
	None 1000
	
	output:
	Cs2Ag1Au1F6
	"""

	list1 = []
	list2 = []
	with open(input1, 'r') as f:
		lines = f.readlines()
		for line in lines:
			list1.append(line.split()[0])

	with open(input2, 'r') as f:
		lines = f.readlines()
		for line in lines:
			list2.append(line.split()[0])

	list1 = set(list1)
	list2 = set(list2)

	with open(output, 'w') as f:
		for i in list1:
			if i not in list2:
				f.write(i + '\n')

def clean(input_file, output_file):
	"""
	clean input and give output

	input:
	Cs2Ag1Al1F6 -3.06382545971
	Cs2Ag1As1F6 -2.46506153321
	Cs2Ag1Au1Br6 -0.897941594
	None 1000

	output:
	Cs2Ag1Al1F6 -3.06382545971 eV
	Cs2Ag1As1F6 -2.46506153321 eV
	Cs2Ag1Au1Br6 -0.897941594 eV
	"""

	out_lines = []
	with open(input_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if not line.startswith('None'):
				out_lines.append(line)

	with open(output_file, 'w') as f:
		for line in out_lines:
			line = line.strip()
			f.write(line + ' eV\n')

def remove_lanthanoid(input_file, output_file):
	"""
	remove all compounds containing lanthanoid elements

	input:
	Cs2Ag1Al1F6 -3.06382545971 eV
	Cs2Ag1As1F6 -2.46506153321 eV
	Cs2Ag1La1Br6 -0.897941594 eV

	output:
	Cs2Ag1Al1F6 -3.06382545971 eV
	Cs2Ag1As1F6 -2.46506153321 eV

	"""
	out_lines = []
	with open(input_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			compound = line.split()[0]
			m = re.match("([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])([A-Za-z]+)([0-9])", compound)
			ele = Element(m.group(5))
			if not ele.is_lanthanoid:
				out_lines.append(line)

	with open(output_file, 'w') as f:
		for line in out_lines:
			f.write(line) 

if __name__ == '__main__':
	#get_diff('data/Dos_result', 'data/formation_result_1', 'data/diff_compounds')
	#clean('data/formation_energy/data_raw/formation_result_raw', 'data/formation_energy/data_raw/formation_all')
	#remove_lanthanoid('data/formation_energy/data_raw/formation_all', 'data/formation_energy/data_raw/formation_standard')