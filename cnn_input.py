
"""Routine for preparing the input for CGCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import time
import numpy as np

from models.data.solid_solution.generate_2116_solid_solution_vasp_input import get_structure_dict, get_pymatgen_formula
from models.crystal_graph import CrystalGraph
from pymatgen.analysis.local_env import *
from sklearn.model_selection import train_test_split

# Dimension of each item
MAX_N_SITES = 10

def generate_dataset(energy_file, structure_file, energies_file, sites_matrices_file, adj_matrices_file):
  """
  given energy_file and structure_file, generate the .npy files that stores all the data

  Args:

    energy_file: the file that contains the energy value for each structure

        Cs2Ag1Al1F6 -3.06382545971 eV
        Cs2Ag1As1F6 -2.46506153321 eV
        Cs2Ag1Au1Br6 -0.897941594 eV

    structure_file: the structure.pkl file

    energies_file, sites_matrices_file, adj_matrices_file: the filepath to store the .npy files which contains all data

  Writes: 
    energies_file, sites_matrices_file, adj_matrices_file
  """
  structure_dict = get_structure_dict(structure_file)
  nn_algorithm = VoronoiNN(tol = 0, cutoff = 10.0)

  energies = []
  sites_matrices = []
  adj_matrices = []

  with open(energy_file, 'r') as f:

    start_time = time.time()
    count = 0

    lines = f.readlines()
    for line in lines:
      count += 1
      if count%10 == 0:
        print("Processed {} items, using {} sec.".format(count, time.time()-start_time))

      compound = line.split(' ')[0]
      formula = get_pymatgen_formula(compound)

      if formula not in structure_dict:
        print("something wrong with {}".format(formula))

      else:
        ##energy
        energies.append(float(line.split(' ')[1]))
        structure = structure_dict[formula]

        ##crystal graph
        structure = structure_dict[formula]
        cg = CrystalGraph(structure, nn_algorithm, 'data/elements/element_features_onehot.csv', False)

        ##sites_matrix, empty sites are filled with zeros
        feature_length = (cg.feature_dict[0]).shape[0]
        sites_matrix = np.zeros((MAX_N_SITES, feature_length))
        for i in cg.feature_dict:
          sites_matrix[i, :] = cg.feature_dict[i].reshape(1, -1)
        sites_matrices.append(sites_matrix)

        ##adj_matrix, each entry represent the number of edges between two sites, \
        ## the number of edge of a site with itself is 1
        adj_matrix = np.zeros((MAX_N_SITES, MAX_N_SITES))
        for i in cg.graph:
          for j in range(MAX_N_SITES):
            if j != i:
              connectivity = cg.graph[i].count(j)
            else:
              connectivity = 0
            adj_matrix[i, j] = connectivity
        adj_matrices.append(adj_matrix)


  print(len(energies))
  print(len(sites_matrices))
  print(len(adj_matrices))

  np.save(energies_file, np.array(energies))
  np.save(sites_matrices_file, np.array(sites_matrices))
  np.save(adj_matrices_file, np.array(adj_matrices))


def get_training_eval(energies_file, sites_matrices_file, adj_matrices_file):
  """
  given all data files, generate training and evaluation sets
  Ensure we get the same split every time

  Args:
    energies_file, sites_matrices_file, adj_matrices_file: the filepath to store the .npy files which contains all data

  Returns:
    training_set, eval_set

    training_set is a dictionary containing train data:

              {'energies': array of shape (n_train, 1), \
               'sites_matrices': array of shape (n_train, max_n_sites, feature_length), \
               'adj_matrices': array of shape (n_train, max_n_sites, max_n_sites)}

    eval_set is a similar dictionary containing evaluation data
  """
  energies = np.load(energies_file).reshape(1, -1).T
  sites_matrices = np.load(sites_matrices_file)
  adj_matrices = np.load(adj_matrices_file)

  training_set = {}
  eval_set = {}

  training_set['energies'], eval_set['energies'], \
  training_set['sites_matrices'], eval_set['sites_matrices'],\
   training_set['adj_matrices'], eval_set['adj_matrices'] = \
  train_test_split(energies, sites_matrices, adj_matrices, test_size = 0.25, random_state=42, shuffle=True)

  return training_set, eval_set
 
def inputs(eval_data, batch_size, training_set, eval_set):
  """
  Construct input to feed to tensorflow graph

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: number of examples per batch
    training_set: training data
    eval_set: evaluation data

  Returns: 
    energies: tensor of shape (batch_size, 1)
    sites_matrices: tensor of shape (batch_size, max_n_sites, feature_length)
    adj_matrices: tensor of shape (batch_size, max_n_sites, max_n_sites)
  """
  if not eval_data:
    dataset = training_set
  else:
    dataset = eval_set

  with tf.name_scope('input'):
      # energies_initializer = tf.constant(dataset['energies'])
      # sites_matrices_initializer = tf.constant(dataset['sites_matrices'])
      # adj_matrices_initializer = tf.constant(dataset['adj_matrices'])
 
      # input_energies = tf.Variable(
      #     energies_initializer, trainable=False, collections=[])
      # input_sites_matrices = tf.Variable(
      #     sites_matrices_initializer, trainable=False, collections=[])
      # input_adj_matrices = tf.Variable(
      #     adj_matrices_initializer, trainable=False, collections=[])

      # Input data, pin to CPU because rest of pipeline is CPU-only
      with tf.device('/cpu:0'):
        input_energies = tf.constant(dataset['energies'])
        input_sites_matrices = tf.constant(dataset['sites_matrices'])
        input_adj_matrices = tf.constant(dataset['adj_matrices'])

      energy, sites_matrix, adj_matrix = tf.train.slice_input_producer(
          [input_energies, input_sites_matrices, input_adj_matrices])

      energies, sites_matrices, adj_matrices = tf.train.batch(
          [energy, sites_matrix, adj_matrix], batch_size=batch_size)

  return energies, sites_matrices, adj_matrices

if __name__=='__main__':
  # generate_dataset('/Users/yao/Google Drive/models/data/formation_energy/formation_standard', \
  #     '/Users/yao/Google Drive/models/data/formation_energy/structure.pkl', \
  #     '/Users/yao/Google Drive/models/data/formation_energy/energies.npy', \
  #     '/Users/yao/Google Drive/models/data/formation_energy/sites_matrices.npy', \
  #     '/Users/yao/Google Drive/models/data/formation_energy/adj_matrices.npy')

  # print(np.load('/Users/yao/Google Drive/models/data/formation_energy/energies.npy'))
  # print(np.load('/Users/yao/Google Drive/models/data/formation_energy/sites_matrices.npy'))
  print(np.load('/Users/yao/Google Drive/models/data/formation_energy/adj_matrices.npy')[0])

  # training_set, eval_set = get_training_eval('/Users/yao/Google Drive/models/data/formation_energy/energies.npy', \
  # 	'/Users/yao/Google Drive/models/data/formation_energy/sites_matrices.npy', \
  # 	'/Users/yao/Google Drive/models/data/formation_energy/adj_matrices.npy')

  # print(eval_set['energies'].shape)
  # print(eval_set['sites_matrices'].shape)
  # print(eval_set['adj_matrices'].shape)
  pass
