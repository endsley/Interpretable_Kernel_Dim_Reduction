#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib
import numpy as np
import random
import itertools
import socket


sys.path.append('./src')
sys.path.append('./src/data_loader')
sys.path.append('./src/algorithms')
sys.path.append('./src/helper')
sys.path.append('./src/optimization')
sys.path.append('./tests')

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from data_input import *
from hsic_algorithms import *
from linear_supv_dim_reduction import *
from ism import *
from orthogonal_optimization import *
from DimGrowth import *

class LSDR(test_base):
	def __init__(self, new_db):
		db = {}

		db['data_name'] = 'SDR'
		db['data_loader'] = data_input
		db['TF_obj'] = linear_supv_dim_reduction
		db['W_optimize_technique'] = ism # orthogonal_optimization, ism, DimGrowth

		db['ignore_verification'] = True
		db['compute_error'] = None
		db['store_results'] = None
		db['run_only_validation'] = True

		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma

		for i in new_db: db[i] = new_db[i]

		test_base.__init__(self, db)

	def train(self):
		self.HA = hsic_algorithms(self.db)
		self.HA.run()

	def get_projection_matrix(self):
		return self.HA.db['W']

	def get_reduced_dim_data(self):
		return self.db['X'].dot(self.HA.db['W'])

