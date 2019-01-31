#!/usr/bin/env python3

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from np_loader import *
from linear_supv_dim_reduction import *
from ism import *
from orthogonal_optimization import *
from DimGrowth import *
from grassman import *

class test_obj(test_base):
	def __init__(self):
		db = {}
		db['data_name'] = 'wine_75.00'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file

		db['data_loader'] = np_loader
		db['TF_obj'] = linear_supv_dim_reduction
		db['W_optimize_technique'] = ism 				# orthogonal_optimization, ism, DimGrowth, grassman

		db['compute_error'] = None
		db['store_results'] = None
		db['run_only_validation'] = True

		db['q'] = 4
		db['num_of_clusters'] = 3
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma

		test_base.__init__(self, db)


	def parameter_ranges(self):
		W_optimize_technique = [ism, DimGrowth, orthogonal_optimization]	
		#id_10_fold = [0] #range(10)
		#lambda_ratio = [2]
		#random.shuffle(σ_ratio)
	
		return [W_optimize_technique]


prog = test_obj()
prog.basic_run()
#prog.batch_run()

