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

class test_obj(test_base):
	def __init__(self):
		db = {}
		db['data_name'] = 'gauss_200'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file

		db['data_loader'] = np_loader
		db['TF_obj'] = linear_supv_dim_reduction
		db['W_optimize_technique'] = ism # orthogonal_optimization, ism, DimGrowth

		db['compute_error'] = None
		db['store_results'] = None
		db['run_only_validation'] = True

		db['q'] = 1
		db['num_of_clusters'] = 2
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma

		test_base.__init__(self, db)


	def parameter_ranges(self):
		W_optimize_technique = [ism, DimGrowth, orthogonal_optimization]	
		return [W_optimize_technique]

	def file_name_ranges(self):
		fnames = []
		for i in range(1, 10):
			n = i*100
			newFname = 'gauss_' + str(2*n)
			fnames.append(newFname)

		for i in range(1, 10):
			d = 20 * i
			newFname = 'gaussD_' + str(d+2)
			fnames.append(newFname)

		W_optimize_technique = [ism, DimGrowth, orthogonal_optimization]	
		return [fnames, W_optimize_technique]

prog = test_obj()
#prog.basic_run()
#prog.batch_run()
prog.batch_file_names()
