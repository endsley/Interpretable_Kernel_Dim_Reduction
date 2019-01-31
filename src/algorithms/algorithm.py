#!/usr/bin/env python3

import numpy as np
import time 
from kernel_lib import *
from format_conversion import *
from classifier import *
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

class algorithm():
	def __init__(self, db):
		self.db = db
		self.optimizer = db['W_optimize_technique'](db)


	def verification_basic_info(self, start_time):
		db = self.db
		db['run_time'] = time.time() - start_time

		outstr = '\n\td : %d'%db['Dloader'].d + '\n'
		outstr += '\tq : %d'%db['q'] + '\n'
		outstr += '\tnum cluster : %d'%db['num_of_clusters'] + '\n'
		outstr += '\tσ_ratio : %d'%db['σ_ratio'] + '\n'
		outstr += '\tkernel type : %s'%db['kernel_type'] + '\n'
		outstr += '\tOptimization method : %s'%db['W_optimize_technique'].__name__ + '\n'
		outstr += '\tHSIC\n'
		outstr += '\t\tRun time : %.3f'%db['run_time'] + '\n'
		return outstr

