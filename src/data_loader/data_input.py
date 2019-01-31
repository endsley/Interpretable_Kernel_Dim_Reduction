#!/usr/bin/env python

from test_base import *
import sklearn.metrics
import numpy as np
from DLoader import *

class data_input(DLoader):
	def __init__(self, db):
		db['data_name'] = 'LSDR'
		self.dtype = np.float64				#np.float32
		self.training_mode = 'training'		# training, test, validation
		self.array_format = 'numpy'			# numpy, pytorch

		self.X = db['X']

		if db['center_and_scale']: self.X = preprocessing.scale(self.X)
		self.Y = db['Y']

		self.N = self.X.shape[0]					# num of samples
		self.d = self.X.shape[1]					# num of Dims
		self.mpd = median_of_pairwise_distance(self.X)
		self.σ = db['σ_ratio']*self.mpd
		self.db = db


