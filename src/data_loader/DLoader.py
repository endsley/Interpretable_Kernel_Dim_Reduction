#!/usr/bin/env python3

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from distances import *
from terminal_print import *
import numpy as np

class DLoader(Dataset):
	def __init__(self, db):
		self.dtype = np.float64				#np.float32
		self.training_mode = 'training'		# training, test, validation
		self.array_format = 'numpy'			# numpy, pytorch

		self.X = np.loadtxt(db['data_file_name'], delimiter=',', dtype=self.dtype)			

		if db['center_and_scale']: self.X = preprocessing.scale(self.X)
		self.Y = np.loadtxt(db['label_file_name'], delimiter=',', dtype=np.int32)

		self.N = self.X.shape[0]					# num of samples
		self.d = self.X.shape[1]					# num of Dims
		self.mpd = median_of_pairwise_distance(self.X)
		self.σ = db['σ_ratio']*self.mpd
		self.db = db


	def load_validation(self):
		db = self.db
		if db['run_only_validation']:
			self.X_valid = self.X
			self.Y_valid = self.Y
		else:
			self.X_valid = np.loadtxt(db['validation_data_file_name'], delimiter=',', dtype=self.dtype)
			if db['center_and_scale']: self.X_valid = preprocessing.scale(self.X_valid)
			self.Y_valid = np.loadtxt(db['validation_label_file_name'], delimiter=',', dtype=np.int32)


	def __getitem__(self, index):
		if self.mode == 'validation':
			return self.x_valid[index], self.y_valid[index], index
		elif self.mode == 'training':
			return self.X[index], self.Y[index], index
		else:
			print('Error unknown mode in dataset : %s'%self.mode)
			import pdb; pdb.set_trace()	

	def __len__(self):
		if self.training_mode == 'training':
			return self.X.shape[0]
		else:
			print('Error undefined mode in dataset : %s'%self.training_mode)
			import pdb; pdb.set_trace()	

