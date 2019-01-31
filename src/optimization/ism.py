#!/usr/bin/env python3

import numpy as np
from optimization import *
from kernel_lib import *
from terminal_print import *


class ism(optimization):
	def __init__(self, db):
		optimization.__init__(self, db)
		self.λ0 = None
		self.conv_threshold = 0.01
		self.W = None
		self.W_λ = None

	def run(self, old_x, max_rep=200):
		q = old_x.shape[1]
		new_x = old_x
		old_λ = np.random.randn(1,q)

		for i in range(max_rep):
			Φ = self.db['compute_Φ'](old_x)
			[new_x, new_λ] = eig_solver(Φ, q)
			if self.inner_converge(new_x, old_x, new_λ, old_λ): 
				self.db['W'] = new_x
				break;
			old_x = new_x
			old_λ = new_λ

			#print('Cost %.3f'%db['compute_cost']())
			#if np.linalg.norm(old_λ - new_λ)/np.linalg.norm(old_λ) < 0.01: break

		return self.db['W']

	def inner_converge(self, new_x, old_x, new_λ, old_λ):
		until_W_converges = False

		if until_W_converges:
			diff_mag = np.linalg.norm(new_x - old_x)/np.linalg.norm(new_x)
			#write_to_current_line('W difference : %.3f'%diff_mag)
			print('\t\tW difference : %.3f'%diff_mag)
			if diff_mag < 0.001:
				return True
		else:
			diff = np.linalg.norm(old_λ - new_λ)/np.linalg.norm(old_λ)
			cost = self.db['compute_cost']()

			#txt = '\teigen differential : %.4f, cost : %.4f'%(diff, cost)
			#print(txt)

			if diff < self.conv_threshold: return True

		return False

