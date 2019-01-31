#!/usr/bin/python3
#	Note : This is designed for Python 3


import autograd.numpy as np
from pymanopt.manifolds import Stiefel
from pymanopt.manifolds import Grassmann
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

class grassman:
	def __init__(self, db):
		self.cost_function = db['compute_cost']
		self.gradient_function = db['compute_gradient']

		# (1) Instantiate a manifold
		#self.manifold = Stiefel(db['Dloader'].d, db['q'])
		self.manifold = Grassmann(db['Dloader'].d, db['q'])


		self.x_opt = None
		self.cost_opt = None
		self.db = db


	def run(self, x_init, max_rep=400):
		problem = Problem(manifold=self.manifold, cost=self.cost_function)
		solver = SteepestDescent()
		self.x_opt = solver.solve(problem)

		return self.x_opt	
