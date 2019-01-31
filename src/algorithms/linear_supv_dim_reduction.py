#!/usr/bin/env python3

import autograd.numpy as np
from algorithm import *
from kernel_lib import *
from sklearn.metrics import accuracy_score
from terminal_print import *
from gradients import *
from Φs import *
from math import e
#import debug

#	max Tr[(DKuD)Kx]
#	W -> Kx -> D -> γ -> Σ ψA_i,j -> W
class linear_supv_dim_reduction(algorithm):
	def __init__(self, db):
		#	parent variables : λ0, conv_threshold, W, W_λ
		db['W'] = np.zeros((db['Dloader'].d ,db['q']))
		self.Ku = Ku_kernel(db['Dloader'].Y)
		N = db['Dloader'].N
			
		self.H = np.eye(N) - (1.0/N)*np.ones((N,N))
		self.γ = double_center(self.Ku, self.H)

		if db['W_optimize_technique'].__name__ == 'grassman':
			db['compute_cost'] = self.grassman_cost_function
		else: db['compute_cost'] = self.compute_cost	

		db['compute_gradient'] = self.compute_f_gradient
		db['compute_Φ'] = self.compute_Φ
		db['compute_γ'] = self.compute_γ	
		algorithm.__init__(self, db)
	
		print('Experiment : linear dimensionality reduction\n')

	def grassman_cost_function(self, W):
		new_X = np.dot(self.db['Dloader'].X, W)
		σ = self.db['Dloader'].σ
		γ = self.db['compute_γ']()

		#	compute gaussian kernel
		bs = new_X.shape[0]
		K = np.empty((0, bs))	
		for i in range(bs):
			Δx = new_X[i,:] - new_X
			exp_val = -np.sum(Δx*Δx, axis=1)/(2*σ*σ)
			K = np.vstack((K, e**(exp_val)))

		return -np.sum(γ*K)

	def compute_γ(self):
		return self.γ

	def compute_Φ(self, old_x):
		self.db['W'] = old_x

		if self.db['kernel_type'] == 'rbf':
			return gaussian_Φ(self.db)
		elif self.db['kernel_type'] == 'rbf_slow':
			return gaussian_Φ_slow(self.db)
		elif self.db['kernel_type'] == 'linear':
			return linear_Φ(self.db)
		elif self.db['kernel_type'] == 'polynomial':
			return polynomial_Φ(self.db)

	def compute_cost(self, W=None):
		[Kx, D] = Kx_D_given_W(self.db, setW=W)		#Kx
		γ = self.compute_γ()
		return -np.sum(γ*Kx)

	def compute_f_gradient(self, old_x):
		self.db['W'] = old_x

		if self.db['kernel_type'] == 'rbf':
			return gaussian_gradient(self.db)
		elif self.db['kernel_type'] == 'rbf_slow':
			return gaussian_gradient(self.db)
		elif self.db['kernel_type'] == 'linear':
			return linear_gradient(self.db)
		elif self.db['kernel_type'] == 'polynomial':
			return polynomial_gradient(self.db)



	def compute_Lagrangian_gradient(self):
		Φ = self.compute_Φ()
		[new_W, W_λ] = eig_solver(Φ, db['q'], mode='smallest')
		gradient = Φ.dot(db['W']) - db['W'].dot(np.diag(W_λ))
		print('Gradient :\n')
		print(gradient)
		return gradient			


	def update_U(self):
		pass

	def initialize_U(self):
		pass

	def initialize_W(self):
		if True:	#	ism W initialization
			if self.db['kernel_type'] == 'rbf':
				Φ = gaussian_Φ_0(self.db)
			elif self.db['kernel_type'] == 'rbf_slow':
				Φ = gaussian_Φ_0(self.db)
			elif self.db['kernel_type'] == 'linear':
				Φ = linear_Φ_0(self.db)
			elif self.db['kernel_type'] == 'polynomial':
				Φ = polynomial_Φ_0(self.db)
	
			[self.db['W'], self.db['W_λ']] = eig_solver(Φ, self.db['q'], mode='smallest')
		else:		#	using random initialization
			d = self.db['W'].shape[0]
			q = self.db['W'].shape[1]
			W = np.random.randn(d,q)
			[self.db['W'],R] = np.linalg.qr(W)		# QR ensures orthogonality

	def update_f(self):
		#gaussian_gradient(self.db)
		#self.db['run_debug_2'] = True
		#self.db['run_debug_3'] = True
		self.db['W'] = self.optimizer.run(self.db['W'])

	def outer_converge(self):
		return True


	def verify_result(self, start_time):
		db = self.db
		if 'ignore_verification' in db: return
		final_cost = self.compute_cost()

		db['Dloader'].load_validation()
		outstr = '\nExperiment : linear supervised dimensionality reduction : %s, final cost : %.3f\n'%(db['data_name'],final_cost)

		Y = db['Dloader'].Y
		X = db['Dloader'].X

		X_valid = db['Dloader'].X_valid
		Y_valid = db['Dloader'].Y_valid

		outstr += self.verification_basic_info(start_time)
		
		if not db['run_only_validation']:
			[out_allocation, nmi, svm_time] = use_svm(X,Y)
			acc = accuracy_score(Y, out_allocation)
			outstr += '\t\tTraining SVM NMI without dimension reduction : %.3f, acc : %.3f, time : %.4f'%(nmi, acc, svm_time) + '\n'
			[out_allocation, nmi, svm_time] = use_svm(X,Y, db['W'])
			acc = accuracy_score(Y, out_allocation)
			outstr += '\t\tTraining SVM NMI with dimension reduction : %.3f , acc : %.3f'%(nmi, acc) + '\n'

		[out_allocation, nmi, svm_time] = use_svm(X_valid, Y_valid)
		acc = accuracy_score(Y_valid, out_allocation)
		outstr += '\t\tTest Set SVM NMI without dimension reduction : %.3f, acc : %.3f, time : %.4f'%(nmi, acc, svm_time) + '\n'

		[out_allocation, nmi, svm_time] = use_svm(X_valid, Y_valid, db['W'])
		acc = accuracy_score(Y_valid, out_allocation)
		outstr += '\t\tTest Set SVM NMI with dimension reduction : %.3f, acc : %.3f '%(nmi, acc) + '\n'



		start_time = time.time() 
		clf = LinearDiscriminantAnalysis(n_components=db['q'])
		clf.fit(X, Y)
		lda_labels = clf.predict(X)
		lda_time = time.time() - start_time
		nmi = normalized_mutual_info_score(lda_labels, Y)
		acc = accuracy_score(Y, lda_labels)

		outstr += '\tLDA\n'
		outstr += '\t\tTraining NMI with LDA : %.3f, acc : %.3f'%(nmi, acc) + '\n'
		outstr += '\t\tLDA Run time : %.3f'%lda_time + '\n'

		start_time = time.time() 
		pca = PCA(n_components=db['q'])
		Xpca1 = pca.fit_transform(X)
		Xpca = pca.transform(X_valid)
		pca_time = time.time() - start_time


		[out_allocation, nmi, svm_time] = use_svm(Xpca1,Y)
		acc = accuracy_score(Y, out_allocation)
		outstr += '\tPCA\n'
		outstr += '\t\tTraining SVM NMI with PCA dimension reduction : %.3f, acc : %.3f'%(nmi, acc) + '\n'

		[out_allocation, nmi, svm_time] = use_svm(Xpca, Y_valid)
		acc = accuracy_score(Y_valid, out_allocation)
		outstr += '\t\tTest Set SVM NMI with PCA dimension reduction : %.3f, acc : %.3f'%(nmi, acc) + '\n'
		outstr += '\t\tPCA training time : %.3f'%pca_time + '\n'

		print(outstr)

		fin = open('./results/LSDR_' + db['data_name'] + '_' + db['W_optimize_technique'].__name__ + '.txt', 'w') 
		fin.write(outstr)
		fin.close()

