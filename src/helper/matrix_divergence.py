#!/usr/bin/env python

import numpy as np
import sklearn.metrics


#	Note : matrices are assumed to be symmetric, PSD and low rank
class matrix_divergence():
	def __init__(self):
		pass


	def set_root_data(self, root_data):
		self.root = root_data
		self.root_mpd = np.median(sklearn.metrics.pairwise.pairwise_distances(root_data))

		gammaV = 1.0/(2*self.root_mpd*self.root_mpd)
		self.root_rbk = sklearn.metrics.pairwise.rbf_kernel(root_data, gamma=gammaV)

		l = self.root_rbk.shape[0]		#	center the RBK
		H = np.eye(l) - (1.0/l)*np.ones((l,l))
		self.root_rbk = H.dot(self.root_rbk).dot(H)


	def set_subset_data(self, subset_data):
		self.subset = subset_data
		self.subset_mpd = np.median(sklearn.metrics.pairwise.pairwise_distances(subset_data))

		gammaV = 1.0/(2*self.subset_mpd*self.subset_mpd)
		self.subset_rbk = sklearn.metrics.pairwise.rbf_kernel(subset_data, gamma=gammaV)

		l = self.subset_rbk.shape[0]	#	center the RBK
		H = np.eye(l) - (1.0/l)*np.ones((l,l))
		self.subset_rbk = H.dot(self.subset_rbk).dot(H)


	def get_kernel_sampling_feature(self, rbk, num_of_eigs=None):
		[U,S,V] = np.linalg.svd(rbk)

		cs = np.cumsum(S)/np.sum(S)
		L1_norm = S/np.sum(S)

		if num_of_eigs is None: num_of_eigs = np.sum(cs < 0.9) + 1
		keepCS = cs[0:num_of_eigs]
		L1_norm = L1_norm[0:num_of_eigs]

		return [keepCS, L1_norm, num_of_eigs]


	def get_divergence(self, root_dat=None, subset_dat=None):
		if root_dat is not None: self.set_root_data(root_dat)
		if subset_dat is not None: self.set_subset_data(subset_dat)

		[ksf_1, eigs_1, num_of_eigs] = self.get_kernel_sampling_feature(self.root_rbk)
		[ksf_2, eigs_2, num_of_eigs] = self.get_kernel_sampling_feature(self.subset_rbk, num_of_eigs)

		if len(eigs_1) > len(eigs_2):	
			extra_pad = len(eigs_1) - len(eigs_2)
			eigs_2 = np.pad(eigs_2, (0,extra_pad), 'constant')
			

		md = np.linalg.norm(eigs_1 - eigs_2, ord=np.inf)
		return md

if __name__ == '__main__':
	np.set_printoptions(precision=4)
	np.set_printoptions(threshold=np.nan)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)

	c1 = np.random.randn(10,2) + np.array([2,2])
	c2 = np.random.randn(10,2) + np.array([-2,-2])
	full_dat = np.vstack((c1,c2))

	c1 = np.random.randn(7,2) + np.array([2,2])
	c2 = np.random.randn(7,2) + np.array([-2,-2])
	sampled_dat = np.vstack((c1,c2))

	md = matrix_divergence()
	md.set_root_data(full_dat)
	md.set_subset_data(sampled_dat)
	print(md.get_divergence())
