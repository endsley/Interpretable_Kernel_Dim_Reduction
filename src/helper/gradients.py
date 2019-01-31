#!/usr/bin/env python

from kernel_lib import *


#	assumes a minimization scheme
def gaussian_gradient(db):
	X = db['Dloader'].X
	N = X.shape[0]
	d = X.shape[1]
	σ = db['Dloader'].σ

	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=γ*Kx/(σ*σ)

	grad_A = np.zeros((d,d))
	for i in range(N):
		for j in range(N):
			x_ij = X[i,:] - X[j,:]
			A_ij = np.outer(x_ij, x_ij)

			grad_A += Ψ[i,j] * A_ij

	grad = grad_A.dot(db['W'])

	##	This is the faster matrix computation to check for error
	#D_Ψ = compute_Degree_matrix(Ψ)
	#Φ = 2*X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)
	#other_grad = Φ.dot(db['W'])

	#print(grad)
	#print('\n')
	#print(other_grad)

	return grad

