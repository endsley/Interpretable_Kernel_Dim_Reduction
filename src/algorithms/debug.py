
import numpy as np

def compare_Φ(db, Φ, Ψ):
	X = db['Dloader'].X
	F = np.zeros((db['Dloader'].d, db['Dloader'].d))
	for m in range(db['Dloader'].N):	
		for n in range(db['Dloader'].N):	
			ΔX = X[m,:] - X[n,:]
			A_ij = np.outer(ΔX,ΔX)
			F = F + Ψ[m,n]*A_ij

	Dif = np.linalg.norm(F - Φ)
	print('Error between Φ : %.3f'%Dif)
	import pdb; pdb.set_trace()

def check_W(db, Φ, W, W_λ):
	Λ = np.diag(W_λ)
	diff = Φ.dot(W) - W.dot(Λ)
	diff_norm = np.linalg.norm(diff)
	print('\nGradient : %.3f'%diff_norm)

