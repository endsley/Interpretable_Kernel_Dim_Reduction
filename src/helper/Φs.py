#!/usr/bin/env python

from kernel_lib import *


def gaussian_Φ_0(db):
	X = db['Dloader'].X
	σ = db['Dloader'].σ

	γ = db['compute_γ']()
	D_γ = compute_Degree_matrix(γ)

	Φ = X.T.dot(D_γ - γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	return Φ

def polynomial_Φ_0(db):
	X = db['Dloader'].X
	p = db['poly_power']
	c = db['poly_constant']

	Kx = poly_sklearn(X, p-1, c)
	γ = self.update_γ()
	Φ = -X.T.dot(γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	return Φ

def linear_Φ_0(self):
	db = self.db
	X = db['Dloader'].X
	γ = self.update_γ()
	Φ = -X.T.dot(γ).dot(X); 			

def gaussian_Φ_slow(db):
	X = db['Dloader'].X
	σ = db['Dloader'].σ
	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=γ*Kx

	F = np.zeros((db['Dloader'].d, db['Dloader'].d))
	for m in range(db['Dloader'].N):	
		for n in range(db['Dloader'].N):	
			ΔX = X[m,:] - X[n,:]
			A_ij = np.outer(ΔX,ΔX)
			F = F + Ψ[m,n]*A_ij

	return F

def gaussian_Φ(db):
	X = db['Dloader'].X
	σ = db['Dloader'].σ

	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=γ*Kx
	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)
	return Φ

def polynomial_Φ(db):
	X = db['Dloader'].X
	p = db['poly_power']
	c = db['poly_constant']

	Kx = poly_sklearn(X, p-1, c)
	γ = self.update_γ()
	Ψ = γ*Kx
	Φ = -X.T.dot(Ψ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	return Φ

def linear_Φ(self):
	db = self.db
	X = db['Dloader'].X
	γ = self.update_γ()
	Φ = -X.T.dot(γ).dot(X); 			

