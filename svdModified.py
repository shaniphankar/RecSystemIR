'''This program implements svd in two ways: One with the help of eigen values and the latter with the help of the inbuilt svd function'''
import numpy as np,scipy as sp
from scipy import linalg
from hardcode import data,retain
import math

def get_sum(sigma):
	'''Obtain the total energy present'''
	lim=min(sigma.shape)
	sum=0
	for i in range(0,lim,1):
		sum+=sigma[i][i]

	return sum

def reduce(U,sigma,V):
	'''Reducing the energy to the parameter present within the hardcode file'''
	lim=min(sigma.shape)
	total=get_sum(sigma)
	prev=total
	vals_req=0
	for i in range(lim-1,-1,-1):
		temp=get_sum(sigma[:i][:])
		if(temp<.9*total):
			vals_req=i+1
			break
		else:
			prev=temp

	return (U[:][:vals_req+1],sigma[:vals_req+1][:vals_req+1],V[:][vals_req+1])	

def SVD1(data):
	'''Manual implementation which finds the eigen values and the eigen vectors'''
	data_trans=data.transpose()
	AAT=np.matmul(data,data_trans)
	eig_vals1,U=linalg.eig(AAT)
	U=np.real(U)

	ATA=np.matmul(data_trans,data)
	eig_vals2,V=linalg.eig(ATA)
	V=np.real(V)

	sigma=np.zeros(shape=data.shape)

	lim=0
	if(eig_vals1.shape[0]<eig_vals2.shape[0]):
		lim=eig_vals1.shape[0]
	else:
		lim=eig_vals2.shape[0]

	for i in range (0,lim,1):
		sigma[i][i]=math.sqrt(eig_vals1[i])

	U,sigma,V=reduce(U,sigma,V)

	return (U,sigma,V)

def SVD2(data):
	'''Library defined implementation for the SVD'''
	U,s_temp,V=np.linalg.svd(data)
	sigma=zeros(shape=data.shape)
	for i in range(0,s_temp.shape[0],1):
		sigma[i][i]=s_temp[i]

	return (U,sigma,V)


