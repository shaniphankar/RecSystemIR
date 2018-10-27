'''This program implements svd in two ways: One with the help of eigen values and the latter with the help of the inbuilt svd function'''
import numpy as np,scipy as sp
# from scipy import np.linalg
# from hardcodeSVD import data,retain
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
	# print(sigma)
	print(sigma.shape)
	lim=min(sigma.shape)
	total=get_sum(sigma)
	prev=total
	vals_req=1
	for i in range(lim,0,-1):
		temp=get_sum(sigma[range(i)])
		# print(i)
		# print(temp)
		# print(total)
		if(temp<.9*total):
			vals_req=i+1
			break
		else:
			prev=temp
	# print(vals_req)
	# print(U[:][:vals_req+1])
	V=V.transpose()
	Unew=U[:,range(vals_req)]
	sigmanew=np.diag(sigma[range(vals_req),range(vals_req)])
	Vnew=V[range(vals_req),:]
	Vnew=Vnew.transpose()
	print(Unew.shape)
	print(sigmanew.shape)
	print(Vnew.shape)
	return (Unew,sigmanew,Vnew)	

def SVD1(data,flag=True):
	'''Manual implementation which finds the eigen values and the eigen vectors'''
	data_trans=data.transpose()
	AAT=np.matmul(data,data_trans)
	eig_vals1,U=np.linalg.eig(AAT)
	U=np.real(U)
	ATA=np.matmul(data_trans,data)
	eig_vals2,V=np.linalg.eig(ATA)
	V=np.real(V)
	# print(eig_vals1)
	# print(eig_vals2)
	sigma=np.zeros(shape=data.shape)
	# print(sigma.shape)
	lim=0
	order1=np.flip(np.argsort(eig_vals1))
	order2=np.flip(np.argsort(eig_vals2))
	# print(eig_vals1[order1])
	# print(eig_vals2[order2])
	eig_vals1=eig_vals1[order1]
	eig_vals2=eig_vals2[order2]
	U=U[order1]
	V=V[order2]
	if(eig_vals1.shape[0]<eig_vals2.shape[0]):
		lim=eig_vals1.shape[0]
	else:
		lim=eig_vals2.shape[0]
	for i in range (0,lim,1):
		if(eig_vals1[i]>=0):
			sigma[i][i]=math.sqrt(eig_vals1[i])
	if(flag):
		U,sigma,V=reduce(U,sigma,V)
	return (U,sigma,V)

def SVD2(data):
	'''Library defined implementation for the SVD'''
	U,s_temp,V=np.np.linalg.svd(data)
	sigma=np.zeros(shape=data.shape)
	for i in range(0,s_temp.shape[0],1):
		sigma[i][i]=s_temp[i]

	return (U,sigma,V)


