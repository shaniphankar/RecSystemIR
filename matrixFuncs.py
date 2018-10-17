import numpy as np
import pprint
import os
from random import choices

def CUR(matrix,r):
	nR=matrix.sum(axis=1)
	nC=matrix.sum(axis=0)
	# print(nR)
	# print(nC)
	full=matrix.sum()
	# print(full)
	weightsR=nR/full
	weightsC=nC/full
	choicesR=range(3952)
	choicesC=range(6040)
	C=choices(choicesC,weightsC,k=r)
	R=choices(choicesR,weightsR,k=r)
	Rmat=matrix[R]
	Cmat=matrix[:,C]
	C=np.sort(C)
	R=np.sort(R)
	W=[]
	for i in C:
		emptyRow=[]
		for j in R:
			emptyRow.append(matrix[j-1][i-1])
		W.append(emptyRow)
	W=np.array(W)
	print(W)
	#Using in built SVD for now. Replace with the other group's version
	X,Sig,Y=np.linalg.svd(W,full_matrices=False)
	SigInv=np.linalg.pinv(np.diag(Sig))
	print(X)
	print(SigInv)
	print(Y)
	U=np.matmul(np.matmul(Y,(np.matmul(SigInv,SigInv))),np.transpose(X))
	print(U)
	return Cmat,U,Rmat