import numpy as np
import pprint
import os
import svdModified
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
	for i in R:
		emptyRow=[]
		for j in C:
			emptyRow.append(matrix[i-1][j-1])
		W.append(emptyRow)
	W=np.array(W)
	# print(W)
	#Using in built SVD for now. Replace with the other group's version
	X,Sig,Y=svdModified.SVD1(W,True)
	# X,Sig,Y=svdModified.SVD2(W)
	# print(Sig.shape)
	SigInv=np.linalg.pinv(Sig)
	# print(X)
	# print(SigInv.shape)
	# print(Y)
	# print(np.matmul(Y,np.matmul(SigInv,SigInv)))
	U=np.matmul(np.matmul(Y,(np.matmul(SigInv,SigInv))),np.transpose(X))
	print(U)
	print(U.shape)
	return Cmat,U,Rmat