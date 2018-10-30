import numpy as np
import pprint
import os
from random import randint
import matrixFuncs
import CF
import CF_Baseline
import svdModified

def sample(number_of_items,number_of_users,data):
	"""
	The following function divides the various data points in a 70:30 fashion.
	To do this, it uses the permuatation function of Python's inbuilt random module.
	number_of_items: contains the number of items in the ratings matrix
	number_of_users: contains the number of users in the ratings matrix
	data: Contains the ratings matrix from which the sampling is done
	"""
	selected_data={}
	training_data=[]
	test_data=[]
	# print(data.shape)
	n=0
	indices=[]
	for i in range(0,number_of_items):
		for j in range(0,number_of_users):
			indices.append((i,j))
	indices=np.random.permutation(indices)
	print(indices.shape)
	for x in range(indices.shape[0]):
		if((x/indices.shape[0])<0.7):
			training_data.append((indices[x][0],indices[x][1],data[indices[x][0]][indices[x][1]]))
		else:
			test_data.append((indices[x][0],indices[x][1]))
	return training_data,test_data
	
def main():
	"""
	This function generates the various collab matrices as output.
	These matrices are later fed into rmse.py to obtain the various metrics associated with each method.
	collab_matrix1: Saved as collab_matrixCF.npy. Corresponds to normal Collaborative Filtering case.
	collab_matrix2: Saved as collab_matrixCFBaseline.npy. Corresponds to Baseline Collaborative Filtering case.
	collab_matrix3: Saved as collab_matrixSVD.npy. Corresponds to normal SVD case.
	collab_matrix4: Saved as collab_matrixSVD90.npy. Corresponds to SVD case with 90% energy retained.
	collab_matrix5: Saved as collab_matrixCUR.npy. Corresponds to normal CUR case.
	collab_matrix6: Saved as collab_matrixCUR90.npy. Corresponds to CUR case in which intermediate SVD retains 90% energy.
	"""
	pp=pprint.PrettyPrinter(indent=4)
	mID_names={}
	number_of_items=3952
	number_of_users=6040
	top_k=5
	mID_uID_rating=np.zeros(shape=(3952,6040))
	f=open(os.getcwd()+'/ml-1m/ratings.dat',encoding='latin-1')#Reading file as input and storing it as 2D numpy array
	for line in f:
		mID_uID_rating[int(line.split('::')[1])-1][int(line.split('::')[0])-1]=line.split('::')[2]
	f.close()
	print("1**************")#This corresponds to the standard Collaborative Filtering Case
	training_data1,test_data1=sample(3952,6040,mID_uID_rating)
	collab_matrix1 = np.zeros(shape=(number_of_items,number_of_users))
	for i in training_data1:
		collab_matrix1[i[0]][i[1]] = i[2]
	np.save("test_dataCF.npy",test_data1)
	collab_matrix1 = CF.normalise_collab_matrix(mID_uID_rating,number_of_items,number_of_users,top_k,collab_matrix1)
	np.save("collab_matrixCF.npy",collab_matrix1)
	print("2**************")#This corresponds to the collaborative filtering with the baseline approach
	training_data2,test_data2=sample(3952,6040,mID_uID_rating)
	collab_matrix2 = np.zeros(shape=(number_of_items,number_of_users))
	for i in training_data2:
		collab_matrix2[i[0]][i[1]] = i[2]
	np.save("test_dataCFBaseline.npy",test_data2)
	collab_matrix2 = CF_Baseline.compute(mID_uID_rating,number_of_items,number_of_users,top_k,collab_matrix2)
	np.save("collab_matrixCFBaseline.npy",collab_matrix2)
	print("3**************")#This corresponds to using SVD for approximating our ratings
	training_data3,test_data3=sample(3952,6040,mID_uID_rating)
	np.save("test_dataSVD.npy",test_data3)
	collab_matrix3 = np.zeros(shape=(number_of_items,number_of_users))
	for i in training_data3:
		collab_matrix3[i[0]][i[1]] = i[2]
	U,Sig,V=svdModified.SVD1(collab_matrix3,False)
	SVDMat=np.matmul(np.matmul(U,Sig),V.transpose())
	np.save("collab_matrixSVD.npy",SVDMat)
	print("4**************")#Here, we use SVD to approximate ratings. However, we remove concepts until 90% of the variation can still be explained
	training_data4,test_data4=sample(3952,6040,mID_uID_rating)
	np.save("test_dataSVD90.npy",test_data4)
	collab_matrix4 = np.zeros(shape=(number_of_items,number_of_users))
	for i in training_data4:
		collab_matrix4[i[0]][i[1]] = i[2]
	U,Sig,V=svdModified.SVD1(collab_matrix4,True)
	SVDMat=np.matmul(np.matmul(U,Sig),V.transpose())
	np.save("collab_matrixSVD90.npy",SVDMat)
	print("5**************")#Here, we use CUR to approximate ratings matrix
	training_data5,test_data5=sample(3952,6040,mID_uID_rating)
	np.save("test_dataCUR.npy",test_data5)
	collab_matrix5 = np.zeros(shape=(number_of_items,number_of_users))
	for i in training_data5:
		collab_matrix5[i[0]][i[1]] = i[2]
	C,U,R=matrixFuncs.CUR(collab_matrix5,1000,True)
	CURMat = (np.matmul((np.matmul(C,U)),R))
	np.save("collab_matrixCUR.npy",CURMat)
	print("6**************")#CUR is used here. However, the SVD routine called from within CUR corresponds to the 90% energy case.
	training_data6,test_data6=sample(3952,6040,mID_uID_rating)
	np.save("test_dataCUR90.npy",test_data6)
	collab_matrix6 = np.zeros(shape=(number_of_items,number_of_users))
	for i in training_data6:
		collab_matrix6[i[0]][i[1]] = i[2]
	C,U,R=matrixFuncs.CUR(collab_matrix6,1000,False)
	CURMat = (np.matmul((np.matmul(C,U)),R))
	np.save("collab_matrixCUR90.npy",CURMat)
	
if __name__ == '__main__':
	main()
