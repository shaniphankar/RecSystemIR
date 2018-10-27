import numpy as np
import pprint
import os
from random import randint
import matrixFuncs
import CF
import svdModified

def sample(number_of_items,number_of_users,data):
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
	# while n/(number_of_items*number_of_users) < 0.7:
	# 	i = np.random.randint(number_of_items)
	# 	j = np.random.randint(number_of_users)
	# 	if selected_data.get((i,j)):
	# 		continue
	# 	else:
	# 		selected_data[(i,j)] = True
	# 		training_data.append((i,j,data[i][j]))
	# 		n+=1
	# 	print((i,j))
	# for i in range(0,number_of_items):
	# 	for j in range(0,number_of_users):
	# 		if (i,j) in selected_data.keys():
	# 			continue
	# 		else:
	# 			selected_data[(i,j)] = True
	# 			test_data+=[(i,j)]
	# 	print((i,j))
	return training_data,test_data

def main():
	pp=pprint.PrettyPrinter(indent=4)
	mID_names={}
	# f=open(os.getcwd()+'/ml-1m/movies.dat',encoding='latin-1')
	# for line in f:
	# 	mID_names[line.split('::')[0]]=line.split('::')[1]
	# pp.pprint(mID_names)
	# f.close()
	number_of_items=3952
	number_of_users=6040
	top_k=10
	mID_uID_rating=np.zeros(shape=(3952,6040))
	f=open(os.getcwd()+'/ml-1m/ratings.dat',encoding='latin-1')
	for line in f:
		mID_uID_rating[int(line.split('::')[1])-1][int(line.split('::')[0])-1]=line.split('::')[2]
	# pp.pprint(mID_uID_rating)
	f.close()
	# C,U,R=matrixFuncs.CUR(mID_uID_rating,1000)
	# pp.pprint(mID_uID_rating)
	# pp.pprint((np.matmul((np.matmul(C,U)),R)))
	# CURMat = (np.matmul((np.matmul(C,U)),R))
	# print(CURMat)
	# print("1**************")
	# training_data1,test_data1=sample(3952,6040,mID_uID_rating)
	# collab_matrix1 = np.zeros(shape=(number_of_items,number_of_users))
	# for i in training_data1:
	# 	collab_matrix1[i[0]][i[1]] = i[2]
	# np.save("test_dataCF.npy",test_data1)
	# collab_matrix1 = CF.normalise_collab_matrix(mID_uID_rating,number_of_items,number_of_users,top_k,collab_matrix1)
	# np.save("collab_matrixCF.npy",collab_matrix1)
	# # print("2**************")
	# # For baseline, not done yet.
	# # np.save("test_data1.npy",test_data)
	# # collab_matrix1 = CF.normalise_collab_matrix(mID_uID_rating,number_of_items,number_of_users,top_k,collab_matrix1)
	# # np.save("collab_matrixCF.npy",collab_matrix1)
	# print("3**************")
	# training_data3,test_data3=sample(3952,6040,mID_uID_rating)
	# np.save("test_dataSVD.npy",test_data3)
	# collab_matrix3 = np.zeros(shape=(number_of_items,number_of_users))
	# for i in training_data3:
	# 	collab_matrix3[i[0]][i[1]] = i[2]
	# U,Sig,V=svdModified.SVD1(collab_matrix3,False)
	# SVDMat=np.matmul(np.matmul(U,Sig),V.transpose())
	# np.save("collab_matrixSVD.npy",SVDMat)
	# print("4**************")
	# training_data4,test_data4=sample(3952,6040,mID_uID_rating)
	# np.save("test_dataSVD90.npy",test_data4)
	# collab_matrix4 = np.zeros(shape=(number_of_items,number_of_users))
	# for i in training_data4:
	# 	collab_matrix4[i[0]][i[1]] = i[2]
	# U,Sig,V=svdModified.SVD1(collab_matrix4,True)
	# SVDMat=np.matmul(np.matmul(U,Sig),V.transpose())
	# np.save("collab_matrixSVD90.npy",SVDMat)
	# print("5**************")
	# training_data5,test_data5=sample(3952,6040,mID_uID_rating)
	# np.save("test_dataCUR.npy",test_data5)
	# collab_matrix5 = np.zeros(shape=(number_of_items,number_of_users))
	# for i in training_data5:
	# 	collab_matrix5[i[0]][i[1]] = i[2]
	# C,U,R=matrixFuncs.CUR(collab_matrix5,1000,True)
	# CURMat = (np.matmul((np.matmul(C,U)),R))
	# np.save("collab_matrixCUR.npy",CURMat)
	print("6**************")
	training_data6,test_data6=sample(3952,6040,mID_uID_rating)
	np.save("test_dataCUR90.npy",test_data6)
	collab_matrix6 = np.zeros(shape=(number_of_items,number_of_users))
	for i in training_data6:
		collab_matrix6[i[0]][i[1]] = i[2]
	C,U,R=matrixFuncs.CUR(collab_matrix6,1000,False)
	CURMat = (np.matmul((np.matmul(C,U)),R))
	np.save("collab_matrixCUR90.npy",CURMat)
	
	# print(CF.root_mean_square_error(test_data,collab_matrix,mID_uID_rating))

if __name__ == '__main__':
	main()
