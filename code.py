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
	print(mID_uID_rating.shape)
	# C,U,R=matrixFuncs.CUR(mID_uID_rating,1000)
	# pp.pprint(mID_uID_rating)
	# pp.pprint((np.matmul((np.matmul(C,U)),R)))
	# CURMat = (np.matmul((np.matmul(C,U)),R))
	# print(CURMat)
	U,Sig,V=svdModified.SVD1(mID_uID_rating)
	SVDMat=np.matmul(np.matmul(U,Sig),V.transpose())
	print("SVD")
	print(SVDMat)
	training_data,test_data=sample(3952,6040,SVDMat)
	# print("Training Data")
	# pp.pprint(training_data)
	print("testing Data")
	for x in test_data:
		pp.pprint(SVDMat[x[0]][x[1]])	
	# np.save("test_data.npy",test_data)
	# collab_matrix = np.zeros(shape=(number_of_items,number_of_users))
	# for i in training_data:
	# 	collab_matrix[i[0]][i[1]] = i[2]
	# collab_matrix = CF.normalise_collab_matrix(SVDMat,number_of_items,number_of_users,top_k,collab_matrix)
	# print(CF.root_mean_square_error(test_data,collab_matrix,SVDMat))
	# collab_matrix = CF.normalise_collab_matrix(mID_uID_rating,number_of_items,number_of_users,top_k,collab_matrix,test_data)
	# print(CF.root_mean_square_error(test_data,collab_matrix,mID_uID_rating))

if __name__ == '__main__':
	main()
