#importing the required libraries
import numpy as np
import pprint
import os
from random import randint
import math

# Loading the dataset
mID_uID_rating=np.zeros(shape=(3952,6040))
f=open(os.getcwd()+'/ml-1m/ratings.dat',encoding='latin-1')
for line in f:
  mID_uID_rating[int(line.split('::')[1])-1][int(line.split('::')[0])-1]=line.split('::')[2]
f.close()

# Definition of the Root Mean Square Error
# We calculate the squared error for every pair in the test data and then take the square root of it
def root_mean_square_error(test_data,collab_matrix,orig_matrix):
	squared_sum = 0
	non_zero = 0
	for pair in test_data:
		# print("collab_matrix had %f, test_data had %d"%(collab_matrix[pair[0]][pair[1]],orig_matrix[pair[0]][pair[1]]) )
		if(orig_matrix[pair[0]][pair[1]] != 0):
			squared_sum += math.pow( (orig_matrix[pair[0]][pair[1]] - collab_matrix[pair[0]][pair[1]]),2 )
			non_zero+=1
	rmse = math.sqrt(squared_sum/non_zero)
	return rmse

# Definition the Spearman correlation coefficient, it tells us about the statistical dependence between the rankings of two variables.
def spearman_correlation_coefficient(test_data, collab_matrix,orig_matrix):
	test_orig_matrix = []
	test_collab_matrix = []
	index = 0;
	for pair in test_data:
		if(orig_matrix[pair[0]][pair[1]] != 0):
			test_orig_matrix.append((index,orig_matrix[pair[0]][pair[1]]))
			test_collab_matrix.append((index,collab_matrix[pair[0]][pair[1]]))
			index +=1
	test_orig_matrix = sorted(test_orig_matrix,key=lambda x: x[1],reverse=True)
	test_collab_matrix = sorted(test_collab_matrix,key=lambda x: x[1],reverse=True)
	rank_orig = []
	rank_collab = []
	#print((test_orig_matrix))
	#print(len(test_collab_matrix))
	for i in range(len(test_orig_matrix)):
		rank_orig.append((i,test_orig_matrix[i][0]))
		rank_collab.append((i,test_collab_matrix[i][0]))
	#print(rank_orig)
	dict1 = dict(rank_orig)
	dict2 = dict(rank_collab)
	#print(dict1)
	a = [dict1[key] - dict2.get(key, 0) for key in dict1.keys()]
	#print(a)
	sum = 0
	for i in a:
		sum += a[i]**2
	n = len(a)
	rho = 1- ((6*sum)/((n)*((n**2) - 1)))
	return(rho)
	#print(dict2)

# Definition of the TopK precision, where K is a variable.
# It tells us about the percentage of relevant documents in the top results returned by the system.
def precision_topk(test_data, collab_matrix,orig_matrix,k,threshold):
	store = []
	for pair in test_data:
		# print("collab_matrix had %f, test_data had %d"%(collab_matrix[pair[0]][pair[1]],orig_matrix[pair[0]][pair[1]]) )
		if(orig[pair[0]][pair[1]] != 0):
			store.append((collab_matrix[pair[0]][pair[1]],orig_matrix[pair[0]][pair[1]]))
	store = sorted(store,key=lambda x: x[0],reverse=True)[:k]
	count = 0
	for i in store:
		if(i[1]>threshold):
			count+=1
	print(count/k)
	return store

def main():

	# Loading the saved data
	test_data = np.load("test_dataCUR.npy")
	collab_matrix = np.load("collab_matrixCUR.npy")

	rmse = root_mean_square_error(test_data,collab_matrix,mID_uID_rating)
	print(rmse)

	rho = spearman_correlation_coefficient(test_data,collab_matrix,mID_uID_rating)
	print(rho)
	
	pk = precision_topk(test_data,collab_matrix,mID_uID_rating,10,2.5)
	print(pk)

if __name__ == '__main__':
	main()