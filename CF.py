'''
	The objetive of this program is to implement collaborative filtering(Item-Item)
	This makes use of 3 functions
		main:
			Set the values for the top correlations and obtain the training and test data for the testing program
			Note: this functionality is not used for the final implementation
		normalise_collab_matrix:
			This normalizes the rating for every item, so that no item has unusually low or high ratings
			It then finds the cosine similarity and then for the top K [Hardcoded] items, it finds the weighed mean for the rating
		root_mean_square_error:
			It finds the root mean square error with respect to the test data
'''
from CF_Data import training_data,test_data,number_of_users,number_of_items,top_k,data
import numpy as np
from scipy.spatial.distance import cosine
import math, heapq

def normalise_collab_matrix(data,number_of_items,number_of_users,top_k,collab_matrix):
	''' This function first normalizes the rating for each item and then does a prediction based on the cosine similarity of the K closest
		raters. Note:K is a hardcoded value. 
		Input:
			data:The ratings corrosponding to every user and item within the dataset.
			number_of_items:The number of items
			number_of_users:The number of users
			top_k:The top k values used for cosine similarity
			collab_matrix:The matrix to be used for predictions
		Output:   
			collab_matrix:The matrix containing all the predictions	
	'''
	average = []
	for i in range(0,number_of_items):
		sum = 0
		num = 0
		for j in range(0,number_of_users):
			if collab_matrix[i][j] != 0:
				sum += collab_matrix[i][j]
				num+=1
		if num != 0:
			average.append(sum/num)
		else:
			average.append(3.0);
		for j in range(0,number_of_users):
			if collab_matrix[i][j] != 0:
				collab_matrix[i][j] -= average[-1]
	print("DONE")

	print("Finished finding the averages");
	#predicting using cosine similarity
	score_holder = []
	for i in range(0,number_of_items):
		cosine_similarity_scores = []
		weighed_sum = 0
		sum_of_weights = 0
		for j in range(0, number_of_items):
			if i != j:
				cosine=cosine(collab_matrix[i],collab_matrix[j])
				if cosine > 0 and len(cosine_similarity_scores) < top_k:
					heapq.heappush(cosine_similarity_scores,(cosine,j))
				elif len(cosine_similarity_scores) >= top_k and cosine_similarity_scores[0][0] < cosine:
					heapq.heappop(cosine_similarity_scores)
					heapq.heappush(cosine_similarity_scores,(cosine,j))
		for k in range(0,number_of_users):
			if collab_matrix[i][k]==0:
				for j in range(0,len(cosine_similarity_scores)):
					weight=cosine_similarity_scores[j][0]
					item=cosine_similarity_scores[j][1]
					weighed_sum+=weight*data[item][k]
					sum_of_weights+=weight
				if sum_of_weights==0:
					collab_matrix[i][k]=0
				else:
					collab_matrix[i][k]=weighed_sum/sum_of_weights+average[i]

	print(collab_matrix)
	# np.save("collab2.npy",collab_matrix)
	return collab_matrix

def root_mean_square_error(test_data,collab_matrix,orig_matrix):
	''' This function finds the error of the predictions of the algorithm, with respect to the training data
		Input:
			test_data:The data which has been used to test the predictions of the algorithm
			collab_matrix: The matrix containing all the predictions
			orig_matix: The matrix containing the original data
		Output:
			rmse: The root mean square error
	'''
	squared_sum = 0
	non_zero = 0
	for pair in test_data:
		print("collab_matrix had %f, test_data had %d"%(collab_matrix[pair[0]][pair[1]],orig_matrix[pair[0]][pair[1]]) )
		if(orig_matrix[pair[0]][pair[1]] != 0):
			squared_sum += math.pow( (orig_matrix[pair[0]][pair[1]] - collab_matrix[pair[0]][pair[1]]),2 )
			non_zero+=1
	rmse = math.sqrt(squared_sum/non_zero)
	return rmse

def main():
	k = 3
	# test_data = CF_Data.test_data
	# training_data = CF_Data.training_data
	# print(training_data)
	# print()
	# print(test_data)

	#filling collab_matrix
	collab_matrix = np.zeros(shape=(number_of_users,number_of_items))
	for i in training_data:
		collab_matrix[i[0]][i[1]] = i[2]

	print(collab_matrix)
	collab_matrix = normalise_collab_matrix(data,number_of_users,number_of_items,top_k,collab_matrix)
	print(root_mean_square_error(test_data,collab_matrix))



if __name__=='__main__':
	main()
