from CF_Data import training_data,test_data,number_of_users,number_of_items,top_k,data
import numpy as np
import math, heapq
def normalise_collab_matrix(top_k,collab_matrix):
	average = []
	for i in range(0,number_of_items):
		sum = 0
		num = 0
		for j in range(0,number_of_users):
			if collab_matrix[j][i] != 0:
				sum += collab_matrix[j][i]
				num+=1
		if num != 0:
			average.append(sum/num)
		else:
			average.append(3.0);
		for j in range(0,number_of_users):
			if collab_matrix[j][i] != 0:
				collab_matrix[j][i] -= average[-1]

	#predicting using cosine similarity
	score_holder = []
	for i in range(0,number_of_items):
		cosine_similarity_scores = []
		weighed_sum = 0
		sum_of_weights = 0
		for j in range(0, number_of_items):
			if i != j:
				dot_product = 0
				for k in range(0,number_of_users):
					dot_product += collab_matrix[k][i] * collab_matrix[k][j]
				if dot_product > 0 and len(cosine_similarity_scores) < top_k:
					heapq.heappush(cosine_similarity_scores,(dot_product,j))
				elif len(cosine_similarity_scores) >= top_k and cosine_similarity_scores[0][0] < dot_product:
					heapq.heappop(cosine_similarity_scores)
					heapq.heappush(cosine_similarity_scores,(dot_product,j))
		for k in range(0,number_of_users):
			if collab_matrix[k][i]==0:
				for j in range(0,len(cosine_similarity_scores)):
					weight=cosine_similarity_scores[j][0]
					item=cosine_similarity_scores[j][1]
					weighed_sum+=weight*data[k][item]
					sum_of_weights+=weight
				if sum_of_weights==0:
					collab_matrix[k][i]=0
				else:
					collab_matrix[k][i]=weighed_sum/sum_of_weights

	print(collab_matrix)
	return collab_matrix

def root_mean_square_error(test_data,collab_matrix):
	squared_sum = 0
	for pair in test_data:
		print("collab_matrix had %f, test_data had %d"%(collab_matrix[pair[0]][pair[1]],data[pair[0]][pair[1]]) )
		squared_sum += math.pow( (data[pair[0]][pair[1]] - collab_matrix[pair[0]][pair[1]]),2 )
	rmse = math.sqrt(squared_sum/len(test_data))
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

	
	collab_matrix = normalise_collab_matrix(top_k,collab_matrix)
	print(root_mean_square_error(test_data,collab_matrix))
	# print(collab_matrix)



if __name__=='__main__':
	main()