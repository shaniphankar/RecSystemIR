import numpy as np, heapq, math
from scipy import spatial
from scipy.spatial.distance import cosine
# from CF_Data import training_data,test_data,number_of_users,number_of_items,top_k,data

def find_cosine_similarity(number_of_items,number_of_users,i,j,collab_matrix):
	''' This function finds the cosine similarity between two vectors
		WARNING: This function is deprecated as we have made use of np.dot and will be removed in subsequent versions
		Input:
			number_of_items:The number of items
			number_of_items:The number of users
			i:The first vector
			j:The secod vector
			collab_matrix: The matrix which contains the normalised values and the predictions
		Output:
		cosinesine_score: The cosine score between vectors i and j
	'''
	sum1 = 0
	sum2 = 0
	cosine_score = 0
	for k in range(0,number_of_users):
		cosine_score += collab_matrix[i][k] * collab_matrix[j][k]
	for k in range(0,number_of_users):
		sum1 += math.pow(collab_matrix[i][k],2)
		sum2 += math.pow(collab_matrix[j][k],2)
	if sum1 == 0 or sum2 == 0:
		cosine_score = 0
	else:
		sum1 = math.sqrt(sum1)
		sum2 = math.sqrt(sum2)
		cosine_score /= (sum1*sum2)
	return cosine_score

def get_prediction(data,number_of_items,number_of_users,top_k,collab_matrix,averageItem,averageUser,overall_average):
	''' 
		This function helps fill the data in the prediction matrix with the help of the intput data. Here we find the 
		top k items based on the normalised cosine similarity and then find the rating by adjusting for the baseline and the 
		bias of each user.
		Input:
			data: The data for each user and the items that the user has used.
			number_of_items: The total number of items
			number_of_users: The total numbrer of users
			top_k: Hardcoded value to tell how many similar vectors to use in the cosine similarity
			collab_matrix: The normalised matrix that will contain the predictions
			averageItem: The average rating for each item
			averageUser: The average rating provided by each user
			overall_average: The average of all the ratings present within the training data
		Output:
			collab_matrix: This matrix contains the predictions based on the algorithm

	'''
	for i in range(0,number_of_items):
		for j in range(0,number_of_users):
			if collab_matrix[i][j] != 0:
				collab_matrix[i][j] -= averageItem[i] + overall_average

	#predicting using cosine similarity
	# print("############################################")
	# print(collab_matrix)
	# print("############################################")
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
		if len(cosine_similarity_scores) == 0:
			for j in range(0,number_of_users):
				collab_matrix[i][j] = 3
		else:
			# for k in range(0,number_of_users):
			for k in range(0,number_of_users):
				if collab_matrix[i][k]==0:
					sum_of_weights = 0
					weighed_sum = 0
					for j in range(0,len(cosine_similarity_scores)):
						weight = cosine_similarity_scores[j][0]
						item = cosine_similarity_scores[j][1]
						# print(i,k)s
						# print()
						# print(item,k)
						baseline_other = overall_average + averageUser[k] + averageItem[item]
						baseline_target = overall_average + averageUser[k] + averageItem[i]
						#print("baseline_target = %f\nbaseline_other = %f"%(baseline_target,baseline_other))
						weighed_sum += weight*(data[item][k] - baseline_other)
						sum_of_weights += weight
						# print("%f is weighed sum, %f is sumOfWeights"%(weighed_sum,sum_of_weights))
					if sum_of_weights == 0:
						collab_matrix[i][k] = 0
					else:
						weighed_sum += baseline_target
						val = weighed_sum/sum_of_weights
						if val>0 and val <=5 :
							collab_matrix[i][k] = val
						elif val>5 :
							collab_matrix[i][k] = 5
						else :
							collab_matrix[i][k] = 0


	print(collab_matrix)
	return collab_matrix

# def root_mean_square_error(test_data,collab_matrix):
# 	squared_sum = 0
# 	for pair in test_data:
# 		print("collab_matrix had %f, test_data had %d"%(collab_matrix[pair[0]][pair[1]],data[pair[0]][pair[1]]) )
# 		squared_sum += math.pow( (data[pair[0]][pair[1]] - collab_matrix[pair[0]][pair[1]]),2 )
# 	rmse = math.sqrt(squared_sum/len(test_data))
# 	return rmse

def findAverage(data,number_of_items,number_of_users,collab_matrix):
	'''
		This function finds the average of each user, each item adnd the overall average of all the ratings within 
		the dataset and will be used by get_prediction
		Input:
			data: The data for the user and the ratings given by the user.
			number_of_items: The total number of items
			number_of_users: The total number of users
			collab_matrix: The normalised matrix on which predictions will be obtained
		Output:
			averageItem: The average item rating for each item. It is a vector whose ith entry gives the rating for item i
			averageUser: The average user rating provided by the user. It is a vector whose ith entry gives the average rating given by user i
			overall_average: The average of all the ratings provided within the data
	'''
	temp = 0
	count = 0
	averageItem = []
	averageUser = []
	overall_average = 0
	n_new = 0
	sum_new = 0
	for i in range(0,number_of_items):
		for j in range(0,number_of_users):
			if data[i][j] != 0:
				sum_new += data[i][j]
				n_new += 1
			else:
				sum_new += 3
				n_new += 1

	if sum_new == 0:
		overall_average = 3
	else:
		overall_average = sum_new/n_new


	for i in range(0,number_of_items):
		col_sum = 0
		col_count = 0
		for j in range(0,number_of_users):
			if data[i][j] != 0 :
				temp += data[i][j]
				count += 1
				col_sum += data[i][j]
				col_count += 1
			else:
				col_sum += 3
				col_count += 1
		if col_count == 0 :
			averageItem.append(3 - overall_average)
		else :
			averageItem.append(col_sum/col_count - overall_average)
	for i in range(0,number_of_users):
		row_sum = 0
		row_count = 0
		for j in range(0,number_of_items):
			if data[j][i] != 0 :
				row_count+=1
				row_sum+=data[j][i]
			else:
				row_sum += 3
				row_count += 1
		if row_count == 0:
			averageUser.append(3 - overall_average)
		else :
			averageUser.append(row_sum/row_count - overall_average)

	print("overall_average = %f"%(overall_average))
	print(averageItem)
	print(averageUser)

	return averageItem,averageUser,overall_average

# def extract_data(training_data,data):
# 	collab_matrix = np.zeros(shape=(number_of_users,number_of_items))
# 	for i in training_data:
# 		collab_matrix[i[0]][i[1]] = data[i[0]][i[1]]
# 	return collab_matrix

def compute(data,number_of_items,number_of_users,top_k,collab_matrix):
	'''
		This function calls the other functions required to perform the algorithm
		Input:
			data: The data of the ratings for each user
			number_of_items: The total number of items
			number_of_users: The total number of users
			top_k: Hardcoded value to tell how many similar vectors to use in the cosine similarity
			collab_matrix: The normalised matrix that will contain the predictions

	'''
	averageItem,averageUser,overall_average = findAverage(data,number_of_items,number_of_users,collab_matrix)
	collab_matrix = get_prediction(data,number_of_items,number_of_users,top_k,collab_matrix,averageItem,averageUser,overall_average)
	return(collab_matrix)
	# print(root_mean_square_error(test_data,collab_matrix))

def main():
	# Code stub
	'''
		Only for testing should take input from CF_Data.py and then run the same functions as compute
	'''
if __name__=='__main__':
	main()