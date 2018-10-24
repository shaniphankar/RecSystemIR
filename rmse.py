import numpy as np
import pprint
import os
from random import randint
import math

test_data = np.load("test_data.npy")
collab_matrix = np.load("collab2.npy")

mID_uID_rating=np.zeros(shape=(3952,6040))
f=open(os.getcwd()+'/ml-1m/ratings.dat',encoding='latin-1')
for line in f:
  mID_uID_rating[int(line.split('::')[1])-1][int(line.split('::')[0])-1]=line.split('::')[2]
# pp.pprint(mID_uID_rating)
f.close()

def root_mean_square_error(test_data,collab_matrix,orig_matrix):
	squared_sum = 0
	non_zero = 0
	for pair in test_data:
		print("collab_matrix had %f, test_data had %d"%(collab_matrix[pair[0]][pair[1]],orig_matrix[pair[0]][pair[1]]) )
		if(orig_matrix[pair[0]][pair[1]] != 0):
			squared_sum += math.pow( (orig_matrix[pair[0]][pair[1]] - collab_matrix[pair[0]][pair[1]]),2 )
			non_zero+=1
	rmse = math.sqrt(squared_sum/non_zero)
	return rmse

rmse = root_mean_square_error(test_data,collab_matrix,mID_uID_rating)
print(rmse)
