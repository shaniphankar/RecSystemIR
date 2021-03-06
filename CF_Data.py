'''
	This program contains test data for CF.py and CF_Baseline.py for testing to check the performace and the correctness of the outputs
		1. In the start, the number of users and the number of items have been hardcoded along with the corresponding data which has been pulled from the class slides
		2. This is followed by the percentage of test data to the training data.
		3. We then generate random numbers to differentiate the test data from the random data.
		4. After doing so the program then takes the corresponding data as the input.

	WARNING:
	Use random permutations as it has a smaller computation overhead, this is precisely which is done in the main program.
'''
import numpy as np
number_of_users=12
number_of_items=6
top_k = 3
data=np.array([[1,0,2,0,0,1],[0,0,4,2,0,0],[3,5,0,4,4,3],[0,4,1,0,3,0],[0,0,2,5,4,3],[5,0,0,0,2,0],[0,4,3,0,0,0],[0,0,0,4,0,2],[5,0,4,0,0,0],[0,2,3,0,0,0],[4,1,5,2,2,4],[0,3,0,0,5,0]])
percentage = 0.7

test_data = []
n = 0
selected_data = {}
training_data = []
while n/(number_of_items*number_of_users) < 0.7:
	i = np.random.randint(number_of_users)
	j = np.random.randint(number_of_items)
	if selected_data.get((i,j)):
		continue
	else:
		selected_data[(i,j)] = True
		training_data.append((i,j,data[i][j]))
		n+=1

for i in range(0,number_of_users):
	for j in range(0,number_of_items):
		if (i,j) in selected_data.keys():
			continue
		else:
			selected_data[(i,j)] = True
			test_data+=[(i,j)]

# print(training_data)
# print()
# print(test_data)
