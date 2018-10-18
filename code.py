import numpy as np
import pprint
import os
from random import randint
import matrixFuncs

def main():
	pp=pprint.PrettyPrinter(indent=4)
	mID_names={}
	# f=open(os.getcwd()+'/ml-1m/movies.dat',encoding='latin-1')
	# for line in f:
	# 	mID_names[line.split('::')[0]]=line.split('::')[1]
	# pp.pprint(mID_names)
	# f.close()
	mID_uID_rating=np.zeros(shape=(3952,6040))
	f=open(os.getcwd()+'/ml-1m/ratings.dat',encoding='latin-1')
	for line in f:
		mID_uID_rating[int(line.split('::')[1])-1][int(line.split('::')[0])-1]=line.split('::')[2]
	# pp.pprint(mID_uID_rating)
	f.close()
	C,U,R=matrixFuncs.CUR(mID_uID_rating,100)
	# pp.pprint(mID_uID_rating)
	# pp.pprint((np.matmul((np.matmul(C,U)),R)))
	# print(data)

if __name__ == '__main__':
	main()