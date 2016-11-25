
from __future__ import division
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import math


def classify():
	mat = scipy.io.loadmat('HW2/news.mat')
	d = mat['X_train'].shape[1]
	m = mat['X_train'].shape[0]
	# top = np.empty((m, d+m+1))
	left = np.empty((m, d))
	right = np.empty((m,1))
	dual_p = np.identity(m)
	dual_a = np.matrix(np.empty((1,m)))

	for i in range(m):
		x = mat['X_train'][i].toarray()
		y = mat['y_train'][0,i]
		y_np = np.array([np.negative(y)])
		left[i] = np.multiply(y_np,x)
		right[i] = np.asmatrix(y_np)
 		# top[i] = np.concatenate(( np.multiply(y_np,x), np.negative(np.ones((1,m))), np.asmatrix(y_np)), axis=1)
		dual_p[i,i] = np.asmatrix(y*y*np.dot(x,x.T))
		dual_a[0,i] = y

	mid = np.negative(np.identity(m))
	top = np.concatenate((left,mid,right),axis=1) 
	bottom = np.concatenate((np.zeros((m,d)), np.negative(np.identity(m)), np.zeros((m,1))), axis = 1)
	G_np = np.concatenate((top, bottom), axis=0)

	p_result, p_obj = primal(d, m, G_np)
	d_result, d_obj = dual(d, m, dual_p, dual_a)


	dual_betas = np.empty((d,1))
	dual_np = np.array(d_result)
	for i in range(dual_np.shape[0]):
		x = mat['X_train'][:,i].toarray()
		y = mat['y_train'][0,i]
		y_np = np.multiply(y,dual_np[i])
		dual_betas[i] = sum(np.dot(y_np,x))

	p_result_np = np.array(p_result)
	primal_betas = p_result_np[:d]
	primal_eps = p_result_np[d:d+m]
	primal_bias = p_result_np[d+m]

	np.savetxt('primal.txt',p_result_np)
	np.savetxt('dual.txt',dual_np)
	np.savetxt('dual_betas.txt',dual_betas)
	# plt.scatter(primal_betas,dual_betas)
	# plt.ylabel("Dual beta")
	# plt.xlabel("Primal beta")
	# plt.show()
	print("Written")

def getMetrics():

	primal = np.loadtxt('primal.txt')
	dual = np.loadtxt('dual.txt')
	dual_betas = np.loadtxt('dual_betas.txt')
	mat = scipy.io.loadmat('HW2/news.mat')
	d = mat['X_train'].shape[1]
	m = mat['X_train'].shape[0]

	primal_betas = primal[:d]
	primal_eps = primal[d:d+m]
	primal_bias = primal[d+m]

	dual_miss = 0
	primal_miss = 0
	for i in range(m):
		x = mat['X_train'][i].toarray()
		y = mat['y_train'][0,i]
		primal_pred = math.copysign(1,np.sum(np.dot(x, primal_betas), primal_bias))

		# dual_pred = math.copysign(1, np.sum(np.dot(dual_betas, x)[0],dual_bias))

		# if (dual_pred!=y):
		# 	dual_miss = dual_miss + 1

		if (primal_pred!=y):
			primal_miss = primal_miss + 1

	print primal_miss
	print "Primal misclassification rate: (Training)", (primal_miss/m)*100
	# print "Dual misclassification rate: (Training)", dual_miss/m
	
	dual_miss = 0
	primal_miss = 0
	l = mat['X_test'].shape[0]
	for i in range(l):
		x = mat['X_test'][i].toarray()
		y = mat['y_test'][0,i]
		primal_pred = math.copysign(1,np.sum(np.dot(x, primal_betas), primal_bias))
		# dual_pred = math.copysign(1, np.dot(dual_betas, x)[0] + dual_bias)

		# if (dual_pred!=y):
		# 	dual_miss = dual_miss + 1

		if (primal_pred!=y):
			primal_miss = primal_miss + 1

	print primal_miss
	print "Primal misclassification rate: (Test)", (primal_miss/l)*100
	# print "Dual misclassification rate: (Test)", dual_miss/m*100

	w_margin = 0
	misclass = 0
	chosen_dual_miss = []
	chosen_dual_within = []
	for i in range(m):
		if (primal_eps[i]>1):
			misclass = misclass + 1
			chosen_dual_miss.append(dual[i])
		elif (primal_eps[i]<=1 and primal_eps[i]>math.pow(10,-6)):
			w_margin = w_margin + 1
			chosen_dual_within.append(dual[i])

	print "Misclassified from training set: ", misclass
	print "Within margin: ", w_margin
	print "Dual variables for misclassified points: ", chosen_dual_miss
	print "Dual variables for points within margin: ", chosen_dual_within

	




def primal(d, m, G_np):

	G = matrix(G_np, tc ='d')

	top = np.concatenate((np.identity(d),np.zeros((d,m+1))), axis=1).astype(np.double)
	P_np = np.concatenate((top,np.zeros((m+1,d+m+1))), axis=0).astype(np.double)
	P = matrix(P_np, tc='d')
	print P.size

	q_np = np.concatenate((np.zeros((d,1)),np.ones((m,1)),np.zeros((1,1))), axis=0).astype(np.double)
	q = matrix(q_np, tc='d')
	print q.size

	H = matrix(np.concatenate((np.negative(np.ones((m,1))), np.zeros((m,1))), axis=0), tc='d')
	print H.size

	sol = solvers.qp(P, q, G, H)
	print(sol['x'])
	print(sol['primal objective'])

	return sol['x'], sol['primal objective']

def dual(d, m, P_np, A_np):

	q_np = np.negative(np.ones((m,1))).astype(np.double)
	q = matrix(q_np, tc='d')
	print q.size

	P = matrix(P_np, tc='d')

	A = matrix(A_np, tc='d')
	b = matrix(np.zeros((1,1)), tc = 'd')

	top = np.identity(m)
	bottom = np.negative(np.identity(m))
	G_np = np.concatenate((top,bottom), axis =0)
	G = matrix(G_np, tc = 'd')

	top = np.ones((m,1))
	bottom = np.zeros((m,1))
	H_np = np.concatenate((top,bottom), axis =0)
	H = matrix(H_np, tc = 'd')

	sol = solvers.qp(P, q, G, H, A, b)

	print(sol['x'])
	print(sol['primal objective'])

	return sol['x'], sol['primal objective']

if __name__ == '__main__':
	# classify()
	getMetrics()


