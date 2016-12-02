#!/usr/bin/python

from math import *
import random
import sys
import re
import numpy as np
from collections import defaultdict
from random import randint
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
import pylab as P

def getNums1(filename):
	real = []
	fake = []
	with open(filename, 'r') as myfile:
		for l in myfile:
			sLine = l.strip().split()
			real.append(float(sLine[0]))
			fake.append(float(sLine[1]))
	
	return (real, fake)


def plotCooccur_avg():
	real,fake = getNums1("avg_cooccur.txt")

	data = [real,fake]
	n, bins, patches = P.hist(data, 30, normed=False, histtype='bar',
                            color=['chartreuse', 'crimson'],
                            label=['Real', 'Fake'])

	P.ylabel('Article Count')
	P.xlabel('Log-Probability')
	P.legend()
	P.show()

def plotCooccur_std():
	real,fake = getNums1("std_cooccur.txt")

	data = [real,fake]
	n, bins, patches = P.hist(data, 30, normed=False, histtype='bar',
                            color=['chartreuse', 'crimson'],
                            label=['Real', 'Fake'])

	P.ylabel('Article Count')
	P.xlabel('Standard Deviation of Log-Probability of Sentences in an Article')
	P.legend()
	P.show()	


if __name__ == '__main__':
	# plotCooccur_avg()
	plotCooccur_std()
	
