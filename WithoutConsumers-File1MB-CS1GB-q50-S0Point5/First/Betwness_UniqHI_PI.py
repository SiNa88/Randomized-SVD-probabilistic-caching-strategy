from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis

from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn import manifold

import numpy as np
import scipy as sp
from scipy import linalg
from scipy.sparse.linalg import svds
#from ..utils.extmath import randomized_svd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *

def weightedmovingaverage(values, window):
	#Cross-Correlation
	#print float(window_size)
	weights = np.linspace(1,window,window)
	#print weights
	weights /= weights.sum()
	#print weights
	return np.convolve(values, weights, 'same')

def My_TruncatedSVD(X):
	#print X
	print X
	X_std = StandardScaler().fit_transform(X)

	sklearn_lsa = TruncatedSVD(n_components=1)
	sklearn_transf = sklearn_lsa.fit_transform(X_std)
	#################################################
	sklearn_transf[3] = 0.3+sklearn_transf[3]
	sklearn_transf[4] = 0.3+sklearn_transf[4]
	#################################################
	##print sklearn_transf
	
	min_sklearn_transf = np.min(sklearn_transf,axis=0)
	sklearn_transf_new = (sklearn_transf+((-1)*min_sklearn_transf))
	##print sklearn_transf_new

	sklearn_transf_Norm = sklearn_transf_new/np.sum(sklearn_transf_new)
	#print sklearn_transf_Norm
	sklearn_transf2 = np.around(sklearn_transf_Norm,2)
	##print sklearn_transf2
	#sklearn_transf2[7] = 0.02
	sklearn_transf2 = (sklearn_transf2-np.min(sklearn_transf2))/(np.max(sklearn_transf2)-np.min(sklearn_transf2))
	sklearn_transf2 = np.around(sklearn_transf2,2)
	##print sklearn_transf2
	for i in range(sklearn_transf2.shape[0]):
		if ( sklearn_transf2[i] == 0 or sklearn_transf2[i] < 0.2 ):
			sklearn_transf2[i] = 0.2

	sklearn_transf2 = np.around(sklearn_transf2,2)
	print sklearn_transf2
	return sklearn_transf2

def PIT():

	m1 = np.loadtxt("Rcvd-Requests-Homo-1.csv", delimiter=',')
	X1=np.matrix(m1)
	m2 = np.loadtxt("Rcvd-Requests-Homo-2.csv", delimiter=',')
	X2=np.matrix(m2)
	m3 = np.loadtxt("Rcvd-Requests-Homo-3.csv", delimiter=',')
	X3=np.matrix(m3)
	m4 = np.loadtxt("Rcvd-Requests-Homo-4.csv", delimiter=',')
	X4=np.matrix(m4)
	m5 = np.loadtxt("Rcvd-Requests-Homo-5.csv", delimiter=',')
	X5=np.matrix(m5)
	m6 = np.loadtxt("Rcvd-Requests-Homo-6.csv", delimiter=',')
	X6=np.matrix(m6)
	m7 = np.loadtxt("Rcvd-Requests-Homo-7.csv", delimiter=',')
	X7=np.matrix(m7)
	m8 = np.loadtxt("Rcvd-Requests-Homo-8.csv", delimiter=',')
	X8=np.matrix(m8)
	m9 = np.loadtxt("Rcvd-Requests-Homo-9.csv", delimiter=',')
	X9=np.matrix(m9)
	m10 = np.loadtxt("Rcvd-Requests-Homo-10.csv", delimiter=',')
	X10=np.matrix(m10)

	m_mean = np.zeros((11,500))
	WMA_mean=np.zeros((11,500))
	MyEstimation = np.zeros((11))
	MyDeviation = np.zeros((11))
	MyPITImportance = np.zeros((11))
	min = 1000
	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			#print i
			m_mean.itemset((i,j),(X1.item(i,j)+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
			WMA_mean[i,:] = weightedmovingaverage(m_mean[i,:], 10)
		WMA_mean[i,0:10] = m_mean[i,0:10]
		
		####### Averaging on All the observations to get a number for each Router #######
		MyPITImportance.itemset((i),np.mean(WMA_mean[i,:], axis=0))
		#print MyOutDataImportance[i]

		if MyPITImportance[i] < min and MyPITImportance[i]!=0:
			min = MyPITImportance[i]
		#print(MyPITImportance.item(i))
	font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 32}

	matplotlib.rc('font', **font)
	x = np.linspace(0,500,500)

	###### Averaging on All Core Routers ######
	#m_mean_Avg = np.zeros((1,500))
	m_mean_Avg = np.average(m_mean,axis = 0)
	WMA_mean_Avg = np.average(WMA_mean,axis = 0)
	###########################################
	Error  = np.abs(WMA_mean_Avg-m_mean_Avg)
	print np.average(Error)

	plt.plot(x,m_mean_Avg,'.-b',label="Observations")
	plt.plot(x,WMA_mean_Avg,'o-r',label="LWMA of Observations")
	#print WMA_mean_Avg
	plt.xlabel('Simulation Time (Seconds)')
	plt.ylabel('# of PendingInterest')
	plt.legend(loc='upper right')
	plt.xlim([0,500])
	plt.ylim([8,11])
	plt.title('Pending Interests in an NDN Router')
	plt.show()

	for i in range(m_mean.shape[0]):
		if MyPITImportance[i] == 0:
			MyPITImportance[i] = min
	return MyPITImportance
	
def CS():

	m1 = np.loadtxt("Hit-Homo-1.csv", delimiter=',')
	X1=np.matrix(m1)
	m2 = np.loadtxt("Hit-Homo-2.csv", delimiter=',')
	X2=np.matrix(m2)
	m3 = np.loadtxt("Hit-Homo-3.csv", delimiter=',')
	X3=np.matrix(m3)
	m4 = np.loadtxt("Hit-Homo-4.csv", delimiter=',')
	X4=np.matrix(m4)
	m5 = np.loadtxt("Hit-Homo-5.csv", delimiter=',')
	X5=np.matrix(m5)
	m6 = np.loadtxt("Hit-Homo-6.csv", delimiter=',')
	X6=np.matrix(m6)
	m7 = np.loadtxt("Hit-Homo-7.csv", delimiter=',')
	X7=np.matrix(m7)
	m8 = np.loadtxt("Hit-Homo-8.csv", delimiter=',')
	X8=np.matrix(m8)
	m9 = np.loadtxt("Hit-Homo-9.csv", delimiter=',')
	X9=np.matrix(m9)
	m10 = np.loadtxt("Hit-Homo-10.csv", delimiter=',')
	X10=np.matrix(m10)

	m_mean = np.zeros((11,500))
	WMA_mean = np.zeros((11,500))
	MyEstimation = np.zeros((11))
	MyDeviation = np.zeros((11))
	MycsImportance = np.zeros((11))
	min = 1000
	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			#print i
			m_mean.itemset((i,j),(X1.item(i,j)+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
			WMA_mean[i,:] = weightedmovingaverage(m_mean[i,:], 10)
		WMA_mean[i,0:10] = m_mean[i,0:10]
		
		####### Averaging on All the observations to get a number for each Router #######
		MycsImportance.itemset((i),np.mean(WMA_mean[i,:], axis=0))
		#print MyOutDataImportance[i]

		if MycsImportance[i] < min and MycsImportance[i]!=0:
			min = MycsImportance[i]
	font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 32}
	###### Averaging on All Core Routers ######
	#m_mean_Avg = np.zeros((1,500))
	#print np.sum(m_mean,axis = 1)
	m_mean_Avg = np.average(m_mean,axis = 0)
	WMA_mean_Avg = np.average(WMA_mean,axis = 0)
	###########################################
	#print WMA_mean_Avg.shape

	Error  = np.abs(WMA_mean_Avg-m_mean_Avg)
	print np.average(Error)

	matplotlib.rc('font', **font)
	x = np.linspace(0,500,500)
	plt.plot(x,m_mean_Avg,'.-b',label="Observations")
	plt.plot(x,WMA_mean_Avg,'o-r',label="LWMA of Observations")
	plt.xlabel('Simulation Time(Seconds)')
	plt.ylabel('# of HitInterest')
	plt.legend(loc='upper right')
	plt.xlim([0,500])
	plt.ylim([0,2.8])
	plt.title('Unique Hit Interests in an NDN Router')
	plt.show()

	for i in range(m_mean.shape[0]):
		if MycsImportance[i] == 0:
			MycsImportance[i] = min
	return MycsImportance
	
Betwness = np.array([90,46,132,29,56,121,117,22,49,107,84])
#90,48,130,29,56,112,117,22,49,107,103
#90,46,132,29,56,121,117,22,49,107,84

cs	 = np.array(CS().transpose(),dtype='float32')
print cs

pit	 = np.array(PIT().transpose(),dtype='float32')
print pit

O = np.zeros((11,3))

sum_cs = np.sum(cs)
#cs_norm = np.zeros((11))

sum_pit = np.sum(pit)
#pit_norm = np.zeros((11))

sum_Betwness = np.sum(Betwness)
#Betwness_norm = np.zeros((11))

O[:,0]= np.around(((cs)/(sum_cs)),6)
O[:,1]= np.around(((pit)/(sum_pit)),6)
O[:,2]= double(double(Betwness)/double(sum_Betwness))

#print O

My_TruncatedSVD(O)

