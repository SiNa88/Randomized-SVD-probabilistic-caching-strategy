import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.cm as cm
from pylab import *
from matplotlib.font_manager import FontProperties


def MyStem():
	m1 = np.loadtxt("Rcvd-Requests-NonHomo-1.csv", delimiter=',')
	X1=np.matrix(m1)
	m2 = np.loadtxt("Rcvd-Requests-NonHomo-2.csv", delimiter=',')
	X2=np.matrix(m2)
	m3 = np.loadtxt("Rcvd-Requests-NonHomo-3.csv", delimiter=',')
	X3=np.matrix(m3)
	m4 = np.loadtxt("Rcvd-Requests-NonHomo-4.csv", delimiter=',')
	X4=np.matrix(m4)
	m5 = np.loadtxt("Rcvd-Requests-NonHomo-5.csv", delimiter=',')
	X5=np.matrix(m5)
	m6 = np.loadtxt("Rcvd-Requests-NonHomo-6.csv", delimiter=',')
	X6=np.matrix(m6)
	m7 = np.loadtxt("Rcvd-Requests-NonHomo-7.csv", delimiter=',')
	X7=np.matrix(m7)
	m8 = np.loadtxt("Rcvd-Requests-NonHomo-8.csv", delimiter=',')
	X8=np.matrix(m8)
	m9 = np.loadtxt("Rcvd-Requests-NonHomo-9.csv", delimiter=',')
	X9=np.matrix(m9)
	m10 = np.loadtxt("Rcvd-Requests-NonHomo-10.csv", delimiter=',')
	X10=np.matrix(m10)
	m_mean = np.zeros((11,500))

	m1Homo = np.loadtxt("Rcvd-Requests-Homo-1.csv", delimiter=',')
	X1Homo=np.matrix(m1Homo)
	m2Homo = np.loadtxt("Rcvd-Requests-Homo-2.csv", delimiter=',')
	X2Homo=np.matrix(m2Homo)
	m3Homo = np.loadtxt("Rcvd-Requests-Homo-3.csv", delimiter=',')
	X3Homo=np.matrix(m3Homo)
	m4Homo = np.loadtxt("Rcvd-Requests-Homo-4.csv", delimiter=',')
	X4Homo=np.matrix(m4Homo)
	m5Homo = np.loadtxt("Rcvd-Requests-Homo-5.csv", delimiter=',')
	X5Homo=np.matrix(m5Homo)
	m6Homo = np.loadtxt("Rcvd-Requests-Homo-6.csv", delimiter=',')
	X6Homo=np.matrix(m6Homo)
	m7Homo = np.loadtxt("Rcvd-Requests-Homo-7.csv", delimiter=',')
	X7Homo=np.matrix(m7Homo)
	m8Homo = np.loadtxt("Rcvd-Requests-Homo-8.csv", delimiter=',')
	X8Homo=np.matrix(m8Homo)
	m9Homo = np.loadtxt("Rcvd-Requests-Homo-9.csv", delimiter=',')
	X9Homo=np.matrix(m9Homo)
	m10Homo = np.loadtxt("Rcvd-Requests-Homo-10.csv", delimiter=',')
	X10Homo=np.matrix(m10Homo)
	m_meanHomo = np.zeros((11,500))
	
	
	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			m_mean.itemset((i,j),(X1.item(i,j)+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
			m_meanHomo.itemset((i,j),(X1Homo.item(i,j)+X2Homo.item(i,j)+X3Homo.item(i,j)+X4Homo.item(i,j)+X5Homo.item(i,j)+X6Homo.item(i,j)+X7Homo.item(i,j)+X8Homo.item(i,j)+X9Homo.item(i,j)+X10Homo.item(i,j))/10)

	font = {'family' : 'normal',
        	'weight' : 'bold',
        	'size'   : 27}

	matplotlib.rc('font', **font)
	x2 = np.linspace(0,11,11)

	AVG_Homo1 = np.zeros((1,500))	
	AVG_Homo1 = np.average(m_meanHomo[:,0:500], axis=0)
	
	AVG_NonHomo1 = np.zeros((1,500))
	AVG_NonHomo1 = np.average(m_mean[:,0:500], axis=0)
	
	print np.average(m_meanHomo[:,0:500], axis = 1)
	print np.average(m_mean[:,0:500], axis = 1)

	plt.plot(x2,np.average(m_meanHomo[:,0:500], axis = 1),'.-g',label="Ubiquitous")
	plt.plot(x2,np.average(m_mean[:,0:500], axis = 1),'^-r',label="ProposedMethod")
	plt.legend(loc='upper left')
	plt.title('Number of PIs in Averaged of Core Routers')
	plt.xlim([0,11.1])
	plt.ylim([9,171])
	plt.xlabel('Core Router ID')
	plt.ylabel('Number of Pending Interests')
	plt.show()

	AVG_NonHomo = np.zeros((1,500))	
	AVG_NonHomo = np.sum(m_mean, axis=0)/11	
	
	AVG_Homo = np.zeros((1,500))	
	AVG_Homo = np.sum(m_meanHomo, axis=0)/11

	j = 0
	tedaad = 5
	Smooth_NonHomo = np.zeros((tedaad,1))
	Smooth_Homo = np.zeros((tedaad,1))
	
	x_axis=np.linspace(0,tedaad,tedaad)
	
	

	for i in range(AVG_NonHomo.shape[0]):
		if ((j+1)*(500/tedaad)) <= i:
			j = j + 1
		Smooth_NonHomo[j] = Smooth_NonHomo[j] + AVG_NonHomo[i]
		Smooth_Homo[j] = Smooth_Homo[j] + AVG_Homo[i]
	for i in range(Smooth_NonHomo.shape[0]):
		Smooth_NonHomo[i] = Smooth_NonHomo[i] / (500/tedaad)
		Smooth_Homo[i] = Smooth_Homo[i] / (500/tedaad)
	print np.around(np.average(np.copy(Smooth_Homo)),6),"\n", np.around(np.average(np.copy(Smooth_NonHomo)),6),"\n"
	print Smooth_Homo,"\n"

	print Smooth_NonHomo,"\n"

	plt.plot(x_axis,Smooth_Homo,'.-c',label="Ubiquitious")
	plt.plot(x_axis,Smooth_NonHomo,'^-r',label="ProposedMethod")
	plt.xlim([-0.1,tedaad+0.1])
	plt.ylim([24.46,24.55])
	plt.xlabel('Time(*100 Seconds)')
	plt.ylabel('Number of Pending Requests')
	plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1),ncol=2, fancybox=True)
	plt.title('SnapShot of PITs')
	plt.show()
MyStem()
