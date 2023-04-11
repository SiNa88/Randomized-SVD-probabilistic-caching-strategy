
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *
from matplotlib.font_manager import FontProperties, findfont

def ExpMovingAvg(values, window):
	weights = np.exp(np.linspace(-1.,0.,window))
	weights /= weights.sum()
	a = np.convolve(values,weights,mode='full') [:len(values)]
	return a
def MovingAvg(values, window):

	a = np.convolve(values,np.ones((window,))/window,mode='full') [:len(values)]
	return a

def CS():
	m1_Bet = np.loadtxt("Bet/AppDelay-Homo-1.csv", delimiter=',')
	X1_Bet=np.matrix(m1_Bet)
	m2_Bet = np.loadtxt("Bet/AppDelay-Homo-2.csv", delimiter=',')
	X2_Bet=np.matrix(m2_Bet)
	m3_Bet = np.loadtxt("Bet/AppDelay-Homo-3.csv", delimiter=',')
	X3_Bet=np.matrix(m3_Bet)
	m4_Bet = np.loadtxt("Bet/AppDelay-Homo-4.csv", delimiter=',')
	X4_Bet=np.matrix(m4_Bet)
	m5_Bet = np.loadtxt("Bet/AppDelay-Homo-5.csv", delimiter=',')
	X5_Bet=np.matrix(m5_Bet)
	m6_Bet = np.loadtxt("Bet/AppDelay-Homo-6.csv", delimiter=',')
	X6_Bet=np.matrix(m6_Bet)
	m7_Bet = np.loadtxt("Bet/AppDelay-Homo-7.csv", delimiter=',')
	X7_Bet=np.matrix(m7_Bet)
	m8_Bet = np.loadtxt("Bet/AppDelay-Homo-8.csv", delimiter=',')
	X8_Bet=np.matrix(m8_Bet)
	m9_Bet = np.loadtxt("Bet/AppDelay-Homo-9.csv", delimiter=',')
	X9_Bet=np.matrix(m9_Bet)
	m10_Bet = np.loadtxt("Bet/AppDelay-Homo-10.csv", delimiter=',')
	X10_Bet=np.matrix(m10_Bet)
	
	m_mean_Bet = np.zeros((12,500))
	
	m1_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-1.csv", delimiter=',')
	X1_NonHomo=np.matrix(m1_NonHomo)
	m2_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-2.csv", delimiter=',')
	X2_NonHomo=np.matrix(m2_NonHomo)
	m3_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-3.csv", delimiter=',')
	X3_NonHomo=np.matrix(m3_NonHomo)
	m4_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-4.csv", delimiter=',')
	X4_NonHomo=np.matrix(m4_NonHomo)
	m5_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-5.csv", delimiter=',')
	X5_NonHomo=np.matrix(m5_NonHomo)
	m6_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-6.csv", delimiter=',')
	X6_NonHomo=np.matrix(m6_NonHomo)
	m7_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-7.csv", delimiter=',')
	X7_NonHomo=np.matrix(m7_NonHomo)
	m8_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-8.csv", delimiter=',')
	X8_NonHomo=np.matrix(m8_NonHomo)
	m9_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-9.csv", delimiter=',')
	X9_NonHomo=np.matrix(m9_NonHomo)
	m10_NonHomo = np.loadtxt("NonHomo/AppDelay-NonHomo-10.csv", delimiter=',')
	X10_NonHomo=np.matrix(m10_NonHomo)
	
	m_mean_NonHomo = np.zeros((12,500))

	for i in range(m_mean_NonHomo.shape[0]):
		for j in range(m_mean_NonHomo.shape[1]):
			m_mean_Bet.itemset((i,j),(X1_Bet.item(i,j)+X2_Bet.item(i,j)+X3_Bet.item(i,j)+X4_Bet.item(i,j)+X5_Bet.item(i,j)+X6_Bet.item(i,j)+X7_Bet.item(i,j)+X8_Bet.item(i,j)+X9_Bet.item(i,j)+X10_Bet.item(i,j))/10)
			m_mean_NonHomo.itemset((i,j),(X1_NonHomo.item(i,j)+X2_NonHomo.item(i,j)+X3_NonHomo.item(i,j)+X4_NonHomo.item(i,j)+X5_NonHomo.item(i,j)+X6_NonHomo.item(i,j)+X7_NonHomo.item(i,j)+X8_NonHomo.item(i,j)+X9_NonHomo.item(i,j)+X10_NonHomo.item(i,j))/10)
			

	font = {'family' : 'STIXGeneral','weight' : 'bold','size'   : 28}
	
	matplotlib.rc('font', **font)
	tedaad = 3
	x = np.linspace(0,tedaad,tedaad)
	x2 = np.linspace(0,12,12)
	AVG_Bet = np.zeros((1,300))
	AVG_Bet = np.copy(np.average(m_mean_Bet[0:12,0:300], axis=0))
	AVG_NonHomo = np.zeros((1,300))
	AVG_NonHomo = np.copy(np.average(m_mean_NonHomo[0:12,0:300], axis=0))
	
	'''print np.average(m_mean_Homo[:,0:500], axis = 1)
	print np.average(m_mean_NonHomo[:,0:500], axis = 1)

	plt.plot(x2,np.average(m_mean_Homo[:,0:500], axis = 1),'-c',label="Ubiquitous")
	plt.plot(x2,np.average(m_mean_NonHomo[:,0:500], axis = 1),'-r',label="ProposedMethod")
	plt.legend(loc='upper left')
	plt.title('Comparison in Interest-Data Delay')
	plt.xlim([0,12])
	plt.ylim([0.6,1.6])
	plt.xlabel('Consumer ID')
	plt.ylabel('Delay (MilliSecond)')
	plt.show()'''

	counter = 0
	
	num_of_chunks = 0

	print  np.around(np.average(m_mean_Bet),6),"\n",np.around(np.average(m_mean_NonHomo),6),"\n"

	print counter
	
	Smooth_Bet = np.zeros((tedaad,1))
	Smooth_NonHomo = np.zeros((tedaad,1))
	j = 0
	for i in range(AVG_NonHomo.shape[0]):
		if ((j+1)*(300/tedaad)) <= i:
			j = j + 1
		Smooth_Bet[j] = Smooth_Bet[j] + AVG_Bet[i]
		Smooth_NonHomo[j] = Smooth_NonHomo[j] + AVG_NonHomo[i]

	for i in range(Smooth_NonHomo.shape[0]):
		Smooth_Bet [i] = Smooth_Bet [i] / (300/tedaad)
		Smooth_NonHomo [i] = Smooth_NonHomo [i] / (300/tedaad)
	
	print np.around(Smooth_Bet,6)
	print np.around(Smooth_NonHomo,6)
	#x = [0.8,1.8,2.8,3.8,4.8]
	#x3 = [ x1+0.2 for x1 in x ]
	plt.bar(0.95,np.around(Smooth_Bet[0],6), width=0.05, color='cyan',label="Betweenness")#,align='center')
	plt.bar(1,np.around(Smooth_NonHomo[0],6), width=0.05, color='red',label="ProposedMethod")#, align='center')

	#plt.plot(x,np.around(Smooth_Homo,6),'-c',label="Ubiquitous",lw=6,ms=5)
	#plt.plot(x,np.around(Smooth_NonHomo,6),'-r',label="ProposedMethod",lw=6,ms=5)
	plt.legend(loc='upper right')
	plt.title('Comparison in Average Interest-Data Delays')
	#plt.xlim([0.85,1.25])	
	plt.ylim([5.125,5.137])
	#plt.xlabel('Simulaton Time(*100 Seconds)')
	plt.ylabel('Delay (MilliSecond)')
	plt.show()
np.array(CS(),dtype='float32')

