import numpy as np
import scipy as sp
from scipy import linalg

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *
	
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
	
	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			m_mean.itemset((i,j),(X1.item(i,j)+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)
	return m_mean
	
def CS2():

	m1 = np.loadtxt("Hit-NonHomo-1.csv", delimiter=',')
	X1=np.matrix(m1)
	m2 = np.loadtxt("Hit-NonHomo-2.csv", delimiter=',')
	X2=np.matrix(m2)
	m3 = np.loadtxt("Hit-NonHomo-3.csv", delimiter=',')
	X3=np.matrix(m3)
	m4 = np.loadtxt("Hit-NonHomo-4.csv", delimiter=',')
	X4=np.matrix(m4)
	m5 = np.loadtxt("Hit-NonHomo-5.csv", delimiter=',')
	X5=np.matrix(m5)
	m6 = np.loadtxt("Hit-NonHomo-6.csv", delimiter=',')
	X6=np.matrix(m6)
	m7 = np.loadtxt("Hit-NonHomo-7.csv", delimiter=',')
	X7=np.matrix(m7)
	m8 = np.loadtxt("Hit-NonHomo-8.csv", delimiter=',')
	X8=np.matrix(m8)
	m9 = np.loadtxt("Hit-NonHomo-9.csv", delimiter=',')
	X9=np.matrix(m9)
	m10 = np.loadtxt("Hit-NonHomo-10.csv", delimiter=',')
	X10=np.matrix(m10)

	m_mean = np.zeros((11,500))
	
	for i in range(m_mean.shape[0]):
		for j in range(m_mean.shape[1]):
			m_mean.itemset((i,j),(X1.item(i,j)+X2.item(i,j)+X3.item(i,j)+X4.item(i,j)+X5.item(i,j)+X6.item(i,j)+X7.item(i,j)+X8.item(i,j)+X9.item(i,j)+X10.item(i,j))/10)			
	return m_mean


cs	 = np.array(CS().transpose(),dtype='float32')
cs2	 = np.array(CS2().transpose(),dtype='float32')
print cs.shape[0]

count_Homo_Hit = np.zeros((300,1)); count_NonHomo_Hit = np.zeros((300,1));
for  i in range(cs.shape[0]-200):
	for j in range(cs.shape[1]):
		count_Homo_Hit[i] = count_Homo_Hit[i] + cs[i,j]
		count_NonHomo_Hit[i] = count_NonHomo_Hit[i] + cs2[i,j]
	#print count_Homo_Hit[i], count_NonHomo_Hit[i]
print np.average(count_Homo_Hit),  np.average(count_NonHomo_Hit)

font = {'family' : 'normal','weight' : 'bold','size'   : 30}
matplotlib.rc('font', **font)

tedaad = 3
tedaad2 = tedaad * 2
x = np.linspace(1,tedaad+0.5,tedaad2)
x2 = np.linspace(1,11,11)

Smooth_Homo    = np.zeros((tedaad2,1))
Smooth_NonHomo = np.zeros((tedaad2,1))
	
j = 0

for i in range(count_NonHomo_Hit.shape[0]):
	if ((j+1)*(300/tedaad2)) <= i:
		j = j + 1
	Smooth_Homo[j]    = Smooth_Homo[j] + count_Homo_Hit[i]
	Smooth_NonHomo[j] = Smooth_NonHomo[j] + count_NonHomo_Hit[i]

for i in range(Smooth_NonHomo.shape[0]):
	Smooth_Homo [i]    = Smooth_Homo [i] / (300/tedaad2)
	Smooth_NonHomo [i] = Smooth_NonHomo [i] / (300/tedaad2)

print Smooth_Homo
print Smooth_NonHomo

plt.plot(x,Smooth_Homo,'-c',label="Betweenness",lw=6,ms=5)
plt.plot(x,Smooth_NonHomo,'-r',label="ProposedMethod",lw=6,ms=5)

plt.legend(loc='upper right')
plt.title('Number of Unique Hit Interests in NDN Routers')
plt.xlim([1,tedaad])
plt.ylim([0.00,1.4])
plt.xlabel('Simulation Time (*100 Seconds)')
plt.ylabel('# of Hits')
plt.show()

