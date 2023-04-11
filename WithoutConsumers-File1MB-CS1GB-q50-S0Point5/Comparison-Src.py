
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *

def CS():
	m1_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-1.csv", delimiter=',')
	X1_HomoHit=np.matrix(m1_HomoHit)
	m2_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-2.csv", delimiter=',')
	X2_HomoHit=np.matrix(m2_HomoHit)
	m3_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-3.csv", delimiter=',')
	X3_HomoHit=np.matrix(m3_HomoHit)
	m4_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-4.csv", delimiter=',')
	X4_HomoHit=np.matrix(m4_HomoHit)
	m5_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-5.csv", delimiter=',')
	X5_HomoHit=np.matrix(m5_HomoHit)
	m6_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-6.csv", delimiter=',')
	X6_HomoHit=np.matrix(m6_HomoHit)
	m7_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-7.csv", delimiter=',')
	X7_HomoHit=np.matrix(m7_HomoHit)
	m8_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-8.csv", delimiter=',')
	X8_HomoHit=np.matrix(m8_HomoHit)
	m9_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-9.csv", delimiter=',')
	X9_HomoHit=np.matrix(m9_HomoHit)
	m10_HomoHit = np.loadtxt("Bet/Hit-Src-Homo-10.csv", delimiter=',')
	X10_HomoHit=np.matrix(m10_HomoHit)
	
	m_mean_HomoHit = np.zeros((4,500))	
	
	m1_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-1.csv", delimiter=',')
	X1_HomoHitMiss=np.matrix(m1_HomoHitMiss)
	m2_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-2.csv", delimiter=',')
	X2_HomoHitMiss=np.matrix(m2_HomoHitMiss)
	m3_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-3.csv", delimiter=',')
	X3_HomoHitMiss=np.matrix(m3_HomoHitMiss)
	m4_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-4.csv", delimiter=',')
	X4_HomoHitMiss=np.matrix(m4_HomoHitMiss)
	m5_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-5.csv", delimiter=',')
	X5_HomoHitMiss=np.matrix(m5_HomoHitMiss)
	m6_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-6.csv", delimiter=',')
	X6_HomoHitMiss=np.matrix(m6_HomoHitMiss)
	m7_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-7.csv", delimiter=',')
	X7_HomoHitMiss=np.matrix(m7_HomoHitMiss)
	m8_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-8.csv", delimiter=',')
	X8_HomoHitMiss=np.matrix(m8_HomoHitMiss)
	m9_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-9.csv", delimiter=',')
	X9_HomoHitMiss=np.matrix(m9_HomoHitMiss)
	m10_HomoHitMiss = np.loadtxt("Bet/HitMiss-Src-Homo-10.csv", delimiter=',')
	X10_HomoHitMiss=np.matrix(m10_HomoHitMiss)
	
	m_mean_HomoHitMiss = np.zeros((4,500))
	HitRateHomo=np.zeros((4,500))

	m1_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-1.csv", delimiter=',')
	X1_NonHomoHit=np.matrix(m1_NonHomoHit)
	m2_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-2.csv", delimiter=',')
	X2_NonHomoHit=np.matrix(m2_NonHomoHit)
	m3_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-3.csv", delimiter=',')
	X3_NonHomoHit=np.matrix(m3_NonHomoHit)
	m4_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-4.csv", delimiter=',')
	X4_NonHomoHit=np.matrix(m4_NonHomoHit)
	m5_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-5.csv", delimiter=',')
	X5_NonHomoHit=np.matrix(m5_NonHomoHit)
	m6_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-6.csv", delimiter=',')
	X6_NonHomoHit=np.matrix(m6_NonHomoHit)
	m7_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-7.csv", delimiter=',')
	X7_NonHomoHit=np.matrix(m7_NonHomoHit)
	m8_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-8.csv", delimiter=',')
	X8_NonHomoHit=np.matrix(m8_NonHomoHit)
	m9_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-9.csv", delimiter=',')
	X9_NonHomoHit=np.matrix(m9_NonHomoHit)
	m10_NonHomoHit = np.loadtxt("NonHomo/Hit-Src-NonHomo-10.csv", delimiter=',')
	X10_NonHomoHit=np.matrix(m10_NonHomoHit)
	
	m_mean_NonHomoHit = np.zeros((4,500))	

	m1_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-1.csv", delimiter=',')
	X1_NonHomoHitMiss=np.matrix(m1_NonHomoHitMiss)
	m2_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-2.csv", delimiter=',')
	X2_NonHomoHitMiss=np.matrix(m2_NonHomoHitMiss)
	m3_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-3.csv", delimiter=',')
	X3_NonHomoHitMiss=np.matrix(m3_NonHomoHitMiss)
	m4_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-4.csv", delimiter=',')
	X4_NonHomoHitMiss=np.matrix(m4_NonHomoHitMiss)
	m5_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-5.csv", delimiter=',')
	X5_NonHomoHitMiss=np.matrix(m5_NonHomoHitMiss)
	m6_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-6.csv", delimiter=',')
	X6_NonHomoHitMiss=np.matrix(m6_NonHomoHitMiss)
	m7_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-7.csv", delimiter=',')
	X7_NonHomoHitMiss=np.matrix(m7_NonHomoHitMiss)
	m8_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-8.csv", delimiter=',')
	X8_NonHomoHitMiss=np.matrix(m8_NonHomoHitMiss)
	m9_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-9.csv", delimiter=',')
	X9_NonHomoHitMiss=np.matrix(m9_NonHomoHitMiss)
	m10_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Src-NonHomo-10.csv", delimiter=',')
	X10_NonHomoHitMiss=np.matrix(m10_NonHomoHitMiss)

	m_mean_NonHomoHitMiss = np.zeros((4,500))
	HitRateNonHomo=np.zeros((4,500))

	for i in range(m_mean_NonHomoHitMiss.shape[0]):
		for j in range(m_mean_NonHomoHitMiss.shape[1]):
			
			m_mean_HomoHit.itemset((i,j),(X1_HomoHit.item(i,j)+X2_HomoHit.item(i,j)+X3_HomoHit.item(i,j)+X4_HomoHit.item(i,j)+X5_HomoHit.item(i,j)+X6_HomoHit.item(i,j)+X7_HomoHit.item(i,j)+X8_HomoHit.item(i,j)+X9_HomoHit.item(i,j)+X10_HomoHit.item(i,j))/10)
			m_mean_HomoHitMiss.itemset((i,j),(X1_HomoHitMiss.item(i,j)+X2_HomoHitMiss.item(i,j)+X3_HomoHitMiss.item(i,j)+X4_HomoHitMiss.item(i,j)+X5_HomoHitMiss.item(i,j)+X6_HomoHitMiss.item(i,j)+X7_HomoHitMiss.item(i,j)+X8_HomoHitMiss.item(i,j)+X9_HomoHitMiss.item(i,j)+X10_HomoHitMiss.item(i,j))/10)
			'''if m_mean_HomoHitMiss.item(i,j) != 0 :
				HitRateHomo.itemset((i,j),m_mean_HomoHit.item(i,j)/m_mean_HomoHitMiss.item(i,j))
			else:
				HitRateHomo.itemset((i,j),0)'''

			m_mean_NonHomoHit.itemset((i,j),(X1_NonHomoHit.item(i,j)+X2_NonHomoHit.item(i,j)+X3_NonHomoHit.item(i,j)+X4_NonHomoHit.item(i,j)+X5_NonHomoHit.item(i,j)+X6_NonHomoHit.item(i,j)+X7_NonHomoHit.item(i,j)+X8_NonHomoHit.item(i,j)+X9_NonHomoHit.item(i,j)+X10_NonHomoHit.item(i,j))/10)
			m_mean_NonHomoHitMiss.itemset((i,j),(X1_NonHomoHitMiss.item(i,j)+X2_NonHomoHitMiss.item(i,j)+X3_NonHomoHitMiss.item(i,j)+X4_NonHomoHitMiss.item(i,j)+X5_NonHomoHitMiss.item(i,j)+X6_NonHomoHitMiss.item(i,j)+X7_NonHomoHitMiss.item(i,j)+X8_NonHomoHitMiss.item(i,j)+X9_NonHomoHitMiss.item(i,j)+X10_NonHomoHitMiss.item(i,j))/10)
			'''if m_mean_NonHomoHitMiss.item(i,j) != 0 :
				HitRateNonHomo.itemset((i,j),m_mean_NonHomoHit.item(i,j)/m_mean_NonHomoHitMiss.item(i,j))
			else:
				HitRateNonHomo.itemset((i,j),0)	'''		

		font = {'family' : 'STIXGeneral','weight' : 'bold','size'   : 30}

	matplotlib.rc('font', **font)
	
	tedaad = 3
	
	x = np.linspace(1,tedaad,tedaad)
	x2 = np.linspace(1,4,4)
	
	count_Homo_Hit = np.zeros((300,1)); count_NonHomo_Hit = np.zeros((300,1));
	for  i in range(m_mean_HomoHit.shape[1]-200):
		for j in range(m_mean_HomoHit.shape[0]):
				count_Homo_Hit[i] = count_Homo_Hit[i] + m_mean_HomoHit[j,i]
				count_NonHomo_Hit[i] = count_NonHomo_Hit[i] + m_mean_NonHomoHit[j,i]
		#print count_Homo_Hit[i], count_NonHomo_Hit[i]
	print np.average(count_Homo_Hit),  np.average(count_NonHomo_Hit)

	#print m_mean_HomoHitMiss.shape[0]
	count_Homo_HitMiss = np.zeros((300,1)); count_NonHomo_HitMiss = np.zeros((300,1))
	count_New1 = 0
	count_New2 = 0
	for  i in range(m_mean_HomoHitMiss.shape[1]-200):
		for j in range(m_mean_HomoHitMiss.shape[0]):
				count_Homo_HitMiss[i] = count_Homo_HitMiss[i] + m_mean_HomoHitMiss[j,i]
				count_NonHomo_HitMiss[i] = count_NonHomo_HitMiss[i] + m_mean_NonHomoHitMiss[j,i]
		#print count_Homo_HitMiss[i], count_NonHomo_HitMiss[i]
		if count_Homo_HitMiss[i] != 0 or count_NonHomo_HitMiss[i] !=0:
			if count_Homo_Hit[i] != 0 or count_NonHomo_Hit[i] !=0:
				count_New1 = count_New1 + float(count_Homo_Hit[i]/count_Homo_HitMiss[i]) 
				count_New2 = count_New2 + float(count_NonHomo_Hit[i]/count_NonHomo_HitMiss[i])
	#print count_New1, count_New2

	Smooth_Homo    = np.zeros((tedaad,1))
	Smooth_NonHomo = np.zeros((tedaad,1))
	
	Smooth_Homo_HitMiss    = np.zeros((tedaad,1))
	Smooth_NonHomo_HitMiss = np.zeros((tedaad,1))

	Smooth_Homo_HitRate    = np.zeros((tedaad,1))
	Smooth_NonHomo_HitRate = np.zeros((tedaad,1))

	j = 0

	for i in range(count_NonHomo_Hit.shape[0]):
		if ((j+1)*(300/tedaad)) <= i:
			j = j + 1
		Smooth_Homo[j]    = Smooth_Homo[j] + count_Homo_Hit[i]
		Smooth_NonHomo[j] = Smooth_NonHomo[j] + count_NonHomo_Hit[i]
		
		Smooth_Homo_HitMiss[j]    = Smooth_Homo_HitMiss[j] + count_Homo_HitMiss[i]
		Smooth_NonHomo_HitMiss[j] = Smooth_NonHomo_HitMiss[j] + count_NonHomo_HitMiss[i]

	for i in range(Smooth_NonHomo.shape[0]):
		Smooth_Homo [i]    = Smooth_Homo [i] / (300/tedaad)
		Smooth_NonHomo [i] = Smooth_NonHomo [i] / (300/tedaad)

		Smooth_Homo_HitMiss [i]    = Smooth_Homo_HitMiss [i] / (300/tedaad)
		Smooth_NonHomo_HitMiss [i] = Smooth_NonHomo_HitMiss [i] / (300/tedaad)
		if Smooth_Homo_HitMiss[i] != 0 or Smooth_NonHomo_HitMiss[i] !=0:
			if Smooth_Homo[i] != 0 or Smooth_NonHomo[i] !=0:
				Smooth_Homo_HitRate[i] = float(Smooth_Homo[i])/float(Smooth_Homo_HitMiss[i])
				Smooth_NonHomo_HitRate[i] = float(Smooth_NonHomo[i])/float(Smooth_NonHomo_HitMiss[i])
				if Smooth_Homo[i]/Smooth_Homo_HitMiss[i] > 1:
					Smooth_Homo_HitRate[i]=1
				if Smooth_NonHomo[i]/Smooth_NonHomo_HitMiss[i] > 1:
					Smooth_NonHomo_HitRate[i]=1

	print "\nNumber of Hits"; print Smooth_Homo;print Smooth_NonHomo
	print "\nNumber of Hits+Misses"; print Smooth_Homo_HitMiss;print Smooth_NonHomo_HitMiss
	#print "\nHit Ratio: ";print Smooth_Homo_HitRate; print Smooth_NonHomo_HitRate
	print "\nHit Ratio: ";print  np.average(Smooth_Homo_HitRate); print  np.average(Smooth_NonHomo_HitRate)
	#plt.plot(x,Smooth_Bet/Smooth_Homo_HitMiss,'-c',label="Ubiquitous",lw=6,ms=5)
	#plt.plot(x,Smooth_NonHomo/Smooth_NonHomo_HitMiss,'-r',label="ProposedMethod",lw=6,ms=5)

	'''x = [0.8,1.8,2.8,3.8,4.8]
	x3 = [ x1+0.2 for x1 in x ]
	print x
	print x3
	plt.bar(x,Smooth_Homo_HitRate, width=0.2, color='green')#,align='center')
	plt.bar(x3,Smooth_NonHomo_HitRate, width=0.2, color='red')#, align='center')'''
	plt.plot(x,Smooth_Homo_HitRate,'-c',label="Betweenness",lw=6,ms=5)
	plt.plot(x,Smooth_NonHomo_HitRate,'-r',label="ProposedMethod",lw=6,ms=5)

	plt.legend(loc='upper right')
	#plt.title('Producers')#Average Hit Ratio of 
	plt.xlim([1,tedaad])
	plt.ylim([0.00,0.005])
	plt.xlabel('Simulation Time (*100 Seconds)')
	plt.ylabel('HitRatio')
	plt.show()

np.array(CS(),dtype='float32')
