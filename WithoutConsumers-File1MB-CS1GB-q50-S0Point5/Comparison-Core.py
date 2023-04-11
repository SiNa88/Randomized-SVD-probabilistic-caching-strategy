
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pylab import *


def CS():
	m1_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-1.csv", delimiter=',')
	X1_HomoHit=np.matrix(m1_HomoHit)
	m2_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-2.csv", delimiter=',')
	X2_HomoHit=np.matrix(m2_HomoHit)
	m3_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-3.csv", delimiter=',')
	X3_HomoHit=np.matrix(m3_HomoHit)
	m4_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-4.csv", delimiter=',')
	X4_HomoHit=np.matrix(m4_HomoHit)
	m5_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-5.csv", delimiter=',')
	X5_HomoHit=np.matrix(m5_HomoHit)
	m6_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-6.csv", delimiter=',')
	X6_HomoHit=np.matrix(m6_HomoHit)
	m7_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-7.csv", delimiter=',')
	X7_HomoHit=np.matrix(m7_HomoHit)
	m8_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-8.csv", delimiter=',')
	X8_HomoHit=np.matrix(m8_HomoHit)
	m9_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-9.csv", delimiter=',')
	X9_HomoHit=np.matrix(m9_HomoHit)
	m10_HomoHit = np.loadtxt("Bet/Hit-Core-Homo-10.csv", delimiter=',')
	X10_HomoHit=np.matrix(m10_HomoHit)
	
	m_mean_HomoHit = np.zeros((11,500))	
	
	m1_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-1.csv", delimiter=',')
	X1_HomoHitMiss=np.matrix(m1_HomoHitMiss)
	m2_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-2.csv", delimiter=',')
	X2_HomoHitMiss=np.matrix(m2_HomoHitMiss)
	m3_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-3.csv", delimiter=',')
	X3_HomoHitMiss=np.matrix(m3_HomoHitMiss)
	m4_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-4.csv", delimiter=',')
	X4_HomoHitMiss=np.matrix(m4_HomoHitMiss)
	m5_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-5.csv", delimiter=',')
	X5_HomoHitMiss=np.matrix(m5_HomoHitMiss)
	m6_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-6.csv", delimiter=',')
	X6_HomoHitMiss=np.matrix(m6_HomoHitMiss)
	m7_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-7.csv", delimiter=',')
	X7_HomoHitMiss=np.matrix(m7_HomoHitMiss)
	m8_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-8.csv", delimiter=',')
	X8_HomoHitMiss=np.matrix(m8_HomoHitMiss)
	m9_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-9.csv", delimiter=',')
	X9_HomoHitMiss=np.matrix(m9_HomoHitMiss)
	m10_HomoHitMiss = np.loadtxt("Bet/HitMiss-Core-Homo-10.csv", delimiter=',')
	X10_HomoHitMiss=np.matrix(m10_HomoHitMiss)
	
	m_mean_HomoHitMiss = np.zeros((11,500))
	HitRateHomo=np.zeros((11,500))

	m1_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-1.csv", delimiter=',')
	X1_NonHomoHit=np.matrix(m1_NonHomoHit)
	m2_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-2.csv", delimiter=',')
	X2_NonHomoHit=np.matrix(m2_NonHomoHit)
	m3_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-3.csv", delimiter=',')
	X3_NonHomoHit=np.matrix(m3_NonHomoHit)
	m4_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-4.csv", delimiter=',')
	X4_NonHomoHit=np.matrix(m4_NonHomoHit)
	m5_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-5.csv", delimiter=',')
	X5_NonHomoHit=np.matrix(m5_NonHomoHit)
	m6_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-6.csv", delimiter=',')
	X6_NonHomoHit=np.matrix(m6_NonHomoHit)
	m7_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-7.csv", delimiter=',')
	X7_NonHomoHit=np.matrix(m7_NonHomoHit)
	m8_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-8.csv", delimiter=',')
	X8_NonHomoHit=np.matrix(m8_NonHomoHit)
	m9_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-9.csv", delimiter=',')
	X9_NonHomoHit=np.matrix(m9_NonHomoHit)
	m10_NonHomoHit = np.loadtxt("NonHomo/Hit-Core-NonHomo-10.csv", delimiter=',')
	X10_NonHomoHit=np.matrix(m10_NonHomoHit)
	
	m_mean_NonHomoHit = np.zeros((11,500))	

	m1_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-1.csv", delimiter=',')
	X1_NonHomoHitMiss=np.matrix(m1_NonHomoHitMiss)
	m2_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-2.csv", delimiter=',')
	X2_NonHomoHitMiss=np.matrix(m2_NonHomoHitMiss)
	m3_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-3.csv", delimiter=',')
	X3_NonHomoHitMiss=np.matrix(m3_NonHomoHitMiss)
	m4_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-4.csv", delimiter=',')
	X4_NonHomoHitMiss=np.matrix(m4_NonHomoHitMiss)
	m5_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-5.csv", delimiter=',')
	X5_NonHomoHitMiss=np.matrix(m5_NonHomoHitMiss)
	m6_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-6.csv", delimiter=',')
	X6_NonHomoHitMiss=np.matrix(m6_NonHomoHitMiss)
	m7_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-7.csv", delimiter=',')
	X7_NonHomoHitMiss=np.matrix(m7_NonHomoHitMiss)
	m8_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-8.csv", delimiter=',')
	X8_NonHomoHitMiss=np.matrix(m8_NonHomoHitMiss)
	m9_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-9.csv", delimiter=',')
	X9_NonHomoHitMiss=np.matrix(m9_NonHomoHitMiss)
	m10_NonHomoHitMiss = np.loadtxt("NonHomo/HitMiss-Core-NonHomo-10.csv", delimiter=',')
	X10_NonHomoHitMiss=np.matrix(m10_NonHomoHitMiss)

	m_mean_NonHomoHitMiss = np.zeros((11,500))
	HitRateNonHomo=np.zeros((11,500))

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
				HitRateNonHomo.itemset((i,j),0)'''

	font = {'family' : 'STIXGeneral','weight' : 'bold','size'   : 30}

	matplotlib.rc('font', **font)
	
	tedaad = 3
	tedaad2 = tedaad * 2
	x = np.linspace(1,tedaad+0.5,tedaad2)
	x2 = np.linspace(1,11,11)
	
	count_Homo_Hit1 = np.zeros((11,1)); count_NonHomo_Hit1 = np.zeros((11,1));
	for  i in range(m_mean_HomoHit.shape[0]):
		for j in range(m_mean_HomoHit.shape[1]-200):
				count_Homo_Hit1[i] = count_Homo_Hit1[i] + m_mean_HomoHit[i,j]
				count_NonHomo_Hit1[i] = count_NonHomo_Hit1[i] + m_mean_NonHomoHit[i,j]
		print count_Homo_Hit1[i], count_NonHomo_Hit1[i]
	#print np.average(count_Homo_Hit1),  np.average(count_NonHomo_Hit1)

	plt.plot(x2,count_Homo_Hit1,'-c',label="Betweenness",lw=6,ms=5)
	plt.plot(x2,count_NonHomo_Hit1,'-r',label="ProposedMethod",lw=6,ms=5)
	plt.legend(loc='upper right')
	#plt.title('NDN Routers')#Number of Unique Hit Interests in 
	plt.xlim([1,11])
	plt.ylim([0.00,50])
	plt.xlabel('Node')
	plt.ylabel('# of Unique Hits')
	plt.show()

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

	Smooth_Homo    = np.zeros((tedaad2,1))
	Smooth_NonHomo = np.zeros((tedaad2,1))
	
	
	Smooth_Homo_HitMiss    = np.zeros((tedaad2,1))
	Smooth_NonHomo_HitMiss = np.zeros((tedaad2,1))

	j = 0

	for i in range(count_NonHomo_Hit.shape[0]):
		if ((j+1)*(300/tedaad2)) <= i:
			j = j + 1
		Smooth_Homo[j]    = Smooth_Homo[j] + count_Homo_Hit[i]
		Smooth_NonHomo[j] = Smooth_NonHomo[j] + count_NonHomo_Hit[i]
		
		Smooth_Homo_HitMiss[j]    = Smooth_Homo_HitMiss[j] + count_Homo_HitMiss[i]
		Smooth_NonHomo_HitMiss[j] = Smooth_NonHomo_HitMiss[j] + count_NonHomo_HitMiss[i]

	Smooth_Homo_HitRate    = np.zeros((tedaad2,1))
	Smooth_NonHomo_HitRate = np.zeros((tedaad2,1))

	for i in range(Smooth_NonHomo.shape[0]):
		Smooth_Homo [i]    = Smooth_Homo [i] / (300/tedaad2)
		Smooth_NonHomo [i] = Smooth_NonHomo [i] / (300/tedaad2)

		Smooth_Homo_HitMiss [i]    = Smooth_Homo_HitMiss [i] / (300/tedaad2)
		Smooth_NonHomo_HitMiss [i] = Smooth_NonHomo_HitMiss [i] / (300/tedaad2)
		if Smooth_Homo_HitMiss[i] != 0 or Smooth_NonHomo_HitMiss[i] !=0:
			if Smooth_Homo[i] != 0 or Smooth_NonHomo[i] !=0:
				Smooth_Homo_HitRate[i] = float(Smooth_Homo[i])/float(Smooth_Homo_HitMiss[i])
				Smooth_NonHomo_HitRate[i] = float(Smooth_NonHomo[i])/float(Smooth_NonHomo_HitMiss[i])
				if Smooth_Homo[i]/Smooth_Homo_HitMiss[i] > 1:
					Smooth_Homo_HitRate[i]=1
				if Smooth_NonHomo[i]/Smooth_NonHomo_HitMiss[i] > 1:
					Smooth_NonHomo_HitRate[i]=1
			
	
	plt.plot(x,Smooth_Homo,'-c',label="Betweenness",lw=6,ms=5)
	plt.plot(x,Smooth_NonHomo,'-r',label="ProposedMethod",lw=6,ms=5)
	plt.legend(loc='upper right')
	#plt.title('NDN Routers')#Number of Unique Hit Interests in 
	plt.xlim([1,tedaad])
	plt.ylim([0.00,1.4])
	plt.xlabel('Simulation Time (*100 Seconds)')
	plt.ylabel('# of Unique Hits')
	plt.show()

	print "\nNumber of Hits"; print Smooth_Homo;print Smooth_NonHomo
	print "\nNumber of Hits+Misses"; print Smooth_Homo_HitMiss;print Smooth_NonHomo_HitMiss
	#print "\nHit Ratio: ";print Smooth_Homo_HitRate; print Smooth_NonHomo_HitRate
	print "\nHit Ratio: ";print  np.average(Smooth_Homo_HitRate); print  np.average(Smooth_NonHomo_HitRate)
	'''f = plt.figure()
	ax = f.add_subplot(111)
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')'''
	#plt.plot(x,Smooth_Homo_HitRate,'-c',label="Betweenness",lw=6,ms=5)
	#plt.plot(x,Smooth_NonHomo_HitRate,'-r',label="ProposedMethod",lw=6,ms=5)
	''' Delay:
	[[ 5.134062]
	 [ 0.      ]
	 [ 0.      ]
	 [ 0.      ]
	 [ 0.      ]]
	[[ 5.133581]
	 [ 0.      ]
	 [ 0.      ]
	 [ 0.      ]
	 [ 0.      ]]
	plt.bar(0.9,np.around(Smooth_Homo[0],6), width=0.1, color='green')#,align='center')
	plt.bar(1,np.around(Smooth_NonHomo[0],6), width=0.1, color='red')#, align='center')

	#plt.plot(x,np.around(Smooth_Homo,6),'-c',label="Ubiquitous",lw=6,ms=5)
	#plt.plot(x,np.around(Smooth_NonHomo,6),'-r',label="ProposedMethod",lw=6,ms=5)
	plt.legend(loc='lower right')
	plt.title('Comparison in Interest-Data Delay')
	plt.xlim([0.8,1.2])	
	plt.ylim([5.125,5.135])
	plt.xlabel('Simulaton Time(*100 Seconds)')
	plt.ylabel('Delay (MilliSecond)')
	'''
	'''plt.bar(0.9,np.average(Smooth_Homo_HitRate), width=0.1, color='green')#,align='center')
	plt.bar(1,np.average(Smooth_NonHomo_HitRate), width=0.1, color='red')#, align='center')'''

	#x = [0.8,1.8,2.8,3.8,4.8]
	#x3 = [ x1+0.2 for x1 in x ]
	#print x
	#print x3
	#plt.errorbar(x,Smooth_Bet/Smooth_Homo_HitMiss,yerr=0.4)# width=0.2, color='green',
	#plt.errorbar(x3,Smooth_NonHomo/Smooth_NonHomo_HitMiss,yerr=0.4)#width=0.2, color='red',
	#plt.bar(x,Smooth_Bet/Smooth_Homo_HitMiss, width=0.2, color='green')#,align='center')
	#plt.bar(x3,Smooth_NonHomo/Smooth_NonHomo_HitMiss, width=0.2, color='red')#, align='center')

	plt.plot(x,Smooth_Homo_HitRate,'-c',label="Betweenness",lw=6,ms=5)
	plt.plot(x,Smooth_NonHomo_HitRate,'-r',label="ProposedMethod",lw=6,ms=5)

	plt.legend(loc='upper right')
	#plt.title('NDN Routers')#Average Hit Ratio of 
	plt.xlim([1,tedaad])
	plt.ylim([0.00,0.7])
	plt.xlabel('Simulation Time (*100 Seconds)')
	plt.ylabel('HitRatio')
	plt.show()
	
np.array(CS(),dtype='float32')
