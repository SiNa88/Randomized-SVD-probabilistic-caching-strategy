"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont

n_groups = 2

means_men = (5.134062, 5.135575)
std_men = (0.0005429, 0.0005429)

means_women = (5.133581, 5.135094)
std_women = (0.0005429, 0.0006429)
font = {'family' : 'STIXGeneral','weight' : 'bold','size'   : 22}
	
matplotlib.rc('font', **font)
#fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.15

opacity = 0.8
error_config = {'ecolor': '0.3'}

#plt.bar(0.95,np.around(Smooth_Bet[0],6), width=0.05, color='cyan',label="Betweenness")#,align='center')
#plt.bar(1,np.around(Smooth_NonHomo[0],6), width=0.05, color='red',label="ProposedMethod")#, align='center')

rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='c',
                 yerr=std_men,
                 error_kw=error_config,
                 label='Betweenness',align='center')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=std_women,
                 error_kw=error_config,
                 label='ProposedMethod',align='center')

#plt.xlim([0.85,1.25])	
plt.ylim([5.125,5.139])
plt.xlabel('Scenario', fontsize=26)
plt.ylabel('Delay (millisecond)', fontsize=26)
plt.title('Comparison of Average Interest-Data Delays')
plt.xticks(index + bar_width / 2, ('Enabled-Cache Consumers', 'Disabled-Cache Consumers'))
plt.legend()

plt.tight_layout()
plt.show()

'''
With
[[ 5.134062]
 [ 0.      ]
 [ 0.      ]]
[[ 5.133581]
 [ 0.      ]
 [ 0.      ]]

Without
[[ 5.135575]
 [ 0.      ]
 [ 0.      ]]
[[ 5.135094]
 [ 0.      ]
 [ 0.      ]]'''

