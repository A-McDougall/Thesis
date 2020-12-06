import sys
import string
import os																																																																							# Used for clearing the screen
import math
import csv																																																																						# Used for reading in data	
import time
import numpy as np
import matplotlib.pyplot as plt

iterations = 10000
map_limits = 3.25


#-- d paramter search space
ld		= 0.2																																																																					# Lower limit for the parameter d
ud		= 5																																																																								# Upper limit for the parameter d
dd		= 0.2																																																																							# Step size for the parameter d
#-- q paramter search space																																																																# repeat for each paramter below...
lq		= np.log10(10**-5)					# = 10^-5
uq		= np.log10(10**0)					# = 1
dq		= 0.2								# = log(1.9952...)


for log_q in np.arange(uq,lq-dq,-dq):																																																											# Loop over d
	for d in np.arange(ld,ud+dd,dd):																																																											# Loop over q																																																															# Print the % complete, elapsed time, and est. time remaining, use \r so that it will udpate this line each loop
		q = 10**(log_q)			

		e1 = 1/(1+q)
		e2 = q/(1+q)
		a = 0.5*d
		b = -(a*(q-1)) / (1+q)

		p = np.zeros([4], complex)
		angle = np.zeros([iterations],np.float64)
		solutions = np.zeros((4,iterations),complex)
		zeta = np.zeros((4,iterations),complex)

		#c4 = np.zeros([iterations], complex)
		#c3 = np.zeros([iterations],complex)
		#c2 = np.zeros([iterations],complex)
		#c1 = np.zeros([iterations],complex)
		#c0 = np.zeros([iterations],complex)



		for k in range (0, iterations):
			angle = 2*np.pi / iterations * k
			
			c4 = -np.exp((1j*1)*angle)
			c3 = 0;
			c2 = 1+2*(a**2)*np.exp((1j*1)*angle);
			c1 = -2*a + 4*e1*a;
			c0 = -(a**4)*np.exp((1j*1)*angle) + a**2;
			
			p = [c4,c3,c2,c1,c0]
			
			solutions[:,k] = np.roots(p)
			
		zeta = solutions.conjugate() - e1/(solutions-a) - (1-e1)/(solutions+a)	
		plt.clf()
		plt.figure(figsize=(10.24,10.24))
		ax = plt.subplot(111)

		for k in range (0,4):
			plt.plot(solutions[k,:].real, solutions[k,:].imag, '.', c='0.5', markersize=0.1)
			plt.plot(zeta[k,:].real, zeta[k,:].imag, '.', c='0', markersize=0.1)


		plt.plot(b,0,'o', mfc='none', mec='0.7', mew=1, ms=4)
		plt.plot([-a,a],[0,0],'x', c='0.7',ms=8)

		title = 'Caustic map (d='+str(d)+', q='+str(q)+')'						# labels for the plot
		plt.axis([-map_limits, map_limits, -map_limits, map_limits])

		ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
		ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

		plt.grid(c='0.75',linestyle='--', which='major')
		plt.grid(c='0.8',linestyle='--', which='minor')

		xlabel = 'u1'
		ylabel = 'u2'
		plt.title(title,fontsize='small')
		plt.xlabel(xlabel,fontsize='small')
		plt.ylabel(ylabel,fontsize='small')
		plt.savefig('caustics/q_%0.15f_d_%0.15f.png'% (q,d) )
