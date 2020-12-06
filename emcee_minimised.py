import numpy as np
import csv
import emcee
import chi2_function
import os
import sys
import time

os.system('clear')

with open("current_working_file.txt", "r") as f:
	file_id = str(f.read())
	
localtime = time.asctime( time.localtime(time.time()) )												# Get the current local time
print ''
print '	Time started:	'+str(localtime)																			# Print current lcoal time (time the code started)
print '	Processing file:  '+str(file_id[7:-1])		
print ''

# First, define the probability distribution that you would like to sample.
def lnprob(p):
    return -chi2_function.chi2_calc(p)/2

ndim = 7
nwalkers = 50
burn_in = 100
sample_steps = 250

p		= np.zeros([ndim],np.float64)
p0_arr	=	np.zeros([ndim],np.float64)


best_params = str(file_id)+"output/best_fit_parameters_"+str(file_id[7:24])+".txt"

best_parameters  = csv.reader(open(best_params,'r'), delimiter=',', skipinitialspace=True)
for line in best_parameters:
	p[0]		=	np.float64(line[0])																							# d
	p[1]		=	np.float64(line[1])																							# q
	p[2]		=	np.float64(line[2])																							# rho
	p[3]		=	np.float64(line[3])																							# u0
	p[4]		=	np.float64(line[4])																							# phi
	p[5]		=	np.float64(line[5])																							# t0
	p[6]		=	np.float64(line[6])																							# tE



#### Use in the best 50 solutions from the grid search as the walkers.
#all_params = str(file_id)+"output/all_fit_parameters_"+str(file_id[7:24])+".txt"
#all_param_values  = csv.reader(open(all_params,'r'), delimiter=',', skipinitialspace=True)
#line_number = -2
#for line in all_param_values:
	#if line_number == -2:
		#line_number += 1
	#elif line_number == -1:
		#ld	=	np.float64(line[0])
		#dd	=	np.float64(line[1])
		#ud	=	np.float64(line[2])
		#lq	=	np.float64(line[3])
		#dq	=	np.float64(line[4])
		#uq	=	np.float64(line[5])
		
		#total_values	= len(np.arange(ld,ud+dd,dd)) * len(np.arange(lq,uq+dq,dq))
		#params			=	np.zeros([8,total_values],np.float32)
		#arr_sorted		=	np.zeros([8,total_values],np.float32)
		
		#line_number += 1
		#count = 0
	#else:
		#params[0,count]	= np.float32(line[0])
		#params[1,count]	= np.float32(line[1])
		#params[2,count]	= np.float32(line[2])
		#params[3,count]	= np.float32(line[3])
		#params[4,count]	= np.float32(line[4])
		#params[5,count]	= np.float32(line[5])
		#params[6,count]	= np.float32(line[6])
		#params[7,count]	= np.float32(line[9])
		
		#count += 1
		#line_number += 1

#np.set_printoptions(threshold=np.nan)
#indxs =  np.argsort(params[7])
#for i in range(0,8):
	#arr_sorted[i,:] = params[i,indxs]




## Choose an initial set of positions for the walkers.
##p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]
#j = 0
#p0 = [np.array(ndim) for i in xrange(nwalkers)]
#while (j<nwalkers):
	#p0_arr[0]	=	arr_sorted[0,j+1]										# d
	#p0_arr[1]	=	arr_sorted[1,j+1]										# q
	#p0_arr[2]	=	arr_sorted[2,j+1] +0.0005*(2*np.random.rand()-1)		# rho
	#p0_arr[3]	=	arr_sorted[3,j+1]										# u0
	#p0_arr[4]	=	arr_sorted[4,j+1]										# phi
	#p0_arr[5]	=	arr_sorted[5,j+1]										# t0
	#p0_arr[6]	=	arr_sorted[6,j+1]										# tE
	#p0[j] =np.array(p0_arr)
	#j += 1




#j = 0
#p0 = [np.array(ndim) for i in xrange(nwalkers)]
#while (j<nwalkers):
	#p0_arr[0]	=	p[0] + 0.01*(2*np.random.rand()-1)														# d
	#p0_arr[1]	=	p[1] + p[1]*0.01*(2*np.random.rand()-1)													# q
	#p0_arr[2]	=	p[2] + p[2]*0.05*(2*np.random.rand()-1)													# rho
	#p0_arr[3]	=	p[3] + p[3]*0.01*(2*np.random.rand()-1)													# u0
	#p0_arr[4]	=	p[4] + 0.006*(2*np.random.rand()-1)														# phi
	#p0_arr[5]	=	p[5] + p[6]*0.02*(2*np.random.rand()-1)													# t0
	#p0_arr[6]	=	p[6] + p[6]*0.02*(2*np.random.rand()-1)													# tE
	#p0[j] =np.array(p0_arr)
	#j += 1


j = 0
p0 = [np.array(ndim) for i in xrange(nwalkers)]
while (j<nwalkers):
	p0_arr[0]	=	p[0] + 0.25*(2*np.random.rand()-1)														# d
	p0_arr[1]	=	p[1] + p[1]*0.1*(2*np.random.rand()-1)													# q
	p0_arr[2]	=	p[2] + p[2]*0.05*(2*np.random.rand()-1)													# rho
	p0_arr[3]	=	p[3] + p[3]*0.1*(2*np.random.rand()-1)													# u0
	p0_arr[4]	=	p[4] + np.pi/25*(2*np.random.rand()-1)														# phi
	p0_arr[5]	=	p[5] + p[6]*0.05*(2*np.random.rand()-1)													# t0
	p0_arr[6]	=	p[6] + p[6]*0.05*(2*np.random.rand()-1)													# tE
	p0[j] =np.array(p0_arr)
	j += 1


## Choose an initial set of positions for the walkers.
##p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]
#j = 0
#p0 = [np.array(ndim) for i in xrange(nwalkers)]
#while (j<nwalkers):
	#p0_arr[0]	=	p[0] + 0.01*(2*np.random.rand()-1)														# d
	#p0_arr[1]	=	p[1] + p[1]*0.01*(2*np.random.rand()-1)													# q
	#p0_arr[2]	=	p[2] + p[2]*0.05*(2*np.random.rand()-1)													# rho
	#p0_arr[3]	=	p[3] + p[3]*0.01*(2*np.random.rand()-1)													# u0
	#p0_arr[4]	=	p[4] + 0.006*(2*np.random.rand()-1)														# phi
	#p0_arr[5]	=	p[5] + p[6]*0.02*(2*np.random.rand()-1)													# t0
	#p0_arr[6]	=	p[6] + p[6]*0.02*(2*np.random.rand()-1)													# tE
	#p0[j] =np.array(p0_arr)
	#j += 1

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

print '	Code Initialised...'
print ''
print '	Start Burn in of '+str(burn_in)+' steps, of '+str(nwalkers)+' walkers, for '+str(ndim)+' dimensions'
print ''

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, burn_in)

# Reset the chain to remove the burn-in samples.
sampler.reset()
print '	Burn in complete'
# Starting from the final position in the burn-in chain, sample for 1000
# steps.

print '\a'
print '\a'

print ''
localtime = time.asctime( time.localtime(time.time()) )												# Get the current local time
print '	Start long sample of '+str(sample_steps)+' steps at: '+str(localtime)

sampler.run_mcmc(pos, sample_steps, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

# If you have installed acor (http://github.com/dfm/acor), you can estimate
# the autocorrelation time for the chain. The autocorrelation time is also
# a vector with 50 entries (one for each dimension of parameter space).
try:
    print "Autocorrelation time:", sampler.acor
except ImportError:
    print "You can install acor: http://github.com/dfm/acor"

    
with open(str(file_id)+"output/mcmc_parameter_chains_"+str(file_id[7:24])+".txt", "w") as f:									# Write minimised d/q chi^2 values to file
	f.write('mcmc parameter outputs for '+str(file_id[7:24]))
	f.write('\n%0.17f, %0.17f, %0.17f'% (nwalkers, burn_in, sample_steps))
	f.write('\n')
	for i in range (0,ndim):
		f.write( ','.join(map(str,sampler.flatchain[:,i])) )
		f.write('\n')

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
#try:
    #import matplotlib.pyplot as pl
#except ImportError:
    #print "Try installing matplotlib to generate some sweet plots..."
#else:
    #pl.hist(sampler.flatchain[:,0], 100)
    ##pl.show()


localtime = time.asctime( time.localtime(time.time()) )												# Get the current local time

print ''
print 'Code finished ('+str(localtime)+')'																			# Output when the code finished
print 'Exiting...'
print '\a'
print '\a'
print '\a'
