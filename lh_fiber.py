#Module lh_fiber
#Created by Aria Coraor 12/6/19

#Analyze the probability distributions of LH-bound 207 nucleosome arrays.
#Create plots in order to better produce an estimate of a useful functional
#Form for the LH effect on fibers.

import numpy as np
import os
from scipy.interpolate.rbf import Rbf
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.stats import gaussian_kde as kde
import sys
#from copy import deepcopy
import argparse
#from vect_quat_util import *
from merlin import *
import molecule
from time import time
from joblib import Parallel,delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HOMEDIR = "/project2/depablo/coraor/fiber_builder/"



def gen_kde_probs():
	"""Generate probabilities from the input datafiles by using gaussian kernel
	density estimation. Normalize so that the integral over all phase space is 1
	for each true distribution.
	Uses a multivariate kernel and normalizes afterwards, since our data confidence
	is structured in 3D. 

	If a distribution has no reads at all, put a single gaussian of width 10 
	angstroms at distance 140 angstroms.

	Writes a tabulation of the free energy at high-resolution.
	"""
	nrls = range(158,208)

	rmids = []
	'''
	for i,nrl in enumerate(nrls):
		if (os.path.isfile("first_cond_%s_kde.dat.npy" % `nrl`) and os.path.isfile("second_cond_%s_kde.dat.npy" % `nrl`) and 
				os.path.isfile("ranges_%s_kde.dat.npy" % `nrl`)):
			rmids.append(i)
	print "removing ids %s" % `rmids`
	'''
	
	for i in reversed(rmids):
		nrls.pop(i)

	#nrls = [187]
	data = ["/project2/depablo/coraor/ris_repex_data/%d/" % ele for ele in nrls]

	#data = ["/media/midway/data.dat"]

	first_conds = []
	second_conds = []    
	#lh_sats = ["0.12500","0.25000","0.75000","0.08333","0.16667","0.50000","1.00000"]
	lh_sats = ["0.12500","0.25000","0.75000","0.16667","0.50000","1.00000"]

	print "Starting parallel kde generation"
	Parallel(n_jobs=-1)(delayed(_gen_kde_prob)(sat) for sat in lh_sats)

	
	


def _gen_kde_prob(nrl):
	"""Helper function to generate the kde probability for a given NRL. Helps
	parallelize this calculation across multiple processors."""
	print "Starting lh saturation: %s" % nrl

	root = "/project2/depablo/coraor/aec_lh_ris/n24/%s/" % nrl
		
	a = np.loadtxt(os.path.join(root,"0/dist_hist_0.dat"))
	b = np.loadtxt(os.path.join(root,"1/dist_hist_1.dat"))
	c = np.loadtxt(os.path.join(root,"2/dist_hist_2.dat"))
	if os.path.isfile(os.path.join(root,"3/dist_hist_3.dat")):
		e = np.loadtxt(os.path.join(root,"3/dist_hist_3.dat"))
		d = np.concatenate((a,b,c,e))
	else:
		d = np.concatenate((a,b,c))

	#d = np.loadtxt(root)

	kernel = kde(d[:,0:3:2].T)
	
	#Now we need to generate a normalized version of the probability kernel
	#for each of the possible values of ralpha (first_cond) or ralpha+ralpha'
	#(second cond)

	#From hand derivations, use the kernel density determined in the other
	#dimensions
	kernel_1d = kde(d[:,0].T, bw_method = kernel.factor)
		


		
	#Confirm condition is normalized

	norm_kernel = lambda alpha,alphaprime: kernel(np.vstack([alpha,alphaprime]))/kernel_1d(alpha)


	'''
	val_min = 10.0
	val_max = 500.0
	val_bins = 1000

	vals = norm_kernel(np.array([180.0]*val_bins),np.linspace(val_min,val_max,val_bins))
	#reweight so vals*interval of vals
	vals *= (val_max-val_min)/val_bins
		
	print "Normed kernel for ralpha' span: " + `vals`
	print "sum: " + `np.sum(vals)`
	'''
	'''
	vals2 = norm_kernel(np.array([170.0]*val_bins),np.linspace(val_min,val_max,val_bins))
	vals3 = norm_kernel(np.array([190.0]*val_bins),np.linspace(val_min,val_max,val_bins))

	vals2 *= (val_max-val_min)
	vals3 *= (val_max-val_min)


	print "sum: " + `np.sum(vals2)`
	print "sum: " + `np.sum(vals3)`

	print "Trying random values"

	randalpha = np.random.rand(200)*80 + 100.0
	randalphap = np.random.rand(200)* 90 + 200.0

	vals4 = norm_kernel(randalpha,randalphap)
	print `vals4`
	vals4 *= np.max(randalphap)-np.min(randalphap)
	print "sum4: " + `np.sum(vals4)`
	'''

	n_bins = 50
	XX,YY = np.meshgrid(np.linspace(np.min(d[:,0]),np.max(d[:,0]),n_bins),np.linspace(np.min(d[:,2]),np.max(d[:,2]),n_bins))

	#plt.contour(XX,YY,-np.log(norm_kernel(XX.ravel(),YY.ravel()).reshape((100,100))),cmap="viridis",levels = range(10))
	#plt.pcolormesh(XX,YY,-np.log(norm_kernel(XX.ravel(),YY.ravel()).reshape((100,100))),cmap="viridis")

	print "Calculating first free energy profile"
	'''
	start1 = time()

	p_first = norm_kernel(XX.ravel(),YY.ravel()).reshape((n_bins,n_bins))
	print "First condition took %s seconds." % `round(time()-start1,3)`
	p_sum = np.sum(p_first,axis=1)
	errors = False
	for i,val in enumerate(p_sum):
		if i < 0.99 or i > 1.01:
			print "Normalization error at alpha %s. Sum: %s" % (`np.linspace(np.min(d[:,0]),np.max(d[:,0]),n_bins)[i]`,`val`)
			errors = True
	if errors:
		for j in range(len(p_first)):
			p_first[j,:] /= np.sum(p_first[j,:])
	#raw kernel
	plot = False
	if plot:
		plt.clf()
		norm = matplotlib.colors.Normalize(vmin=10.0,vmax=30.0)
		Z_vals = (-np.log(kernel(np.vstack([XX.ravel(),YY.ravel()]))).reshape((n_bins,n_bins)))
		plt.contour(XX,YY,Z_vals,colors='black',levels = range(7,20),linewidths=0.5)
		plt.pcolormesh(XX,YY,(-np.log(kernel(np.vstack([XX.ravel(),YY.ravel()]))).reshape((n_bins,n_bins))),cmap="viridis")

		cbar = plt.colorbar(norm = norm)
		cbar.set_clim(8.0,20.0)
		cbar.set_label("Probability density")
		cbar.set_label("Free energy (kT)")

		#Scatter a random subset
		#subset = np.take(d,np.random.randint(0,len(d),len(d)/10),axis=0)
		#plt.scatter(subset[:,0],subset[:,2],color="black",marker='+')

		plt.xlabel("$r_\\alpha$")
		plt.ylabel("$r_{\\alpha'}$")
		plt.title("NRL %s First Condition" % nrl)

		plt.savefig("kde_%s.png" % nrl,dpi=600)
	#Normalized kernel:
	if plot:
		plt.clf()
		Z_vals = -np.log(p_first)
		norm = matplotlib.colors.Normalize(vmin=np.min(Z_vals),vmax=np.min(Z_vals)+20.0)
		plt.contour(XX,YY,Z_vals,colors='black',levels = range(int(np.floor(np.min(Z_vals))),20),linewidths=0.5)
		plt.pcolormesh(XX,YY,Z_vals,cmap="viridis")

		cbar = plt.colorbar(norm = norm)
		cbar.set_clim(vmin=np.min(Z_vals),vmax=np.min(Z_vals)+20.0)
		cbar.set_label("Probability density")
		cbar.set_label("Free energy (kT)")


		plt.xlabel("$r_\\alpha$")
		plt.ylabel("$r_{\\alpha'}$")
		plt.title("NRL %s First Condition Normalized" % nrl)

		plt.savefig("norm_kde_%s.png" % nrl,dpi=600)

	#hquit()


		

	
	#Z_vals = -np.log(norm_kernel(XX.ravel(),YY.ravel())).reshape((n_bins,n_bins))
	#First condition free energies
	g_first = -np.log(p_first)
	#Confirm kernel is normed


	np.save("first_cond_%s_kde.dat" % nrl,g_first)
	'''
	ranges_ralpha = np.linspace(np.min(d[:,0]),np.max(d[:,0]),n_bins)
	ranges_ralphaprime = np.linspace(np.min(d[:,2]),np.max(d[:,2]),n_bins)
	ranges_rbeta = np.linspace(np.min(d[:,1]),np.max(d[:,1]),n_bins)

		
	#Second condition free energies: 
	#print "Making second kde"
	kernel_second = kde(d.T)
	kernel_2d = kde(d[:,0:3:2].T, bw_method = kernel_second.factor)

	#print "second kde done"
	normed_2kernel = lambda alpha, beta, alphaprime: kernel_second(np.vstack([alpha,beta,alphaprime]))/kernel_2d(np.vstack([alpha,alphaprime]))
	np.save("ranges_%s_kde.dat" % nrl,np.array([ranges_ralpha,ranges_rbeta,ranges_ralphaprime]))
	XX,BB,YY = np.meshgrid(ranges_ralpha,ranges_rbeta,ranges_ralphaprime)
	'''
	#Test second kernel
	val_min = 10.0
	val_max = 600.0
	val_bins = 2000

	vals = normed_2kernel(np.array([180.0]*val_bins),np.linspace(val_min,val_max,val_bins),np.array([180.0]*val_bins))
	#reweight so vals*interval of vals
	vals *= ((val_max-val_min)/val_bins)
		
	#print "Normed kernel for ralpha' span: " + `vals`
	print "sum: " + `np.sum(vals)`
	quit()
	'''


	print "Calculating second free energy profile"
	start2 = time()
	p_second = normed_2kernel(XX.ravel(),BB.ravel(),YY.ravel()).reshape((n_bins,n_bins,n_bins))
	

	p_sum1 = np.sum(p_second,axis=1)
	valid = (p_sum1 > 0.99)*(p_sum1 < 1.01)
	if not np.all(valid):
		errors = np.argwhere(valid < 0.5)
		print "Normalization errors at alpha,betas: %s" % (`errors`)
		print "Sums: %s" % (`p_sum1`)

		for j in range(len(p_second)):
			for l in range(len(p_second[0][0])):
				p_second[j,:,l] /= np.sum(p_second[j,:,l])
	print "second condition took %s seconds." % `round(time()-start2,3)`
	print "Saving profiles"
	#shape = e_second.shape
	#print "Sum along rbeta axis for midpoint: " + `np.sum(e_second[shape[0]/2.0,:,shape[2]/2.0])`
	#print "Modified probability array for 2nd cond: " + `e_second`
	#raw_input()
	#g_first = g_first.reshape((n_bins,n_bins))
	#g_second = g_second.reshape((n_bins,n_bins.n_bins))
	g_second = -np.log(p_second)

	np.save("second_cond_%s_kde.dat" % nrl,g_second)

	#np.save("energies_%d.dat" % nrl,g)
	print "Saved data from nrl %s" % nrl
	#np.savetxt("hist_%d.dat" % nrl,e)


def ind_lh_bound(fname):
	"""Return the nucleosome indices which have bound linker histones in in.lammps."""
	atoms,bonds,angles = files.read_input(fname,read_atoms=True)

	atoms = [ele for ele in atoms if ele.mytype == 1]
	gh_bonds = [bond for bond in bonds if bond.mytype == 17]
	gh_ids = np.array([[bond.atom1,bond.atom2] for bond in gh_bonds])
	lh_nucs = []

	for a in atoms:
		if np.any(a.index == gh_ids):
			lh_nucs.append(a.index)

	return lh_nucs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-kde','--kde',action='store_const',const=True,default=False,help="Generate kernel density estimator approximations of trinucleosome distributions")
	args = parser.parse_args()
	if not args.kde:
		s = ''
		lh_nucs = ind_lh_bound(sys.argv[1])
		for l in lh_nucs:
			s += `l` + " "
		s = s[:-1]
		print s
	else:
		gen_kde_probs()