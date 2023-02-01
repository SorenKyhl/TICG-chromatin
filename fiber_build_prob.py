#Module Fiber_builder_prob
#Created by Aria Coraor 12/6/19

#Build a fiber from specified NRLs, drawing purely from the three-body
#probability distribution for p(r_alpha,r_beta,r_alpha')

import numpy as np
import os
from scipy.interpolate.rbf import Rbf
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.stats import gaussian_kde as kde
from copy import deepcopy
import argparse
from vect_quat_util import *
from merlin import *
import molecule
import matplotlib
from time import time
from joblib import Parallel,delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#from scipy.spatial.transform import Rotation as R

#First nucleosome is at (0.0,0.0,0.0)

#Usage: python fiber_builder_prob.py <nnucl> <NRL> <output.dat> 
HOMEDIR = "/project2/depablo/coraor/fiber_builder/"

def gen_probs():
	"""Create probability profiles for all NRLs for r_alpha, r_beta,
		and r_alpha'. Write to hist_<NRL>.dat.npy here."""
	nrls = list(range(158,208))
	data = ["/project2/depablo/coraor/ris_repex_data/%d/" % ele for ele in nrls]

	first_conds = []
	second_conds = []    

	for i,root in enumerate(data):
		a = np.loadtxt(os.path.join(root,"0/dist_hist_0.dat"))
		b = np.loadtxt(os.path.join(root,"1/dist_hist_1.dat"))
		c = np.loadtxt(os.path.join(root,"2/dist_hist_2.dat"))
		d = np.concatenate((a,b,c))
		e = np.histogramdd(d)
		#Calculate base triplet free energies
		p = e[0]/len(d)
		g = -np.log(p)
		#energies.append(g)

		#Calculate ralpha' from ralpha, NRL
		#Collapse histograms in the rBeta dimension
		e_first = np.sum(e[0],axis=1)
		for j in range(len(e_first)):
			e_first[j,:] /= np.sum(e_first[j,:])

		g_first = -np.log(e_first)
		g_first = np.nan_to_num(g_first)
		
		#Calculate rbeta conditional on ralpha, ralpha' 
		#print "Histogram along rbeta axis: " + `e_first`
		#raw_input()
		
		#Create 1d count: normalize all rbeta counts for a given ralpha, ralpha'
		#so that sum(e) over all rbeta given fixed ralpha, ralpha' = 1
		e_second = deepcopy(e[0])
		for j in range(len(e_second)):
			for l in range(len(e_second[0][0])):
				e_second[j,:,l] /= np.sum(e_second[j,:,l])

		g_second = -np.log(e_second)
		g_second = np.nan_to_num(g_second)
		#shape = e_second.shape
		#print "Sum along rbeta axis for midpoint: " + `np.sum(e_second[shape[0]/2.0,:,shape[2]/2.0])`
		#print "Modified probability array for 2nd cond: " + `e_second`
		#raw_input()

		np.save("first_cond_%d.dat" % nrls[i],g_first)
		np.save("second_cond_%d.dat" % nrls[i],g_second)

		np.save("energies_%d.dat" % nrls[i],g)
		np.save("ranges_%d.dat" % nrls[i],e[1])
		print("Saved data from nrl %d" % nrls[i])
		first_conds.append(g_first)
		second_conds.append(g_second)
		#np.savetxt("hist_%d.dat" % nrls[i],e)
	

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
	nrls = list(range(158,208))

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

	print("Starting parallel kde generation")
	Parallel(n_jobs=-1)(delayed(_gen_kde_prob)(nrl) for nrl in nrls)

	
	


def _gen_kde_prob(nrl):
	"""Helper function to generate the kde probability for a given NRL. Helps
	parallelize this calculation across multiple processors."""
	print("Starting nrl %s" % repr(nrl))

	root = "/project2/depablo/coraor/ris_repex_data/%s/" % repr(nrl)
		
	a = np.loadtxt(os.path.join(root,"0/dist_hist_0.dat"))
	b = np.loadtxt(os.path.join(root,"1/dist_hist_1.dat"))
	c = np.loadtxt(os.path.join(root,"2/dist_hist_2.dat"))
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
	n_bins = 200
	XX,YY = np.meshgrid(np.linspace(np.min(d[:,0]),np.max(d[:,0]),n_bins),np.linspace(np.min(d[:,2]),np.max(d[:,2]),n_bins))

	#plt.contour(XX,YY,-np.log(norm_kernel(XX.ravel(),YY.ravel()).reshape((100,100))),cmap="viridis",levels = range(10))
	#plt.pcolormesh(XX,YY,-np.log(norm_kernel(XX.ravel(),YY.ravel()).reshape((100,100))),cmap="viridis")


	#raw kernel
	plot = False
	if plot:
		plt.clf()
		norm = matplotlib.colors.Normalize(vmin=10.0,vmax=30.0)
		Z_vals = (-np.log(kernel(np.vstack([XX.ravel(),YY.ravel()]))).reshape((n_bins,n_bins)))
		plt.contour(XX,YY,Z_vals,colors='black',levels = list(range(7,20)),linewidths=0.5)
		plt.pcolormesh(XX,YY,(-np.log(kernel(np.vstack([XX.ravel(),YY.ravel()]))).reshape((n_bins,n_bins))),cmap="viridis_r")

		cbar = plt.colorbar(norm = norm)
		cbar.set_clim(8.0,20.0)
		cbar.set_label("Probability density")
		cbar.set_label("Free energy (kT)")

		#Scatter a random subset
		#subset = np.take(d,np.random.randint(0,len(d),len(d)/10),axis=0)
		#plt.scatter(subset[:,0],subset[:,2],color="black",marker='+')

		plt.xlabel("$r_\\alpha$")
		plt.ylabel("$r_{\\alpha'}$")
		plt.title("NRL %s First Condition" % repr(nrl))

		plt.savefig("kde_%s.png" % repr(nrl),dpi=600)
	#Normalized kernel:
	if plot:
		plt.clf()
		Z_vals = -np.log(norm_kernel(XX.ravel(),YY.ravel())).reshape((n_bins,n_bins))
		norm = matplotlib.colors.Normalize(vmin=np.min(Z_vals),vmax=np.min(Z_vals)+20.0)
		plt.contour(XX,YY,Z_vals,colors='black',levels = list(range(int(np.floor(np.min(Z_vals))),20)),linewidths=0.5)
		plt.pcolormesh(XX,YY,Z_vals,cmap="viridis_r")

		cbar = plt.colorbar(norm = norm)
		cbar.set_clim(vmin=np.min(Z_vals),vmax=np.min(Z_vals)+20.0)
		cbar.set_label("Probability density")
		cbar.set_label("Free energy (kT)")


		plt.xlabel("$r_\\alpha$")
		plt.ylabel("$r_{\\alpha'}$")
		plt.title("NRL %s First Condition Normalized" % repr(nrl))

		plt.savefig("norm_kde_%s.png" % repr(nrl),dpi=600)

	#hquit()


		
	print("Calculating first free energy profile")

	start1 = time()
	
	#Z_vals = -np.log(norm_kernel(XX.ravel(),YY.ravel())).reshape((n_bins,n_bins))
	#First condition free energies
	g_first = -np.log(norm_kernel(XX.ravel(),YY.ravel())).reshape((n_bins,n_bins))
	print("First condition took %s seconds." % repr(round(time()-start1,3)))
	np.save("first_cond_%d_kde.dat" % nrl,g_first)
	ranges_ralpha = np.linspace(np.min(d[:,0]),np.max(d[:,0]),n_bins)
	ranges_ralphaprime = np.linspace(np.min(d[:,2]),np.max(d[:,2]),n_bins)
	ranges_rbeta = np.linspace(np.min(d[:,1]),np.max(d[:,1]),n_bins)

		
	#Second condition free energies: 
	#print "Making second kde"
	kernel_second = kde(d.T)
	kernel_2d = kde(d[:,0:3:2].T, bw_method = kernel_second.factor)

	#print "second kde done"
	normed_2kernel = lambda alpha, beta, alphaprime: kernel_second(np.vstack([alpha,beta,alphaprime]))/kernel_2d(np.vstack([alpha,alphaprime]))
	np.save("ranges_%d_kde.dat" % nrl,np.array([ranges_ralpha,ranges_rbeta,ranges_ralphaprime]))
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
	print("Calculating second free energy profile")
	start2 = time()
	g_second = -np.log(normed_2kernel(XX.ravel(),BB.ravel(),YY.ravel())).reshape((n_bins,n_bins,n_bins))
	print("second condition took %s seconds." % repr(round(time()-start2,3)))
	print("Saving profiles")
	#shape = e_second.shape
	#print "Sum along rbeta axis for midpoint: " + `np.sum(e_second[shape[0]/2.0,:,shape[2]/2.0])`
	#print "Modified probability array for 2nd cond: " + `e_second`
	#raw_input()
	#g_first = g_first.reshape((n_bins,n_bins))
	#g_second = g_second.reshape((n_bins,n_bins.n_bins))

	np.save("second_cond_%d_kde.dat" % nrl,g_second)

	#np.save("energies_%d.dat" % nrl,g)
	print("Saved data from nrl %d" % nrl)
	#np.savetxt("hist_%d.dat" % nrl,e)


def gen_dih_kde():
	"""Create probability profiles for all NRLs for Dihedral angles. Write to
	dih_energies_<NRL>.dat.npy"""
	nrls = list(range(158,208))
	data = ["/project2/depablo/coraor/ris_repex_data/%d/" % ele for ele in nrls]

	first_conds = []
	second_conds = []    
	n_bins = 200

	for i,root in enumerate(data):
		a = np.loadtxt(os.path.join(root,"0/diheangles.dat"))
		b = np.loadtxt(os.path.join(root,"1/diheangles.dat"))
		c = np.loadtxt(os.path.join(root,"2/diheangles.dat"))
		d = np.concatenate((a,b,c))
		kernel = kde(d)

		grid = np.linspace(-180.0,180.0,n_bins)
		p = kernel(grid)
		p /= np.sum(p)
		g = -np.log(p)
		#energies.append(g)
		g = np.nan_to_num(g)
		np.save("dih_energies_%d.npy" % nrls[i],g)
		np.save("dih_ranges_%d.npy" % nrls[i],grid)
		print("Saved dihedral data from nrl %d" % nrls[i])

	
def interp():
	"""Interpolate between datapoints of the dataset to create scipy.interpolate.Rbf
	objects trained on the datasets."""
	nrls = list(range(158,208))

	for i,nrl in enumerate(nrls):
		engs = np.load("energies_%d.dat.npy" % nrl)
		dists = np.load("ranges_%d.dat.npy" % nrl)
		mid_dists = np.zeros((3,len(dists[0])-1))
		for j,row in enumerate(mid_dists):
			for k in range(len(row)):
				row[k] = np.average(dists[j,k:k+2])
		x,y,z = np.meshgrid(mid_dists[0],mid_dists[1],mid_dists[2])
		q = lambda x: x.flatten()
		x = q(x)
		y = q(y)
		z = q(z)
		pos = np.column_stack((x,y,z))
		flat_engs = np.nan_to_num(q(engs))
		x = pos[:,0]
		y = pos[:,1]
		z = pos[:,2]
		intp = Rbf(x,y,z,flat_engs)
		return intp

def main():
	"""Build a fiber with args.nnucl nucleosomes, single nrl"""
	print("Generating %s nucleosomes." % repr(args.nnucl))

	first_dens = np.load(os.path.join(HOMEDIR,"first_cond_%d_kde.dat.npy" % args.NRL))
	second_dens = np.load(os.path.join(HOMEDIR,"second_cond_%d_kde.dat.npy" % args.NRL))
	dih_dens = np.loadtxt(os.path.join(HOMEDIR,"dih_energies_%d.dat" % args.NRL))
	first_probs = np.exp(-first_dens)
	second_probs = np.exp(-second_dens)
	dih_probs = np.exp(-dih_dens)

	first_probs = np.nan_to_num(first_probs)
	second_probs = np.nan_to_num(second_probs)
	dih_probs = np.nan_to_num(dih_probs)

	ranges = np.load(os.path.join(HOMEDIR,"ranges_%d_kde.dat.npy" % args.NRL))
	dih_ranges = np.loadtxt(os.path.join(HOMEDIR,"dih_ranges.dat"))

	#Drawing from dihedral now so excluded volume *could* be included

	print("All probabilities and ranges loaded.")
	print("Generating chain.")
	#Convert to cumulative probability distributions
	cum_prob = 0.0
	first_cum = np.cumsum(first_probs,axis=1) # For all ralpha, create a searchable ralpha'
	second_cum = np.cumsum(second_probs,axis=1) #for ralpha, ralpha', search rbeta
	dih_cum = np.cumsum(dih_probs)
	#first_cum = np.reshape(np.cumsum(first_probs.ravel()),first_probs.shape)
	#second_cum = np.reshape(np.cumsum(second_probs.ravel()),second_probs.shape)

	nucs = np.zeros((args.nnucl,3)) # x,y,z
	alpha_bets = np.zeros((args.nnucl,4)) #alpha, beta, alpha', dihedral
	ab_inds = np.zeros((args.nnucl,4),dtype=int) #Indices found from searching

	alpha_bets[0,0] = 150.0
	ab_inds[0,0] = np.searchsorted(ranges[0],150.0) - 1
	#For now, let nuc 2 be +150 Angstroms from nuc 1.	

	fvu0 = np.eye(3)
	quat0 = tu2rotquat(1e-5,[1,0,0])
	fvu = quat_fvu_rot(fvu0,quat0)
	quat = quat0


	#Set up first 3 along xy plane
	nucs[0] = np.array([0.0,0.0,0.0])
	nucs[1] = np.array([150.0,0.0,0.0])


	#Modify: don't go outside the following ranges
	min_abs = np.array([73.2,64.8,73.2])
	max_abs = np.array([143.5,206.9,143.5])

	min_inds = np.array([np.searchsorted(ranges[i],min_abs[i],side='left') for i in range(3)])
	max_inds = np.array([np.searchsorted(ranges[i],max_abs[i],side='right')-1 for i in range(3)])


	#Calculate real box bounds
	bounds = (7765.77*args.nnucl)**(1.0/3)*10

	#Still getting errors?
	bounds /= 2

	#Start drawing from probability distribution
	for i in range(args.nnucl-2):

		valid = np.all(ab_inds[i,1:] > min_inds) and np.all(ab_inds[i,1:] < max_inds)
		while(not valid):
			probs = np.random.rand(3)
			ab_inds[i,2] = np.searchsorted(first_cum[ab_inds[i,0]],probs[0])
			if ab_inds[i,2] == 200:
				ab_inds[i,2] -= 1
			ab_inds[i,1] = np.searchsorted(second_cum[ab_inds[i,0],:,ab_inds[i,2]],probs[1])
			ab_inds[i,3] = np.searchsorted(dih_cum,probs[2])
			if ab_inds[i,1] == 200:
				ab_inds[i,1] -= 1
			if ab_inds[i,3] == 200:
				ab_inds[i,3] -= 1
			valid = np.all(ab_inds[i,1:] > min_inds) and np.all(ab_inds[i,1:] < max_inds)

		#Create alpha_bet real values by using probability scaling of histograms
		if ab_inds[i,2] == 0:
			lever_1_distance = probs[0]/first_cum[ab_inds[i,0],0]
			alpha_bets[i,2] = lever_1_distance*ranges[2,1] + (1-lever_1_distance)*ranges[2,0]
		elif ab_inds[i,2] == len(ranges[2])-1:
			inv_lever_1_size = (1.0-first_cum[ab_inds[i,0],-2])
			lever_1_distance = (probs[0] - first_cum[ab_inds[i,0],-2])/inv_lever_1_size
			alpha_bets[i,2] = ranges[2,-1]*lever_1_distance + (1-lever_1_distance)*ranges[2,-2]
		else:
			#Calculate lever rule
			lever_1_size = first_cum[ab_inds[i,0],ab_inds[i,2]] - first_cum[ab_inds[i,0],ab_inds[i,2]-1]
			lever_1_distance = (probs[0] - first_cum[ab_inds[i,0],ab_inds[i,2]-1])/lever_1_size
			alpha_bets[i,2] = lever_1_distance*ranges[2,ab_inds[i,2]+1] + (
					1-lever_1_distance)*ranges[2,ab_inds[i,2]]

		#Real beta value
		if ab_inds[i,1] == 0:
			lever_2_distance = probs[1]/second_cum[ab_inds[i,0],0,ab_inds[i,2]]
			alpha_bets[i,1] = lever_2_distance*ranges[1,1] + (
				1-lever_2_distance)*ranges[1,0]

		elif ab_inds[i,1] == len(ranges[1])-1:
			inv_lever_2_size = (1.0-second_cum[ab_inds[i,0],-2,ab_inds[i,2]])
			lever_2_distance = (probs[1]-second_cum[ab_inds[i,0],-2,ab_inds[i,2]])/inv_lever_2_size
			alpha_bets[i,1] = lever_2_distance*ranges[1,-1] + (
				1-lever_2_distance)*ranges[1,-2]

		else:
			#print("Indices: %s" % repr(ab_inds[i,:]))
			lever_2_size = second_cum[ab_inds[i,0],ab_inds[i,1],
				ab_inds[i,2]] - second_cum[ab_inds[i,0],ab_inds[i,1]-1,
				ab_inds[i,2]]
			lever_2_distance = (probs[1] - second_cum[ab_inds[i,0],ab_inds[i,1]-1,
				ab_inds[i,2]])/lever_2_size
			alpha_bets[i,1] = lever_2_distance*ranges[1,ab_inds[i,1]+1] + (1-
					lever_2_distance)*ranges[1,ab_inds[i,1]]

		#Real Dihedral value
		if ab_inds[i,3] == 0:
			lever_dih_distance = probs[2]/(dih_cum[0])
			alpha_bets[i,3] = lever_dih_distance*dih_ranges[1] + (1-lever_dih_distance)*dih_ranges[0]
		else:
			lever_dih_size = dih_cum[ab_inds[i,3]] - dih_cum[ab_inds[i,3]-1]
			lever_dih_distance = (probs[2] - dih_cum[ab_inds[i,3]-1])/lever_dih_size
			alpha_bets[i,3] = lever_dih_distance*dih_ranges[ab_inds[i,3]+1] + (
					1-lever_dih_distance)*dih_ranges[ab_inds[i,3]]



		#Calculate realspace coordinates
		arg = ((alpha_bets[i,2]**2 + alpha_bets[i,1]**2 - alpha_bets[i,0]**2)/(2*alpha_bets[i,1]*alpha_bets[i,2]))
		if arg > 1 or arg < -1:
			alpha_bets[i,1] = (alpha_bets[i,0] + alpha_bets[i,2])*0.99
		within_bounds = False
		ang = m.acos((alpha_bets[i,2]**2 + alpha_bets[i,1]**2 - alpha_bets[i,0]**2)/(2*alpha_bets[i,1]*alpha_bets[i,2]))
		
		q = tu2rotquat(alpha_bets[i,3]*m.pi/180.0,fvu[0])
		quat = quat_multiply(q,quat)
		fvu = quat_fvu_rot(fvu0,quat)

		q = tu2rotquat(-ang,fvu[2])
		quat = quat_multiply(q,quat)
		fvu = quat_fvu_rot(fvu0,quat)

		nucs[i+2] = fvu[0] * alpha_bets[i,0] + nucs[i]
		within_bounds = np.all(nucs[i+1] < bounds) and np.all(nucs[i+1] > 0.0)


		dih_ang = alpha_bets[i,3]*m.pi/180.0
		while (not within_bounds):
			#Histogram bins may not align with geometry. If so, decrease rbeta.


			#undo rotation in prep for redo
			q = tu2rotquat(ang,fvu[2])
			quat = quat_multiply(q,quat)
			fvu = quat_fvu_rot(fvu0,quat)

			q = tu2rotquat(-dih_ang,fvu[0])
			quat = quat_multiply(q,quat)
			fvu = quat_fvu_rot(fvu0,quat)

			#Pick new ralpha, rbeta, dihedral angle from within allowable range

			alpha_bets[i,1] = np.random.rand()*(max_abs[1]-min_abs[1]) + min_abs[1]
			alpha_bets[i,3] = np.random.rand()*360.0-180.0
			alpha_bets[i,0] = np.random.rand()*(max_abs[0]-min_abs[0]) + max_abs[0]
			alpha_bets[i-1,2] = alpha_bets[i,0]

			arg = ((alpha_bets[i,2]**2 + alpha_bets[i,1]**2 - alpha_bets[i,0]**2)/(2*alpha_bets[i,1]*alpha_bets[i,2]))
			if arg > 1 or arg < -1:
				alpha_bets[i,1] = (alpha_bets[i,0] + alpha_bets[i,2])*0.99
			ang = m.acos((alpha_bets[i,2]**2 + alpha_bets[i,1]**2 - alpha_bets[i,0]**2)/(2*alpha_bets[i,1]*alpha_bets[i,2]))
			dih_ang = alpha_bets[i,3]*m.pi/180.0
			


			q = tu2rotquat(dih_ang,fvu[0])
			quat = quat_multiply(q,quat)
			fvu = quat_fvu_rot(fvu0,quat)

			q = tu2rotquat(-ang,fvu[2])
			quat = quat_multiply(q,quat)
			fvu = quat_fvu_rot(fvu0,quat)

			nucs[i+1] = fvu[0] * alpha_bets[i,0] + nucs[i]
			#print(repr(nucs[i+1]))
			within_bounds = np.all(nucs[i+1] < bounds) and np.all(nucs[i+1] > 0.0)

		

		if i < args.nnucl -1:
			ab_inds[i+1,0] = ab_inds[i,2]
			alpha_bets[i+1,0] = alpha_bets[i,2]

	print("ab_inds: " + repr(ab_inds))
	print("Distances: " + repr(alpha_bets))
	print("Positions: " + repr(nucs))

	#Reduce by factor of 10...
	nucs /= 10.0
	np.savetxt(args.o,nucs)
	
	print("saving xyz")
	with open(args.o + '.xyz','w') as f:
		f.write("%s\n" % repr(args.nnucl))
		f.write("atoms\n")
		for i,pos in enumerate(nucs):
			f.write("%s\t%s\t%s\t%s\t%s\n" % (repr(i),repr(pos[0]),repr(pos[1]),repr(pos[2]),repr(int(i<2500))))
	print(".xyz saved to %s" % args.o)


def prep_ticg():
	"""Redone fiber builder to prepare for TICG. Build a fiber according to the
	normal distribution, except alter dihedral until system is in-bounds if accidentally
	out of bounds."""

	first_dens = np.load(os.path.join(HOMEDIR,"first_cond_%d_kde.dat.npy" % args.NRL))
	second_dens = np.load(os.path.join(HOMEDIR,"second_cond_%d_kde.dat.npy" % args.NRL))
	dih_dens = np.loadtxt(os.path.join(HOMEDIR,"dih_energies_%d.dat" % args.NRL))
	first_probs = np.exp(-first_dens)
	second_probs = np.exp(-second_dens)
	dih_probs = np.exp(-dih_dens)

	first_probs = np.nan_to_num(first_probs)
	second_probs = np.nan_to_num(second_probs)
	dih_probs = np.nan_to_num(dih_probs)

	ranges = np.load(os.path.join(HOMEDIR,"ranges_%d_kde.dat.npy" % args.NRL))
	dih_ranges = np.loadtxt(os.path.join(HOMEDIR,"dih_ranges.dat"))

	#Drawing from dihedral now so excluded volume *could* be included


	#Convert to cumulative probability distributions
	cum_prob = 0.0
	first_cum = np.cumsum(first_probs,axis=1) # For all ralpha, create a searchable ralpha'
	second_cum = np.cumsum(second_probs,axis=1) #for ralpha, ralpha', search rbeta
	dih_cum = np.cumsum(dih_probs)
	#first_cum = np.reshape(np.cumsum(first_probs.ravel()),first_probs.shape)
	#second_cum = np.reshape(np.cumsum(second_probs.ravel()),second_probs.shape)

	nucs = np.zeros((args.nnucl,3)) # x,y,z
	alpha_bets = np.zeros((args.nnucl,4)) #alpha, beta, alpha', dihedral
	ab_inds[0,0] = np.searchsorted(ranges[0],150.0) - 1
	#For now, let nuc 2 be +150 Angstroms from nuc 1.	

	fvu0 = np.eye(3)
	quat0 = tu2rotquat(1e-5,[1,0,0])
	fvu = quat_fvu_rot(fvu0,quat0)
	quat = quat0

	#In order to allow any NRL distribution, all segments must be within the
	#minimum and maximum bounds:
	min_abs = np.array([73.2,64.8,73.2])
	max_abs = np.array([143.5,206.9,143.5])


	#Set up first 3 along xy plane
	nucs[0] = np.array([0.0,0.0,0.0])
	probs = np.random.rand((args.nnucl-1)*3).reshape((args.nnucl-1,3))
	nucs[1] = np.array([np.random.rand()*(max_abs-min_abs)+min_abs,0.0,0.0])

	#Draw random configurations within this range.
	#Start drawing from probability distribution
	for i in range(2,args.nnucl):

		probs = np.random.rand(3)
		ab_inds[i,2] = np.searchsorted(first_cum[ab_inds[i,0]],probs[0])
		print(repr(ab_inds[i,:]))
		if ab_inds[i,2] == 200:
			ab_inds[i,2] -= 1
		ab_inds[i,1] = np.searchsorted(second_cum[ab_inds[i,0],:,ab_inds[i,2]],probs[1])
		ab_inds[i,3] = np.searchsorted(dih_cum,probs[2])
		if ab_inds[i,1] == 200:
			ab_inds[i,1] -= 1
		if ab_inds[i,3] == 200:
			ab_inds[i,3] -= 1

		#Create alpha_bet real values by using probability scaling of histograms
		if ab_inds[i,2] == 0:
			lever_1_distance = probs[0]/first_cum[ab_inds[i,0],0]
			alpha_bets[i,2] = lever_1_distance*ranges[2,1] + (1-lever_1_distance)*ranges[2,0]
		elif ab_inds[i,2] == len(ranges[2])-1:
			inv_lever_1_size = (1.0-first_cum[ab_inds[i,0],-2])
			lever_1_distance = (probs[0] - first_cum[ab_inds[i,0],-2])/inv_lever_1_size
			alpha_bets[i,2] = ranges[2,-1]*lever_1_distance + (1-lever_1_distance)*ranges[2,-2]
		else:
			#Calculate lever rule
			lever_1_size = first_cum[ab_inds[i,0],ab_inds[i,2]] - first_cum[ab_inds[i,0],ab_inds[i,2]-1]
			lever_1_distance = (probs[0] - first_cum[ab_inds[i,0],ab_inds[i,2]-1])/lever_1_size
			alpha_bets[i,2] = lever_1_distance*ranges[2,ab_inds[i,2]+1] + (
					1-lever_1_distance)*ranges[2,ab_inds[i,2]]

		#Real beta value
		if ab_inds[i,1] == 0:
			lever_2_distance = probs[1]/second_cum[ab_inds[i,0],0,ab_inds[i,2]]
			alpha_bets[i,1] = lever_2_distance*ranges[1,1] + (
				1-lever_2_distance)*ranges[1,0]

		elif ab_inds[i,1] == len(ranges[1])-1:
			inv_lever_2_size = (1.0-second_cum[ab_inds[i,0],-2,ab_inds[i,2]])
			lever_2_distance = (probs[1]-second_cum[ab_inds[i,0],-2,ab_inds[i,2]])/inv_lever_2_size
			alpha_bets[i,1] = lever_2_distance*ranges[1,-1] + (
				1-lever_2_distance)*ranges[1,-2]

		else:
			print("Indices: %s" % repr(ab_inds[i,:]))
			lever_2_size = second_cum[ab_inds[i,0],ab_inds[i,1],
				ab_inds[i,2]] - second_cum[ab_inds[i,0],ab_inds[i,1]-1,
				ab_inds[i,2]]
			lever_2_distance = (probs[1] - second_cum[ab_inds[i,0],ab_inds[i,1]-1,
				ab_inds[i,2]])/lever_2_size
			alpha_bets[i,1] = lever_2_distance*ranges[1,ab_inds[i,1]+1] + (1-
					lever_2_distance)*ranges[1,ab_inds[i,1]]

		#Real Dihedral value
		if ab_inds[i,3] == 0:
			lever_dih_distance = probs[2]/(dih_cum[0])
			alpha_bets[i,3] = lever_dih_distance*dih_ranges[1] + (1-lever_dih_distance)*dih_ranges[0]
		else:
			lever_dih_size = dih_cum[ab_inds[i,3]] - dih_cum[ab_inds[i,3]-1]
			lever_dih_distance = (probs[2] - dih_cum[ab_inds[i,3]-1])/lever_dih_size
			alpha_bets[i,3] = lever_dih_distance*dih_ranges[ab_inds[i,3]+1] + (
					1-lever_dih_distance)*dih_ranges[ab_inds[i,3]]


		#Calculate realspace coordinates
		arg = ((alpha_bets[i,2]**2 + alpha_bets[i,1]**2 - alpha_bets[i,0]**2)/(2*alpha_bets[i,1]*alpha_bets[i,2]))

		#Histogram bins may not align with geometry. If so, decrease rbeta.
		if arg > 1:
			alpha_bets[i,1] = (alpha_bets[i,0] + alpha_bets[i,2])*0.99
		
		ang = m.acos((alpha_bets[i,2]**2 + alpha_bets[i,1]**2 - alpha_bets[i,0]**2)/(2*alpha_bets[i,1]*alpha_bets[i,2]))

		q = tu2rotquat(alpha_bets[i,3],fvu[0])
		quat = quat_multiply(q,quat)
		fvu = quat_fvu_rot(fvu0,quat)

		q = tu2rotquat(-ang,fvu[2])
		quat = quat_multiply(q,quat)
		fvu = quat_fvu_rot(fvu0,quat)

		nucs[i+1] = fvu[0] * alpha_bets[i,0] + nucs[i]
		

		if i < args.nnucl -1:
			ab_inds[i+1,0] = ab_inds[i,2]
			alpha_bets[i+1,0] = alpha_bets[i,2]

	print("ab_inds: " + repr(ab_inds))
	print("Distances: " + repr(alpha_bets))
	print("Positions: " + repr(nucs))
	np.savetxt(args.o,nucs)



def energy(r_vect):
	"""Calculates the energy of the fiber purely from trinucleosome results.

	Parameters:
		r_vect: *list*
				list of utils.Atoms, nucleosomes in a fiber.
	Returns:
		eng: *float*
				Energy from first and second conditions on the fiber."""
	try:
		x = repr(args)
	except:
		class struct(object):
			pass
		args = struct()
		args.NRL = 167
	first_dens = np.load("first_cond_%d.dat.npy" % args.NRL)
	second_dens = np.load("second_cond_%d.dat.npy" % args.NRL)
	dih_dens = np.loadtxt("dih_energies_%d.dat" % args.NRL)


	''' #Not really necessary. 
	#Restructure energies per trapezoidal rule, extending trapezoids out to bin
	#edges.
	first_ext = np.zeros(len(first_dens)+2,len(first_dens[0])+2)
	second_ext = np.zeros(len(second_dens)+2,len(second_dens[0])+2)
	first_ext[1:-1,1:-1] = first_dens
	second_dens[1:-1,1:-1,1:-1] = second_dens

	#Extrapolate edges of first and second for smooth calculation
	first_ext[0,1:-1] = first_ext[1]

	'''
	#Figure out dih later


	#Gotta renormalize so the boltzman-averaged energy from every alpha is the same
	#So the MC only chooses an alpha based on the prior trinucleosome

	ranges = np.load("ranges_%d_kde.dat.npy" % args.NRL)
	dih_ranges = np.loadtxt("dih_ranges.dat")
	#xyz
	pos = np.zeros((len(r_vect),3))
	for i,atom in enumerate(r_vect):
		pos[i] = atom.pos
		#pos[i,0] = atom.x
		#pos[i,1] = atom.y
		#pos[i,2] = atom.z
	print(repr(pos))
	#alpha, beta, alpha'
	alpha_bets = np.zeros(pos.shape)
	alpha_bets[:-1,0] = np.linalg.norm(pos[1:] - pos[:-1],axis=1)
	alpha_bets[:-2,1] = np.linalg.norm(pos[2:] - pos[:-2],axis=1)
	alpha_bets[:-2,2] = alpha_bets[1:-1,0]
	alpha_bets = alpha_bets[:-2]
	mid_ranges =  (ranges[:,:-1] + ranges[:,1:])/2.0
	first_intp = RGI((mid_ranges[0],mid_ranges[2]),first_dens)
	second_intp = RGI((mid_ranges[0],mid_ranges[1],mid_ranges[2]),second_dens)

	#Use scipy to interpolate, not manual.
	'''
	#Assume bins are linear between beginning and end
	alpha_large_lever = (alpha_bets[:,0]-ranges[0,0])/(ranges[0,-1]-ranges[0,0])*12.0
	beta_large_lever = (alpha_bets[:,1]-ranges[1,0])/(ranges[1,-1]-ranges[1,0])*12.0
	alphaprime_large_lever = (alpha_bets[:,0]-ranges[2,0])/(ranges[2,-1]-ranges[2,0])*12.0

	#Create index array for all of these
	alpha_indless = np.floor(alpha_large_lever).astype(int)
	alpha_indmore = np.ceil(alpha_large_lever).astype(int)
	beta_indless = np.floor(beta_large_lever).astype(int)
	beta_indmore = np.ceil(beta_large_lever).astype(int)
	alpha_indless = np.floor(alpha_large_lever).astype(int)
	alpha_indmore = np.ceil(alpha_large_lever).astype(int)
	'''
	print("alpha_bets reshape: %s" % repr(alpha_bets[:,0:3:2]))
	print("Input shape: %s" % repr((np.array((mid_ranges[0],mid_ranges[2])).shape)))
	print("mid ranges, min max: %s, %s|| %s, %s" % (repr(mid_ranges[0][0]),
		repr(mid_ranges[0][-1]), repr(mid_ranges[2][0]),repr(mid_ranges[2][-1])))
	eng1 = first_intp(alpha_bets[:,0:3:2])
	eng2 = second_intp(alpha_bets)

	total_eng = np.sum(eng1) + np.sum(eng2)
	return total_eng

def kde_energy(r_vect):
	"""Calculates the energy of the fiber purely from trinucleosome results.

	Parameters:
		r_vect: *np.array*
				nx3 array of nucleosome positions.
	Returns:
		eng: *float*
				Energy from first and second conditions on the fiber."""
	try:
		x = repr(args)
	except:
		class struct(object):
			pass
		args = struct()
		args.NRL = 167
	first_dens = np.load("first_cond_%d_kde.dat.npy" % args.NRL)
	second_dens = np.load("second_cond_%d_kde.dat.npy" % args.NRL)
	dih_dens = np.loadtxt("dih_energies_%d.dat" % args.NRL)


	''' #Not really necessary. 
	#Restructure energies per trapezoidal rule, extending trapezoids out to bin
	#edges.
	first_ext = np.zeros(len(first_dens)+2,len(first_dens[0])+2)
	second_ext = np.zeros(len(second_dens)+2,len(second_dens[0])+2)
	first_ext[1:-1,1:-1] = first_dens
	second_dens[1:-1,1:-1,1:-1] = second_dens

	#Extrapolate edges of first and second for smooth calculation
	first_ext[0,1:-1] = first_ext[1]

	'''
	#Figure out dih later


	#Gotta renormalize so the boltzman-averaged energy from every alpha is the same
	#So the MC only chooses an alpha based on the prior trinucleosome

	ranges = np.load("ranges_%d_kde.dat.npy" % args.NRL)
	dih_ranges = np.loadtxt("dih_ranges.dat")
	#xyz
	pos = r_vect
	#pos = np.zeros((len(r_vect),3))
	#for i,atom in enumerate(r_vect):
	#	pos[i] = atom.pos
		#pos[i,0] = atom.x
		#pos[i,1] = atom.y
		#pos[i,2] = atom.z
	#print `pos`
	#alpha, beta, alpha'
	alpha_bets = np.zeros(pos.shape)
	alpha_bets[:-1,0] = np.linalg.norm(pos[1:] - pos[:-1],axis=1)
	alpha_bets[:-2,1] = np.linalg.norm(pos[2:] - pos[:-2],axis=1)
	alpha_bets[:-2,2] = alpha_bets[1:-1,0]
	alpha_bets = alpha_bets[:-2]
	first_intp = RGI((ranges[0],ranges[2]),first_dens)
	second_intp = RGI((ranges[0],ranges[1],ranges[2]),second_dens)

	#Use scipy to interpolate, not manual.
	'''
	#Assume bins are linear between beginning and end
	alpha_large_lever = (alpha_bets[:,0]-ranges[0,0])/(ranges[0,-1]-ranges[0,0])*12.0
	beta_large_lever = (alpha_bets[:,1]-ranges[1,0])/(ranges[1,-1]-ranges[1,0])*12.0
	alphaprime_large_lever = (alpha_bets[:,0]-ranges[2,0])/(ranges[2,-1]-ranges[2,0])*12.0

	#Create index array for all of these
	alpha_indless = np.floor(alpha_large_lever).astype(int)
	alpha_indmore = np.ceil(alpha_large_lever).astype(int)
	beta_indless = np.floor(beta_large_lever).astype(int)
	beta_indmore = np.ceil(beta_large_lever).astype(int)
	alpha_indless = np.floor(alpha_large_lever).astype(int)
	alpha_indmore = np.ceil(alpha_large_lever).astype(int)
	'''
	print("alpha_bets reshape: %s" % repr(alpha_bets[:,0:3:2]))
	eng1 = first_intp(alpha_bets[:,0:3:2])
	eng2 = second_intp(alpha_bets)
	print("First eng: %s, second eng: %s" % (repr(eng1),repr(eng2)))
	total_eng = np.sum(eng1) + np.sum(eng2)
	return total_eng

def rbf_fit(r_vect):
	"""Create an rbf fit of the 3d FES."""
	first_dens = np.load("first_cond_%d.dat.npy" % args.NRL)
	second_dens = np.load("second_cond_%d.dat.npy" % args.NRL)
	dih_dens = np.loadtxt("dih_energies_%d.dat" % args.NRL)
	first_probs = np.exp(-first_dens)
	second_probs = np.exp(-second_dens)

	first_probs = np.nan_to_num(first_probs)
	second_probs = np.nan_to_num(second_probs)

	ranges = np.load("ranges_%d.dat.npy" % args.NRL)
	dih_ranges = np.loadtxt("dih_ranges.dat")

	mid_ranges =  (ranges[:,:-1] + ranges[:,1:])/2.0
	X,Y,Z = np.meshgrid(mid_ranges[0],mid_ranges[1],mid_ranges[2])

	XX,ZZ = np.meshgrid(mid_ranges[0],mid_ranges[2])

	for i in range(len(first_dens.ravel())):
		if first_dens.ravel()[i] > 1e100:
			first_dens.ravel()[i] = 200
			#first_dens.ravel()[i] = first_dens.ravel()[i]**0.5

	for i in range(len(second_dens.ravel())):
		if second_dens.ravel()[i] > 1e100:
			second_dens.ravel()[i] = 200

	first_rbf = Rbf(XX.flatten(),ZZ.flatten(),first_dens.flatten())	
	second_rbf = Rbf(X.flatten(),Y.flatten(),Z.flatten(),second_dens.flatten())

	pos = np.zeros((len(r_vect),3))
	for i,atom in enumerate(r_vect):
		pos[i] = atom.pos

	alpha_bets = np.zeros(pos.shape)
	alpha_bets[:-1,0] = np.linalg.norm(pos[1:] - pos[:-1],axis=1)
	alpha_bets[:-2,1] = np.linalg.norm(pos[2:] - pos[:-2],axis=1)
	alpha_bets[:-2,2] = alpha_bets[1:-1,0]

	alpha_bets = alpha_bets[:-2]

	
	#Print values
	print("Values: " + repr(first_rbf.di))
	print("xi: " + repr(first_rbf.xi))
	print("epsilon: " + repr(first_rbf.nodes))

	print("Second: ")
	print("Values: " + repr(second_rbf.di))
	print("xi: " + repr(second_rbf.xi))
	print("epsilon: " + repr(second_rbf.nodes))
	

	#np.savetxt("rbf/first_xi_%s.dat" % args.NRL,first_rbf.xi)
	#np.savetxt("rbf/first_di_%s.dat" % args.NRL,first_rbf.nodes)
	#np.savetxt("rbf/first_epsilon_%s.dat" % args.NRL,np.array([first_rbf.epsilon]))

	#np.savetxt("rbf/second_xi_%s.dat" % args.NRL,second_rbf.xi)
	#np.savetxt("rbf/second_di_%s.dat" % args.NRL,second_rbf.nodes)
	#np.savetxt("rbf/second_epsilon_%s.dat" % args.NRL,np.array([second_rbf.epsilon]))

	'''
	#Turn into meshgrid for rbf
	print "First energy:"
	print `first_rbf(XX.ravel(),ZZ.ravel())`
	print `first_dens.ravel()`

	'''
	'''
	print "second energy:"
	print `second_rbf(X.ravel(),Y.ravel(),Z.ravel())`
	print `second_dens.ravel()`
	raw_input()
	print `second_dens[0]`
	print `second_dens[-1,:,-1]`
	raw_input()
	'''

	#Create an energy plot
	n_axis = 200
	plot_XX,plot_ZZ = np.meshgrid(np.linspace(np.min(XX),np.max(XX),n_axis),np.linspace(np.min(ZZ),np.max(ZZ),n_axis))
	plt.contour(plot_XX,plot_ZZ,first_rbf(plot_XX.ravel(),plot_ZZ.ravel()).reshape((n_axis,n_axis)),cmap="viridis",levels = list(range(10)))
	plt.pcolormesh(plot_XX,plot_ZZ,first_rbf(plot_XX.ravel(),plot_ZZ.ravel()).reshape((n_axis,n_axis)),cmap="viridis")
	cbar = plt.colorbar(norm = matplotlib.colors.Normalize(vmin=0.0,vmax=5.0))
	cbar.set_clim(0.0,5.0)
	cbar.set_label("Free Energy (kT)")


	plt.xlabel("$r_\\alpha$")
	plt.ylabel("$r_{\\alpha'}$")
	plt.title("NRL 187 First Condition")

	plt.savefig("first_cond_plotted.png",dpi=600)

	plt.clf()
	#1D-slice
	horiz = [ranges[0,4]]*len(first_dens[4,:])
	plt.scatter(first_dens[4,:],horiz,marker="o",color="blue")
	horiz2 = [ranges[0,4]]*n_axis
	print(repr(horiz2))
	plt.plot(first_rbf(np.array(horiz2),np.linspace(ranges[2,0],ranges[2,-1],n_axis)),horiz2,marker='',linestyle="-",color="red")
	plt.ylim(0.0,10.0)
	plt.savefig("first_cond_slice.png",dpi=600)


	first_manual = _build_rbf_interpolator(first_rbf.xi,first_rbf.nodes,first_rbf.epsilon)
	second_manual = _build_rbf_interpolator(second_rbf.xi,second_rbf.nodes,second_rbf.epsilon)

	eng1 = first_rbf(alpha_bets[:,0],alpha_bets[:,2])
	eng2 = second_rbf(alpha_bets[:,0],alpha_bets[:,1],alpha_bets[:,2])

	man_eng1 = first_manual(alpha_bets[:,0:3:2])
	man_eng2 = second_manual(alpha_bets)

	print("Interpolation error in the first energy: " + repr(man_eng1-np.sum(eng1)))
	print("Interpolation error in the second energy: " + repr(man_eng2-np.sum(eng2)))


	total_eng = np.sum(eng1) + np.sum(eng2)
	return total_eng


def _build_rbf_interpolator(xi,nodes,epsilon):
	"""Returns a callable of the form f(vect_x)
	which calculates the rbf interpolation centered at positions xi with
	magnitude nodes. Uses multiquadric interpolation.

	Parameters:
		xi: *np.array*
			array of dxN floats marking the coordinates of the knots. d is the 
			dimensionality of the vector space and N is the number of datapoints.
		nodes: *np.array*
			array of N floats, the magnitude of the knots. """

	def rbf(vect):
		s = 0.0
		for j in range(len(vect)):
			for i in range(xi.shape[1]):
				dist = np.linalg.norm(vect[j]-xi[:,i])
				s += nodes[i]*np.sqrt((dist/epsilon)**2+1)
		return s

	return rbf


def eng_load():
	"""Return a list of first, second, dih, ranges from fiber analysis."""
	nrls = list(range(158,208))
	firsts = []
	seconds = []
	ranges = []
	dihs = []
	print("Loading energy files.")
	for nrl in nrls:
		first_dens = np.load(os.path.join(HOMEDIR,"first_cond_%d_kde.dat.npy" % nrl))
		second_dens = np.load(os.path.join(HOMEDIR,"second_cond_%d_kde.dat.npy" % nrl))
		dih_dens = np.loadtxt(os.path.join(HOMEDIR,"dih_energies_%d.dat" % nrl))
		ran = np.load(os.path.join(HOMEDIR,"ranges_%d_kde.dat.npy" % nrl))
		firsts.append(first_dens)
		seconds.append(second_dens)
		dihs.append(dih_dens)
		ranges.append(ran)

	return np.array(firsts),np.array(seconds),np.array(dihs),np.array(ranges)

def second_cond_plot():
	"""Plot second_condition free energy as a function of rbeta. Create two sets
	of line plots:

	1. Line plots of F(rbeta) given ralpha = ralphaprime = lambda*rbeta
	2. Surface plots, F(rbeta, ralpha). For each ralpha, use distribution from
	maximum likelihood ralphaprime.
	"""
	firsts,seconds,dihs,ranges = eng_load()
	rbetas = [] # rbeta for each NRL, nrl x rbeta
	ralphas = [] # ralpha list: lambda x nrl x rbeta
	ralphaprimes = [] #ralphaprime list
	lambdas = np.array([1.3,1.1,1,0.9,0.7])

	#Calculate ralphas for each lambda value.
	for i,nrl in enumerate(nrls):
		rbetas.append(ranges[i][1])
	rbetas = np.array(rbetas)
	#rbeta inds are just range(len(rbetas))
	ralphas = np.array([lambd*rbeta for lambd in lambdas]) #3d array

	ralpha_inds = []
	for lambd,ralpha in enumerate(ralphas):
		nrl_inds = []
		for nrl,ralph in enumerate(ralpha):
			comp_ralph = ranges[nrl][0]
			inds = (ralph-comp_ralph[0])/(comp_ralph[1]-comp_ralph[0])
			inds = np.array(np.round(inds),dtype=int)
			nrl_inds.append(inds)
		ralpha_inds.append(nrl_inds)

	energies = [] # energy list: lambda x nrl x rbeta
	for lambd,ralpha in enumerate(ralphas):
		nrl_engs = []
		for nrl,ralph in enumerate(ralpha):
			rbeta = rbetas[nrl]

			comp_ralph = ranges[nrl][0]
			inds = (ralph-comp_ralph[0])/(comp_ralph[1]-comp_ralph[0])
			inds = np.array(np.round(inds),dtype=int)
			nrl_engs.append(inds)
		ralpha_inds.append(nrl_engs)


	rap_inds = []
	#Get ralphaprimes from alpha span, argmin of energy for each alpha.
	for i,nrl in enumerate(nrls):
		rbeta = rbetas[i]
		ralpha = ranges[i][0]
		#2d iteration over rbeta, ralpha.
		for j in range(len(rbeta)):
			for j in range(len(ralpha)):
				#ralphaprime index is the index of the lowest 
				pass






if __name__ == '__main__':
	#gen_probs()
	#gen_dih()
	parser = argparse.ArgumentParser()

	parser.add_argument('-kde','--kde',action='store_const',const=True,default=False,help="Generate kernel density estimator approximations of trinucleosome distributions")
	parser.add_argument('-n','--nnucl',type=int,help="Number of nucleosomes")
	parser.add_argument('-nrl','--NRL',type=int,help="Nucleosome repeat length")
	parser.add_argument('-o','--outputfile',dest='o',type=str,default = 'fiber.out',help='Save config in outputfile.')
	parser.add_argument('-ef','--energyfile',dest='ef',default=False,help="Calculate the energy of the last configuration in this lammpstrj")
	parser.add_argument('-rbf','--rbf',dest='rbf',action='store_const', const=True, default=False,help='Use a radial basis function fit')
	args = parser.parse_args()


	if args.kde:
		gen_kde_probs()
	elif not args.ef:
		main()
	else:
		if "lammpstrj" in args.ef:
			r_vect = files.read_lammpstrj(args.ef).frames[-1]
		else:
			r_vect,bonds,angles = files.read_input(args.ef,read_atoms=True)
		r_vect = [ele for ele in r_vect if ele.mytype == 1]
		r_vect.sort(key= lambda x: int(x.index))

		if not args.rbf:
			print("Final energy: " + repr(energy(r_vect)) + "kT")
		else:
			print("Final rbf energy: " + repr(rbf_fit(r_vect)) + "kT")
	

















