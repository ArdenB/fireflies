"""A function to make a histogram of the data.
args:
	rm_val value to be removed
To DO
	Check the range and add percentiles to the graph
"""
#==============================================================================

__title__ = "Hist Maker"
__author__ = "Arden Burrell"
__version__ = "1.1 (11.03.2018)"
__email__ = "arden.burrell@gmail.com"

#==============================================================================
import numpy as np
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import ipdb
import pdb
import warnings as warn
# =============================================================================
def _RangeCheck(array, mapdet):
	""" 
	check the range of the data, and see if there is a good reason to
	move the cmins

	"""
	perc_vals = np.percentile(array, [1, 99])
	permax = np.absolute(np.max(np.percentile(array, [1, 99])))
	setmax = np.max(np.absolute([mapdet.cmin, mapdet.cmax]))
	if setmax < permax:
		print ( 
			"the set values for the plot do not incapulate the 99th percentile\n", 
			"set values: ", mapdet.cmin, mapdet.cmax, "99th percnt: ", perc_vals)
		Procede = raw_input('Do you want to procede anyway? (Y/N): ')
		pro = ["yes", "y", "Y", "YES"]
		if Procede in pro:
			pass
		else:
			raise ValueError("Sanity checked to prevent me masking out too much")
	elif setmax > 2*permax:
		if ((".p" in mapdet.var) or ("r.squared" in mapdet.var)) and setmax==1:
			pass
		else:
			print( 
				"the set values for the plot are >2x 99th percentile, ", 
				"set values: ", mapdet.cmin, mapdet.cmax, "99th percnt: ", 
				perc_vals
				)
			print("percentile:\n [0.1, 1, 10, 25, 50, 75, 90, 99, 99.9]\n")
			print (np.percentile(array, [0.1,1, 10, 25, 50, 75, 90, 99, 99.9]))
			# pdb.set_trace()
	else:
		#In the future this can me used to mark percentiles on graphs?
		pass

# =============================================================================
def histmaker(
	results, rundet, mapdet, vmin=None, vmax=None, mlist=None, color='darkviolet', 
	bins=1000, axvline=None
	):
	""" A function to make a histogram of the values
	Args:
		results
		vmin, value below which values are set to NAN and excluded
		vmax, value above which values are set to NAN and excluded
		mlist, list of values to be masked
		color
		"""
	# ========== class check input variables ==========
	if type(rundet).__name__ != 'tssrun':
		raise TypeError("rundet must be of class tssrun")
	if type(mapdet).__name__ != 'mapclass':
		raise TypeError("mapdet must be of class mapclass")

	# ========== pull out the relevant column ==========
	if not (mapdet.mask is None):
		# Mask out the data so it only shows drylands
		array = []
		for line in range(1, results.shape[0]-1):
			if mapdet.mask[int(results[line, 0]), int(results[line, 1])]==1:
				array.append(results[line, mapdet.column])
		array = np.array(array)
	else:
		array = results[1:, mapdet.column].copy()
	if np.nansum(array) == 0:
		raise ValueError("The results and column being tested has a nansum of zero")

	# ========== set excluded values to NAN ==========
	if not (vmin is None):
		array[array<=vmin]=np.NAN
	if not (vmax is None):
		array[array>=vmax]=np.NAN
	if not (mlist is None):
		if type(mlist) == int or type(mlist) == float:
			mlist = [mlist]
		elif type(mlist)==list:
			pass
		else:
			raise ValueError("mlist must be a float, int or a list of same")
		for msk in mlist:
			array[array==msk]=np.NAN
	
	# ========== Remove the excluded values ==========
	if array[~np.isnan(array.astype(float))].size == 0:
		warn.warn("something is wrong, the size of the array is zero. going interactive")
		ipdb.set_trace()
	else:
		array = array[~np.isnan(array.astype(float))]
	
	# ========== Sanity check the color ranges ===========
	_RangeCheck(array, mapdet)

	# ========== Setup the sns plot ==========
	# plt.figure(num="Histogram of %s" % mapdet.var)
	sns.set(style="ticks")
	sns.set_context("talk")

	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 28}

	matplotlib.rc('font', **font)
	# ipdb.set_trace()
	if mapdet.hist_b:
		f, (ax_box, ax_hist) = plt.subplots(
			2, figsize=(18,9), sharex=True, gridspec_kw={"height_ratios": (.15, .85)},
			num=("Histogram of %s" % mapdet.var))
		# +++++ Add the box plot ++++++
		# old version was just (array, ....). data= added to fix bug
		sns.boxplot(data=array, ax=ax_box, color=color, fliersize=1)
	else:
		f, ax_hist = plt.subplots(figsize=(15,10), 
			# gridspec_kw={"height_ratios": (.15, .85)},
			num=("Histogram of %s" % mapdet.var))
	# +++++ Add the histogram ++++++
	# ipdb.set_trace()
	sns.distplot(array, ax=ax_hist, bins=bins, kde=False, color=color)

	# +++++ Add an arbitary line ++++++
	if not (axvline is None):
		plt.axvline(axvline, color="k", linestyle="--", alpha=0.25)

	if mapdet.hist_b:
		ax_box.set(yticks=[])
		sns.despine(ax=ax_box, left=True)

	sns.despine(ax=ax_hist)
	# ipdb.set_trace()
	if not (mapdet.hist_x is None):
		# plt.xlabel = mapdet.hist_x
		ax_hist.set_xlabel(mapdet.hist_x, fontweight= 'bold', fontsize='medium')  

	if not (mapdet.hist_y is None):
		ax_hist.set_ylabel(mapdet.hist_y, fontweight= 'bold', fontsize='medium')  
	# ========== save the plot ==========
	# ipdb.set_trace()
	for t in ax_hist.get_yticklabels():
		t.set_fontsize(16)

	# Adjust the x acis
	for t in ax_hist.get_xticklabels():
		t.set_fontsize(16)

	if not (rundet.plotpath is None):
		if rundet.ensemble:
			# catch the different types of ensembles
			if "diff" in mapdet.var:
				# difference between two runs:
				# Make a pdf version
				fnm_pdf = "%s%d.%d.hist_%s.pdf" % (
					rundet.plotpath, (mapdet.column-2), rundet.paper,  mapdet.var)
				plt.savefig(fnm_pdf)
				
				# make png version 
				fname = "%s%d.%d.hist_%s.png" % (
						rundet.plotpath, (mapdet.column-2), rundet.paper,  mapdet.var)
				plt.savefig(fname)
				
				plotinfo = "HIST INFO: rundif: %dsub%d hist plot of %s made using %s:v.%s" % (
					rundet.run[0], rundet.run[1], mapdet.var, __title__, __version__)
			else:
				# ===== Run Ensembles ===== 
				# Make a pdf version
				fnm_pdf = "%s%d.%s_ensemble_hist_%s.pdf" % ( 
					rundet.plotpath, rundet.paper,rundet.desc, mapdet.var)

				plt.savefig(fnm_pdf)
				
				# make png version 
				fname = "%s%d.%s_ensemble_hist_%s.png" % ( 
					rundet.plotpath, rundet.paper,rundet.desc, mapdet.var)
				plt.savefig(fname)
				
				plotinfo = "HIST INFO: Ensmble hist plot of %s made using %s:v.%s" % (
					mapdet.var, __title__, __version__)
		
		else:
			# Make a pdf version
			fnm_pdf = "%s%d.%d_BasicFigs_hist_%s.pdf" % (
				rundet.plotpath, (mapdet.column-2), rundet.run, mapdet.var)
			plt.savefig(fnm_pdf)
				
			# make png version 
			fname = "%s%d.%d_BasicFigs_hist_%s.png" % (
				rundet.plotpath, (mapdet.column-2), rundet.run, mapdet.var)
			plt.savefig(fname)
			plotinfo = "HIST INFO: Run:%d hist plot of %s made using %s:v.%s" % (
				rundet.run, mapdet.var, __title__, __version__)
	else:
		fname = None

	if mapdet.pshow:
		plt.show()
	plt.close()
	plt.rcParams.update(plt.rcParamsDefault)
	return plotinfo, fname
# =============================================================================
