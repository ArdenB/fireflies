"""
Script goal, to produce trends in netcdf files
This script can also be used in P03 if required

"""
#==============================================================================

__title__ = "Boreal CLimate plots"
__author__ = "Arden Burrell"
__version__ = "v1.0(04.05.2019)"
__email__ = "arden.burrell@gmail.com"

#==============================================================================
# +++++ Check the paths and set ex path to fireflies folder +++++
import os
import sys
if not os.getcwd().endswith("fireflies"):
	if "fireflies" in os.getcwd():
		p1, p2, _ =  os.getcwd().partition("fireflies")
		os.chdir(p1+p2)
	else:
		raise OSError(
			"This script was called from an unknown path. CWD can not be set"
			)
sys.path.append(os.getcwd())

#==============================================================================
# Import packages
import numpy as np
import pandas as pd
import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
from netCDF4 import Dataset, num2date, date2num 
from scipy import stats
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
import glob
from scipy import stats
import statsmodels.stats.multitest as smsM
import myfunctions.PlotFunctions as pf
import myfunctions.corefunctions as cf

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)
#==============================================================================
def main():

	resinfo = results()

	# ========== loop over each dataset type ==========
	for dst in resinfo:
		# ========== Load the data ==========
		ds   = xr.open_dataset(resinfo[dst]["fname"])
		# ========== Load the mask for the correct grid ==========
		mask = xr.open_dataset(
		"./data/other/ForestExtent/BorealForestMask_%s.nc"%(resinfo[dst]["grid"]))
		
		# ========== Build the map ==========
		# xr_mapmaker(dst, ds, mask, resinfo[dst])

		# ========== Get the stats ==========
		statsmaker(dst, ds, mask, resinfo[dst])

#==============================================================================

def xr_mapmaker(dst, ds, mask, dsinfo):
	"""
	Function for setting up the plots
	args:
		dst:	str 
			name of the variable being mapped
		ds:		xr DS
			the dataset containg the result to be plotted
		mask:	xr DS
			the dataset with the mask
		dsinfo:	dict
			infomation about the dataset

	"""
	# ========== make my map detiles object ==========
	mapdet = pf.mapclass("boreal")

	# ========== add infomation to mapdet ==========
	mapdet.var      = "slope" #the thing to be plotted
	mapdet.mask     =  mask    # dataset to maks with
	mapdet.masknm   = "BorealForest" # When the mask file is an xr dataset, the var name
	mapdet.sigmask  = "Significant" # used for passing the column of significance maskis around
	mapdet.sighatch = True

	# ========== Get the colorbar values ==========
	cmap, vmin, vmax, ticks  = cbvals(dst, "slope")
	
	# ========== Add the values to the mapdet ==========
	mapdet.cmap  = cmap # Colormap set later
	mapdet.cmin  = vmin # the min of the colormap
	mapdet.cmax  = vmax # the max of the colormap
	mapdet.ticks = ticks
	mapdet.dpi   = 500 
	mapdet.save  = True
	# mapdet.cblabel  = "Trend in %s (%s)" % (dsinfo["param"], dsinfo["units"]) 
	mapdet.cblabel  = "%s" % (dsinfo["units"]) 
	mapdet.plotpath = "./plots/bookchapter/firstdraft/"

	mapdet.fname    = "%s%s_%s_%s_%s_%dyrMW_%sFDR" %(
		mapdet.plotpath, dsinfo["source"], dsinfo["param"],
		dsinfo["test"], mapdet.var,
		dsinfo["window"], dsinfo["FDRmethod"])
	if dst == "tmean":
		mapdet.set_x  = 2.25 
		mapdet.extend = "max"

	cf.pymkdir(mapdet.plotpath)
	warn.warn("Polar projection needs work")
	# mapdet.projection = ccrs.NorthPolarStereo()

	# ========== Make the map ==========
	fname, plotinfo = pf.mapmaker(ds, mapdet)
	
	# ========== Make metadata infomation ========== 
	if not (fname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotinfo, fname, gitinfo]
		cf.writemetadata(fname, infomation)

	# ipdb.set_trace()

def statsmaker(dst, ds, mask, dsinfo):
	"""
	Function for setting up the plots
	args:
		dst:	str 
			name of the variable being mapped
		ds:		xr DS
			the dataset containg the result to be plotted
		mask:	xr DS
			the dataset with the mask
		dsinfo:	dict
			infomation about the dataset

	"""
	# ========== Get some info ==========
	maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
		__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
	gitinfo = pf.gitmetadata()

	# ========== Build some stats ==========
	stats = ["Stats for %s \n" % dst, maininfo, gitinfo, "\n"]
	
	# ========== mask the data to the boreal zone ==========
	DA    = ds["slope"      ]* mask.BorealForest.values
	DA_SM = ds['Significant']* mask.BorealForest.values
	
	# ========== add some findings ==========
	stats.append("Mean +- SV change per year: %f +- %f (%s)"% (DA.mean(), DA.var(), dsinfo["units"]))
	stats.append("Max : %f (%s)"% (DA.max(), dsinfo["units"]))
	stats.append("Min : %f (%s)"% (DA.min(), dsinfo["units"]))

	stats.append("fraction sig increasing:  %f \n"% (
		np.logical_and((DA>0), (DA_SM==1)).sum()/ (~np.isnan(DA)).sum().astype(float)))
	stats.append("fraction sig decreasing:  %f \n"% (
		np.logical_and((DA<0), (DA_SM==1)).sum()/ (~np.isnan(DA)).sum().astype(float)))
	stats.append("fraction no sig change:  %f \n"% (
		(DA_SM==0).sum()/ (~np.isnan(DA)).sum().astype(float)))

	# DA
	# ========== save the info out ==========
	outpath = "./plots/bookchapter/firstdraft/"
	fname   = "%s%s_%s_%s_%s_%dyrMW_%sFDR_BasicStats" %(
			outpath, dsinfo["source"], dsinfo["param"],
			dsinfo["test"], "slope",
			dsinfo["window"], dsinfo["FDRmethod"])
	cf.writemetadata(fname, stats)
	print(stats)
	ipdb.set_trace()



def Polar_maker():
	"""
	this will turn into a polar version of my mapmaker script
	"""
	fig, ax = plt.subplots(
		1, 1, subplot_kw={'projection': mapdet.projection}, 
		num=("Map of %s" % mapdet.var), dpi=mapdet.dpi) 
	
	DA.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=mapdet.cmap, vmin=-4, vmax=4)
	
	# ========== Add features to the map ==========
	ax.add_feature(cpf.LAND, facecolor=mapdet.maskcol, alpha=1, zorder=0)
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(cpf.COASTLINE, zorder=101)
	if mapdet.national:
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)

	plt.show()
#==============================================================================
# ================= Functions to setup infomation about plots =================
#==============================================================================
def results():
	"""
	Function to return the infomation about the results
	"""
	res = OrderedDict()
	res["tmean"] = ({
		"fname":"./results/netcdf/TerraClimate_Annual_RollingMean_tmean_theilsento2017_GlobalGIMMS.nc",
		"source":"TerraClimate","test":"Theisen", "FDRmethod":"BenjaminiHochberg",
		"window":20, "grid":"GIMMS", "param":"MeanAnnualTemperature", 
		"units":r"$^{o}$C yr$^{-1}$"})
	res["ppt"] = ({
		"fname":"./results/netcdf/TerraClimate_Annual_RollingMean_ppt_theilsento2017_GlobalGIMMS.nc",
		"source":"TerraClimate","test":"Theisen", "FDRmethod":"BenjaminiHochberg",
		"window":20, "grid":"GIMMS", "param":"AnnualPrecipitation", 
		"units":r"mm yr$^{-1}$"})
	return res


def cbvals(var, ky):

	"""Function to store all the colorbar infomation i need """
	cmap  = None
	vmin  = None
	vmax  = None
	ticks = None
	if ky == "slope":
		if var == "tmean":
			vmax =  0.06
			vmin =  0.00
			# cmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_10.mpl_colors)
			cmap = mpc.ListedColormap(palettable.colorbrewer.sequential.OrRd_6.mpl_colors)
			ticks = ticks = np.arange(vmin, vmax+0.1, 0.02)
		elif var =="ppt":
			vmin = -4.0
			vmax =  4.0
			cmap = mpc.ListedColormap(palettable.cmocean.diverging.Curl_8_r.mpl_colors)
			ticks = np.arange(vmin, vmax+1, 2.0)
	elif ky == "pvalue":
		cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20.hex_colors)
		vmin = 0.0
		vmax = 1.0
	elif ky == "rsquared":
		cmap = mpc.ListedColormap(palettable.matplotlib.Viridis_20.hex_colors)
		vmin = 0.0
		vmax = 1.0
		# cmap =  
	elif ky == "intercept":
		cmap = mpc.ListedColormap(palettable.cmocean.sequential.Ice_20_r.mpl_colors)
		if var == "tmean":
			# vmax =  0.07
			# vmin = -0.07
			# cmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors)
			# ipdb.set_trace()
			pass
		elif var =="ppt":
			vmin = 0
			vmax = 1000
			# cmap = mpc.ListedColormap(palettable.cmocean.diverging.Curl_20_r.mpl_colors)
	return cmap, vmin, vmax, ticks

if __name__ == '__main__':
	main()