"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

"""
#==============================================================================

__title__ = "FRIsr Comparison"
__author__ = "Arden Burrell"
__version__ = "v1.0(02.04.2020)"
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
# import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import shutil
import time
from dask.diagnostics import ProgressBar

from collections import OrderedDict
# from scipy import stats
# from numba import jit


# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl

import palettable 

# import seaborn as sns
import matplotlib as mpl 
import cartopy as ct
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from itertools import chain


# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)
print("cartopy version : ", ct.__version__)

#==============================================================================

def main():
	# ========== Setup the params ==========
	TCF     = 10
	mwb     = 1#, 2]#, 5]
	dsnams1 = ["HANSEN_AFmask", "esacci"]#, "HANSEN_AFmask", "HANSEN"]
	dsnams2 = ["HANSEN_AFmask", "HANSEN"]
	scale   = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20, "Risk":20})
	dsts    = [dsnams1, dsnams2]
	proj    = "polar"
	maskver = "Boreal"	
	mask    = True
	# for var in :
	# for dsnames, vmax in zip(dsts, [10000, 10000]):
	formats = [".png"]#, ".pdf"] # None 
	# mask    = True
	if TCF == 0:
		tcfs = ""
	else:
		tcfs = "_%dperTC" % np.round(TCF)


	# ========== Setup the plot dir ==========
	plotdir = "./plots/ShortPaper/PF06_FRIsr/"
	cf.pymkdir(plotdir)
	compath, backpath = syspath()
	dsinfo  = dsinfomaker(compath, backpath, mwb, tcfs)
	
	# ========== Setup the dataset ==========
	datasets = OrderedDict()
	for dsnm in list(chain.from_iterable(dsts)):
		if dsnm.startswith("H"):
			# +++++ make a path +++++
			ppath = compath + "/BurntArea/HANSEN/FRI/"
			fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, tcfs, mwb)
			# fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		else:
			# fname = "Hansen_GFC-2018-v1.6_regrided_esacci_FRI_%ddegMW_SIBERIA" % (mwb)
			ppath = compath + "/BurntArea/%s/FRI/" %  dsnm
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		# +++++ open the datasets +++++
		cf.pymkdir(ppath)
		datasets[dsnm] = ppath+fname #xr.open_dataset(ppath+fname)
		# ipdb.set_trace()
	bounds = [10.0, 170.0, 70.0, 49.0]#[-10.0, 180.0, 70.0, 40.0]
	exper  = OrderedDict()
	exper["StandReplacingFireFraction"] = ({
		"data":dsnams1,
		"var" :"AnBF", 
		"vmax":1, 
		"vmin":0,
		"func":"div",
		"long_name":f'FRI$_{{{"SR"}}}$ fraction',
		"extend":"neither"
		})
	exper["DRIwithoutfires"] = ({
		"data":dsnams2,
		"var" :"FRI",
		"vmax":10000,
		"vmin":0,
		"func":"sub",
		"long_name":f'DRI$_{{{"SF"}}}$',
		"extend":"max"
		})
	# for mask, var in zip([True, True], ["FRI", "AnBF"]):
	plotmaker(exper, dsinfo, datasets, mwb, plotdir, formats, mask, compath, backpath, proj, scale, bounds, maskver)
	breakpoint()

				# ipdb.set_trace()

#==============================================================================
def plotmaker(exper, dsinfo, datasets, mwb, plotdir, formats, mask, compath, backpath, proj, scale, bounds, maskver):
	""" Function to make the plots """

	# ========== make the plot name ==========
	plotfname = plotdir + f"PF06_sup_StandReplacingFire"
	if mask:
		plotfname += "_ForestMask_V2"

	# ========== Setup the font ==========
	font = ({
		'weight' : 'bold',
		'size'   : 11, 
		})
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

		
	# ========== setup the figure ==========
	if proj == "polar":
		latiMid=np.mean([bounds[2], bounds[3]])
		longMid=np.mean([bounds[0], bounds[1]])
		yv = len(exper)
		xv = 1
		shrink=0.95

		fig, axs = plt.subplots(
			yv, xv, figsize=(12,5*len(datasets)), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})
	else:
		latiMid=np.mean([bounds[2], bounds[3]])
		longMid=np.mean([bounds[0], bounds[1]])
		fig, axs = plt.subplots(
			len(datasets), 1, sharex=True, 
			figsize=(16,9), subplot_kw={'projection': ccrs.PlateCarree()})
		shrink = None
	# ========== Loop over the experiments ==========
	for ax, exp in zip(axs, exper):
		im = _subplotmaker(exp, ax, exper, dsinfo, datasets, mwb, plotdir, formats, mask, compath, backpath, proj, scale, bounds, maskver)	
		ax.set_aspect('equal')



	plt.subplots_adjust(top=0.971,bottom=0.013,left=0.008,right=0.993,hspace=0.063,wspace=0.0)

	if not (formats is None): 
		# ========== loop over the formats ==========
		for fmt in formats:
			print(f"starting {fmt} plot save at:{pd.Timestamp.now()}")
			plt.savefig(plotfname+fmt)#, dpi=dpi)
	print("Starting plot show at:", pd.Timestamp.now())
	
	plt.show()
	if not (plotfname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotfname, gitinfo]
		cf.writemetadata(plotfname, infomation)
	breakpoint()

def _subplotmaker(exp, ax, exper, dsinfo, datasets, mwb, plotdir, formats, mask, compath, backpath, proj, scale, bounds, maskver, region = "SIBERIA", shrink=0.85): 
	""" 
	function to load all the data, calculate the experiment and add it to the axis 
	"""
	dlist = []
	for dsn in exper[exp]['data']:
		print(f"Starting {dsn} load at: {pd.Timestamp.now()}")
		if not os.path.isfile(datasets[dsn]):
			# +++++ The file is not in the folder +++++
			warn.warn(f"File {datasets[dsn]} could not be found")
			breakpoint()
		else:
			frame = _fileopen(dsinfo, datasets, dsn, exper[exp]["var"], scale, proj, mask, compath, region, bounds, maskver)
			dlist.append(frame)
	
	breakpoint()
	# ========== Set the colors ==========
	# cmap, norm, vmin, vmax, levels = _colours(var, vmax, dsn)

	# ========== Create the Title ==========
	title = ""
	

#==============================================================================

def _fileopen(dsinfo, datasets, dsn, var, scale, proj, mask, compath, region, bounds, maskver, func = "mean"):
	ds_dsn = xr.open_dataset(datasets[dsn])
	# xbounds [-10.0, 180.0, 70.0, 40.0]
	xbounds = [-10.0, 180.0, 70.0, 40.0]
	# ========== Get the data for the frame ==========
	frame = ds_dsn[var].isel(time=0).sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1]))).drop("time")
	
	if proj == "polar" and not dsn == "GFED":
		# ========== Coarsen to make plotting easier =========
		if func == "mean":
			frame = frame.coarsen(
				{"latitude":scale[dsn], "longitude":scale[dsn]
				}, boundary ="pad", keep_attrs=True).mean().compute()
		elif func == "max":
			frame = frame.coarsen(
				{"latitude":scale[dsn], "longitude":scale[dsn]
				}, boundary ="pad", keep_attrs=True).max().compute()
		else:
			print("Unknown Function")
			breakpoint()
	
	frame.attrs = dsinfo[dsn]#{'long_name':"FRI", "units":"years"}

	# ========== mask ==========
	if mask:
		# +++++ Setup the paths +++++
		# stpath = compath +"/Data51/ForestExtent/%s/" % dsn
		stpath = compath + "/masks/broad/"

		if dsn.startswith("H") or (dsn == "Risk"):
			fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedToesacci.nc" % (region)
			fnBmask = f"./data/LandCover/Regridded_forestzone_esacci.nc"
		else:
			fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsn)
			fnBmask = f"./data/LandCover/Regridded_forestzone_{dsn}.nc"

		# +++++ Check if the mask exists yet +++++
		if os.path.isfile(fnmask):
			with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask, xr.open_dataset(fnBmask).drop(["DinersteinRegions", "GlobalEcologicalZones", "LandCover"]) as Bmask:
				# breakpoint()
				if maskver == "Boreal":
					msk    = (dsmask.mask.isel(time=0)*((Bmask.BorealMask.isel(time=0)>0).astype("float32")))#.sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
				else:
					msk    = (dsmask.mask.isel(time=0)).astype("float32")
				
				if proj == "polar" and not dsn == "GFED":
					msk = msk.coarsen({"latitude":scale[dsn], "longitude":scale[dsn]}, boundary ="pad").median()
				
				msk = msk.values

				# +++++ Change the boolean mask to NaNs +++++
				msk[msk == 0] = np.NAN
				
				print("Masking %s frame at:" % dsn, pd.Timestamp.now())
				# +++++ mask the frame +++++
				# breakpoint()
				frame *= msk

				# +++++ close the mask +++++
				msk = None
				print(f"masking complete, begining ploting at {pd.Timestamp.now()}")


		else:
			print("No mask exists for ", dsn)
			breakpoint()
	# breakpoint()
	return frame


def dsinfomaker(compath, backpath, mwb, tcfs, SR="SR"):
	"""
	Contains infomation about the Different datasets
	"""
	dsinfo = OrderedDict()
	# ==========
	dsinfo["GFED"]          = ({"alias":"GFED4.1","long_name":"FRI", "units":"yrs"})
	dsinfo["MODIS"]         = ({"alias":"MCD64A1", "long_name":"FRI","units":"yrs", "version":"v006"})
	dsinfo["esacci"]        = ({"alias":"FireCCI5.1", "long_name":"FRI","units":"yrs"})
	dsinfo["COPERN_BA"]     = ({"alias":"CGLS", "long_name":"FRI","units":"yrs"})
	dsinfo["HANSEN_AFmask"] = ({"alias":"Hansen GFC & MCD14ML", "long_name":f'FRI$_{{{SR}}}$',"units":"yrs"})
	dsinfo["HANSEN"]        = ({"alias":"Hansen GFC", "long_name":"DRI","units":"yrs"})
	dsinfo["Risk"]          = ({"alias":"Forest Loss Risk"})

	for dsnm in dsinfo:
		if dsnm.startswith("H"):
			# +++++ make a path +++++
			ppath = compath + "/BurntArea/HANSEN/FRI/"
			fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, tcfs, mwb)
			# fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		else:
			# fname = "Hansen_GFC-2018-v1.6_regrided_esacci_FRI_%ddegMW_SIBERIA" % (mwb)
			ppath = compath + "/BurntArea/%s/FRI/" %  dsnm
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		# +++++ open the datasets +++++
		dsinfo[dsnm]["fname"] = ppath+fname


	return dsinfo

def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	backpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "/mnt/d/Data51"
		# dpath = "./data"
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
		dpath = "/media/ubuntu/Harbinger/Data51"
	# elif 'ccrc.unsw.edu.au' in sysname:
	# 	dpath = "/srv/ccrc/data51/z3466821"
	elif sysname == 'DESKTOP-T77KK56':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		backpath = "/mnt/f/fireflies"
		chunksize = 500
	elif sysname == 'DESKTOP-KMJEPJ8':
		dpath = "./data"
		backpath = "/mnt/g/fireflies"
		chunksize = 500
	elif sysname == 'arden-Precision-5820-Tower-X-Series':
		# WHRC linux distro
		dpath = "./data"
		breakpoint()
		# dpath= "/media/arden/Harbinger/Data51/BurntArea"
	elif sysname == 'LAPTOP-8C4IGM68':
		dpath     = "./data"
		backpath = "/mnt/d/fireflies"
	else:
		ipdb.set_trace()
	return dpath, backpath

#==============================================================================
if __name__ == '__main__':
	main()