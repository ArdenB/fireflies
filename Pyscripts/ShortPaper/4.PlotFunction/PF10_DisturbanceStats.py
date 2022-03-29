"""
Make maps of the the future risk cats

"""

#==============================================================================

__title__ = "Disturbance stat maker"
__author__ = "Arden Burrell"
__version__ = "v1.0(11.11.2020)"
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
import subprocess as subp
from dask.diagnostics import ProgressBar
import dask

from collections import OrderedDict
# from cdo import *

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
import seaborn as sns

# import seaborn as sns
import cartopy as ct
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
import matplotlib.colors as mpc
import matplotlib.patheffects as pe
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from statsmodels.stats.weightstats import DescrStatsW
import pickle
from sklearn import metrics as sklMet

# ========== Import ml packages ==========
import sklearn as skl
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
import ipdb
from numba import vectorize, float64



print("numpy version   : ", np.__version__)
print("pandas version  : ", pd.__version__)
print("xarray version  : ", xr.__version__)
print("cartopy version : ", ct.__version__)
print("sklearn version : ", skl.__version__)

#==============================================================================
def main():
	# ========== find the paths ==========
	# dpath, cpath = syspath()
	# ppath = "./plots/ShortPaper/PF08_FutRisk/"
	# cf.pymkdir(ppath)
	plotdir = "./plots/ShortPaper/PF10_DRIStats/"
	cf.pymkdir(plotdir)
	mask    = True
	bounds  = [10.0, 170.0, 70.0, 49.0]
	maskver = "Boreal"
	proj    = "polar"
	var     = "AnBF"
	# formats = [".png"]#, ".pdf"]
	formats = [".png", ".tiff", ".eps"]# ".pdf"


	# dsnams1 = ["GFED", "MODIS", "esacci", "COPERN_BA"]#, "HANSEN_AFmask", "HANSEN"]
	# dsnams2 = ["HANSEN_AFmask", "HANSEN", "SRfrac"]
	# dsnams3 = ["Risk"]
	scale   = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20, "Risk":20, "SRfrac":20, "FutureRisk":20})
	
	# info = OrderedDict()
	# info ["FutureRisk"] = ({
	# 	"fnpre":"./results/ProjectSentinal/FRImodeling/S03_FRIdrivers_esacci_v0_100yr_trendPrediction_forests_sigclim.nc",
	# 	"fnSRF":""
	# 	})
	mask = True
	compath, backpath = syspath()
	TCF     = 10
	if TCF == 0:
		tcfs = ""
	else:
		tcfs = "_%dperTC" % np.round(TCF)

	dsinfo  = dsinfomaker(compath, backpath, 1, tcfs)
	dsn1    = "HANSEN_AFmask"
	dsn2    = "HANSEN"
	# ds = xr.open_dataset(dsinfo[dsn2]["fname"])
	frame1 = _fileopen(dsinfo, dsn1, var, scale, proj, mask, compath, bounds, maskver, func ="sum")
	frame2 = _fileopen(dsinfo, dsn2, var, scale, proj, mask, compath, bounds, maskver, func ="sum")
	stamaker(frame1, frame2, plotdir)
	breakpoint()


#==============================================================================
def stamaker(frame1, frame2, plotdir):
	Scriptinfo = "File saved from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, str(pd.Timestamp.now()))
	gitinfo = pf.gitmetadata()

	# ========== Calculate the stats ==========
	stats = OrderedDict()
	stats["title"] = "Fraction of Disturbance that is attributable to fire"
	stats["value"]  =  (frame1.sum() / frame2.sum()).values
	keystats = pd.DataFrame(stats)
	# breakpoint()

	outpath = plotdir+"stats/"
	cf.pymkdir(outpath) 
	cf.writemetadata(outpath+f"PF10_stats", [Scriptinfo, gitinfo])
	keystats.to_csv(outpath+f"PF10_stats.csv")

	breakpoint()





def _fileopen(dsinfo, dsn, var, scale, proj, mask, compath, bounds, maskver, func = "mean", region = "SIBERIA",):
	ds_dsn = xr.open_dataset(dsinfo[dsn]["fname"])
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
				}, boundary ="pad").mean().compute()
		elif func == "max":
			frame = frame.coarsen(
				{"latitude":scale[dsn], "longitude":scale[dsn]
				}, boundary ="pad").max().compute()
		elif func == "sum":
			frame = frame.coarsen(
				{"latitude":scale[dsn], "longitude":scale[dsn]
				}, boundary ="pad").sum().compute()
		else:
			print("Unknown Function")
			breakpoint()
	
	frame.attrs = dsinfo[dsn]#{'long_name':"FRI", "units":"years"}

	# ========== mask ==========
	if mask:
		# +++++ Setup the paths +++++
		# stpath = compath +"/Data51/ForestExtent/%s/" % dsn
		stpath = compath + "/masks/broad/"

		if dsn.startswith("H") or (dsn in ["Risk", "SRfrac", "FutureRisk"]):
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
				
				# breakpoint()
				msk = msk.values

				# +++++ Change the boolean mask to NaNs +++++
				msk[msk == 0] = np.NAN
				
				print("Masking %s frame at:" % dsn, pd.Timestamp.now())
				# +++++ mask the frame +++++
				frame *= msk

				# +++++ close the mask +++++
				msk = None
				print(f"masking complete, begining ploting at {pd.Timestamp.now()}")


		else:
			print("No mask exists for ", dsn)
			breakpoint()
	return frame
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
	elif sysname in ['LAPTOP-8C4IGM68', 'DESKTOP-N9QFN7K']:
		dpath     = "./data"
		backpath = "/mnt/d/fireflies"
	else:
		ipdb.set_trace()
	return dpath, backpath

def dsinfomaker(compath, backpath, mwb, tcfs,  SR="SR"):#yrs, ves,
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
	# dsinfo["FutureRisk"]    = ({"alias":"Forest Loss Risk"})
	dsinfo["SRfrac"]        = ({"alias":"Stand Replacing Fire Percentage", "long_name":f'FRI$_{{{"SR"}}}$ %'})

	for dsnm in dsinfo:
		if dsnm.startswith("H"):
			# +++++ make a path +++++
			ppath = compath + "/BurntArea/HANSEN/FRI/"
			fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, tcfs, mwb)
			# fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		elif dsnm == "Risk":
			ppath = compath + "/BurntArea/Risk/FRI/"
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
			cf.pymkdir(ppath)
		# elif dsnm == "FutureRisk":
		# 	ppath = compath + "/BurntArea/Risk/FRI/"
		# 	fname = f"{dsnm}_annual_burns_MW_{mwb}degreeBox_{yrs}yrs_{ves}.nc" 
		# 	cf.pymkdir(ppath)
		else:
			# fname = "Hansen_GFC-2018-v1.6_regrided_esacci_FRI_%ddegMW_SIBERIA" % (mwb)
			ppath = compath + "/BurntArea/%s/FRI/" %  dsnm
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		# +++++ open the datasets +++++
		dsinfo[dsnm]["fname"] = ppath+fname


	return dsinfo
# ==============================================================================
#==============================================================================
if __name__ == '__main__':
	main()
