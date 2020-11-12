"""
Script goal, 

Calculate key facts and figures for the manuscript

"""
#==============================================================================

__title__ = "FRI stat calculator"
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
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from statsmodels.stats.weightstats import DescrStatsW


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

#==============================================================================
def main():

	# ========== Setup the params ==========
	TCF = 10
	mwbox   = [1]#, 2]#, 5]
	dsnames = ["GFED", "MODIS", "esacci", "COPERN_BA", "HANSEN_AFmask", "HANSEN"]
	# dsnams2 = ["HANSEN_AFmask", "HANSEN"]
	# dsts = [dsnams1, dsnams2]
	# vmax    = 120
	# vmax    = 80
	# vmax    = 100

	for var in ["FRI", "AnBF"]:
		formats = [".png"]#, ".pdf"] # None
		# mask    = True
		if TCF == 0:
			tcfs = ""
		else:
			tcfs = "_%dperTC" % np.round(TCF)


		# ========== Setup the plot dir ==========
		plotdir = "./plots/ShortPaper/"
		cf.pymkdir(plotdir)
		griddir = "./data/gridarea/"
		# compath = "/media/ubuntu/Seagate Backup Plus Drive"
		compath, backpath = syspath()

		for mwb in mwbox:
			# ========== Setup the dataset ==========
			datasets = OrderedDict()
			for dsnm in dsnames:
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
				# ipdb.set_trace()
				datasets[dsnm] = ppath+fname #xr.open_dataset(ppath+fname)
				# ipdb.set_trace()
			stat = OrderedDict()
			for dsn in datasets:
				print(dsn)
				stat[dsn] = statcal(dsn, var, datasets, compath, backpath, region = "SIBERIA")
			keystats = pd.DataFrame(stat).T

			# ========== save the info out ==========
			# ========== Create the Metadata ==========
			Scriptinfo = "File saved from %s (%s):%s by %s, %s" % (__title__, __file__, 
				__version__, __author__, str(pd.Timestamp.now()))
			gitinfo = pf.gitmetadata()

			outpath = plotdir+"stats/"
			cf.pymkdir(outpath) 
			cf.writemetadata(outpath+f"PF02_{var}stats", [Scriptinfo, gitinfo, keystats])
			# df.groupby("ACC_CD").aream2.sum() * 1e-12
			print(keystats)
			ipdb.set_trace()


def statcal(dsn, var, datasets, compath, backpath, region = "SIBERIA", griddir = "./data/gridarea/"):
		
	cf.pymkdir(griddir)
	# ========== open the dataset ==========
	if not os.path.isfile(datasets[dsn]):
		# The file is not in the folder
		warn.warn(f"File {datasets[dsn]} could not be found")
		breakpoint()
	else:
		# /// the grid area dataset \\\
		gafn   = f"{griddir}{dsn}_gridarea.nc"
		if not os.path.isfile(gafn):
			subp.call(f"cdo gridarea {datasets[dsn]} {gafn}", shell=True)

		ds_ga = xr.open_dataset(gafn).astype(np.float32).sel(
			dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
		ds_ga["cell_area"] *= 1e-6 # Convert from sq m to sq km
		# /// the dataset \\\
		ds_dsn = xr.open_dataset(datasets[dsn])

	# ========== Get the data for the frame ==========
	frame = ds_dsn[var].isel(time=0).sortby("latitude", ascending=False).sel(
		dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0))).drop("time")
	bounds = [-10.0, 180.0, 70.0, 40.0]

	# ========== mask ==========
	stpath = compath + "/masks/broad/"

	if not dsn.startswith("H"):
		fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsn)
	else:
		fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedToesacci.nc" % (region)

	# +++++ Check if the mask exists yet +++++
	if os.path.isfile(fnmask):
		with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask:
			
			msk    = dsmask.mask.isel(time=0).astype("float32").values

			# +++++ Change the boolean mask to NaNs +++++
			msk[msk == 0] = np.NAN
			
			print("Masking %s frame at:" % dsn, pd.Timestamp.now())
			# +++++ mask the frame +++++
			frame *= msk

			# +++++ close the mask +++++
			msk = None

	# ========== Calculate the stats ==========
	stats = OrderedDict()

	# ========== create the weights ==========
	weights        = np.cos(np.deg2rad(frame.latitude))
	weights.name   = "weights"
	
	# ========== calculate the number of nans and the number of  ==========
	# (frame.isnull()).weighted(weights).sum() / (~frame.isnull()).weighted(weights).sum()
	NN = ((~frame.isnull()).weighted(weights).sum()).values
	NA = ((frame.isnull()).weighted(weights).sum()).values
	stats["NonNan"] = NN / (NN+NA)

	# ========== Mask ouside the range ==========
	if var =="FRI":
		stats["OutRgnFrac"] = ((frame>10000.).weighted(weights).sum() / NN).values
		stats["OutRgnsqkm"] = ((frame>10000.) * ds_ga["cell_area"]).sum().values
		# stats["Median"]     = frame.quantile(0.5).values
		# ========== Calculate the weighted median ==========
		# dft = frame.to_dataframe().dropna().reset_index()
		# dft["weights"] = np.cos(np.deg2rad(dft["latitude"]))
		# dft.drop(["latitude", "longitude"],axis=1, inplace=True).
		# dft.sort_values('FRI', inplace=True)
		# dft = dft.reset_index()
		# cumsum = dft.FRI.cumsum()
		# cutoff = dft.FRI.sum() / 2.0
		# stats["median"] = dft.FRI[cumsum >= cutoff].iloc[0]
		# del dft

		# ========== Mask ouside the range ==========
		frame = frame.where(~(frame>10000.), 10001)
	elif var == "AnBF":
		stats["OutRgnFrac"] = ((frame<0.0001).weighted(weights).sum() / NN).values
		frame = frame.where(frame>0.0001)
	
	# ========== Use statsmodels to calculate the key statistis ==========
	d1    = DescrStatsW(frame.values[~frame.isnull()], weights=ds_ga["cell_area"].values[~frame.isnull()])
	stats[f"Mean{var}"] = d1.mean
	stats[f"std{var}"]  = d1.std
	if var == "FRI":
		stats["FRIsub15"] =  ((frame  < 15).weighted(weights).sum()/NN).values
		stats["FRIsub30"] =  (((frame < 30).weighted(weights).sum()/NN) - stats["FRIsub15"]).values
		stats["FRIsub60"] =  (((frame < 60).weighted(weights).sum()/NN) - (stats["FRIsub15"]+stats["FRIsub30"])).values
		stats["FRIsub15sqkm"] =  ((frame  < 15)* ds_ga["cell_area"]).sum().values
		stats["FRIsub30sqkm"] =  (((frame < 30)* ds_ga["cell_area"]).sum() - stats["FRIsub15sqkm"]).values
		stats["FRIsub60sqkm"] =  (((frame < 60)* ds_ga["cell_area"]).sum() - (stats["FRIsub15sqkm"]+stats["FRIsub30sqkm"])).values
	# ========== Do the weighted quantiles ==========
	cquants = [0.001, 0.01,  0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
	quant   = d1.quantile(cquants)
	for cq in cquants:
		stats[f"{cq*100}percentile"] = quant[cq]
	del frame
	return pd.Series(stats)


def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	backpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "/mnt/d/Data51"
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
		dpath = "/media/ubuntu/Harbinger/Data51"
	# elif 'ccrc.unsw.edu.au' in sysname:
	# 	dpath = "/srv/ccrc/data51/z3466821"
	elif sysname == 'burrell-pre5820':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
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