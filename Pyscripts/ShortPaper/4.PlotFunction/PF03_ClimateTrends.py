"""
Script goal, 

Make the climate maps

"""
#==============================================================================

__title__ = "Climate stat calculator"
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

# import seaborn as sns
import cartopy as ct
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
import matplotlib.colors as mpc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from statsmodels.stats.weightstats import DescrStatsW


# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
import ipdb

print("numpy version   : ", np.__version__)
print("pandas version  : ", pd.__version__)
print("xarray version  : ", xr.__version__)
print("cartopy version : ", ct.__version__)

#==============================================================================
def main():
	# ========== find the paths ==========
	dpath, cpath = syspath()
	ppath = "./plots/ShortPaper/PF03_Climate/"
	cf.pymkdir(ppath)

	# ========== Build the annual plots ==========
	Seasonalplotmaker(setupfunc("seasonal"), dpath, cpath, ppath)

	# ========== Build the annual plots ==========
	AnnualPlotmaker(setupfunc("annual"), dpath, cpath, ppath)
	



#==============================================================================

def Seasonalplotmaker(setup, dpath, cpath, ppath):
	""" 
	Function fo making the Seasonal plots
	args:
		setup: Ordered dict 
			contains the cmap and vrng infomation
		dpath:	str
		cpath:	str
			path to the climate data
	"""
	# ========== load the mask ==========
	dsmask = xr.open_dataset(f"{dpath}/masks/broad/Hansen_GFC-2018-v1.6_SIBERIA_ProcessedToTerraClimate.nc")
	
	# ========== set the mpl rc params ==========
	font = {'weight' : 'bold'}
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# ========== Create the figure ==========
	fig, axs = plt.subplots(
		4, 2, sharex=True, 
		subplot_kw={'projection': ccrs.Orthographic(
			dsmask.longitude.median().values, 
			dsmask.latitude.median().values)}, 
		figsize=(14,12)
		)

	# ========== Loop over the rows ==========
	for sea, (row, raxes) in zip(["DJF", "MAM", "JJA", "SON"], enumerate(axs)):
		# ========== Loop over the variables ==========
		for va, ax in zip(setup, raxes):
			# ========== Read in the data and mask the boreal zone ==========
			ds = xr.open_dataset(f"{cpath}TerraClim_{va}_{sea}trend_1985to2015.nc")
			ds = ds.where(dsmask.datamask.values == 1)
			ds.slope.attrs = setup[va]["attrs"]



			p  = ds.slope.isel(time=0).plot(
				cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
				transform=ccrs.PlateCarree(), ax=ax,
				    cbar_kwargs={"pad": 0.02, "shrink":0.97, "extend":"both"})
			# 
			# ========== work out the stippling ==========
			slats, slons = _stippling(ds, squeeze=30, nanfrac = 0.15, sigfrac=0.5)
			ax.scatter(
				slons, slats, s=4, c='k', marker='X', 
				facecolors='none', edgecolors="none",  
				alpha=0.35, transform=ccrs.PlateCarree())
			# p.axes.coastlines(resolution ="50m", zorder=101)
			# if not mapdet.sigmask is None:
			# 	# Calculate the lat and lon values
			# 	slats = rundet.lats[mapdet.sigmask["yv"]]
			# 	slons = rundet.lons[mapdet.sigmask["xv"]]


			ax.gridlines()

			# p.axes.add_feature(cpf.COASTLINE, , zorder=101)
			coast_50m = cpf.GSHHSFeature(scale="high")
			ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
			ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
			ax.add_feature(coast_50m, zorder=101, alpha=0.5)
			ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
			ax.add_feature(cpf.RIVERS, zorder=104)
			
			# ========== Set the center titles ==========
			if row == 0:
				vanm = setup[va]["lname"]
				ax.set_title(f"{vanm}")	
			else:
				ax.set_title("")
			# ========== Set the left titles ==========
			if va == "ppt":
				ax.set_title(f"{sea}", loc= 'left')
			
			print(sea, va, ds.slope.quantile([0.01,0.05, 0.50,0.95,0.99]))

	plt.subplots_adjust(top=0.971,bottom=0.013, left=0.011, right=0.97, hspace=0.135,wspace=0.0)
	# plt.subplots_adjust(top=0.971, bottom=0.013, left=0.012, right=0.988, hspace=0.063, wspace=0.2)
	# ========== Save the plots ==========
	plotfname = f"{ppath}PF03_SeasonalClimateTrend."
	for fmt in ["pdf", "png"]:
		print(f"Starting {fmt} plot save at:{pd.Timestamp.now()}")
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



def AnnualPlotmaker(setup, dpath, cpath, ppath):
	""" Function fo making the annual plot
	args:
		setup: Ordered dict 
			contains the cmap and vrng infomation
		dpath:	str
		cpath:	str
			path to the climate data
	"""
	# ========== load the mask ==========
	dsmask = xr.open_dataset(f"{dpath}/masks/broad/Hansen_GFC-2018-v1.6_SIBERIA_ProcessedToTerraClimate.nc")
	
	# ========== set the mpl rc params ==========
	font = {'weight' : 'bold'}
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})
	
	# ========== Create the figure ==========
	fig, axs = plt.subplots(
		2, 1, sharex=True, 
		subplot_kw={'projection': ccrs.Orthographic(
			dsmask.longitude.median().values, 
			dsmask.latitude.median().values)}, 
		figsize=(13,12))

	# ========== load the datasets ==========
	for va, ax, let in zip(setup, axs, ["a", "b"]):
		# ========== Read in the data and mask the boreal zone ==========
		ds = xr.open_dataset(f"{cpath}TerraClim_{va}_annualtrend_1985to2015.nc")
		ds = ds.where(dsmask.datamask.values == 1)
		ds.slope.attrs = setup[va]["attrs"]

		p  = ds.slope.isel(time=0).plot(
			cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
			transform=ccrs.PlateCarree(), ax=ax,
			    cbar_kwargs={
			    "pad": 0.02, "shrink":0.97, "extend":"both"
			    })
			    # "label": "custom label",
			# subplot_kws=dict(projection=ccrs.Orthographic(
			# 	dsmask.longitude.median().values, dsmask.latitude.median().values)
			# )
		# p.axes.coastlines(resolution ="50m", zorder=101)

		# ========== work out the stippling ==========
		slats, slons = _stippling(ds, squeeze=10, nanfrac = 0.15, sigfrac=0.5)
		ax.scatter(
			slons, slats, s=4, c='k', marker='X', 
			facecolors='none', edgecolors="none",  
			alpha=0.35, transform=ccrs.PlateCarree())
		ax.gridlines()
		coast = cpf.GSHHSFeature(scale="intermediate")
                                        # edgecolor='face',
                                        # facecolor=cfeature.COLORS['land']

		# p.axes.add_feature(cpf.COASTLINE, , zorder=101)
		ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
		ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
		ax.add_feature(coast, zorder=101, alpha=0.5)
		ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		ax.add_feature(cpf.RIVERS, zorder=104)
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
		# ========== Set the titles ==========
		vanm = setup[va]["lname"]
		ax.set_title("")
		ax.set_title(f"{let}) {vanm}", loc= 'left')
		print(f"Annual trend {va}", ds.slope.quantile([0.01,0.05, 0.50,0.95,0.99]))

		
	# ========== Save the plots ==========

	plt.subplots_adjust(top=0.971, bottom=0.013, left=0.012, right=0.988, hspace=0.063, wspace=0.2)
	plotfname = f"{ppath}PF03_AnnualClimateTrend."
	for fmt in ["pdf", "png"]:
		print(f"Starting {fmt} plot save at:{pd.Timestamp.now()}")
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

# ==============================================================================
def _stippling(ds, squeeze=10, nanfrac = 0.25, sigfrac=0.5):
	"""
	function to change calculate the stipling
	"""
	# /// pull fraction sig \\\\
	da = (ds.FDRsig.squeeze("time").coarsen({"latitude":squeeze, "longitude":squeeze}, boundary ="pad").mean())>sigfrac#).astype(int)
	# /// pull the nan fraction \\\\
	da *= (ds.FDRsig.squeeze("time").isnull().coarsen({"latitude":squeeze, "longitude":squeeze}, boundary ="pad").mean()<nanfrac)
	da = da.astype(np.float32).where(da)
	# /// convert to a dataframe \\\\
	df = xr.Dataset({"sig":da}).to_dataframe().drop("time", axis=1).dropna().reset_index()
	return df.latitude.values, df.longitude.values

def setupfunc(time):
	cmaps = _cmapsfun()
	# ========== Build an ordered dict with key info ==========
	setup = OrderedDict()
	if time == "seasonal":
		setup["ppt"]   = ({"vmin":-2., "vmax":2, "cmap":cmaps["ppt"], "lname":"Precipitation",
			"attrs":{'long_name':"Trend", "units":r"mm yr$^{-1}$"}})
		setup["tmean"] = ({"vmin":-0.08, "vmax":0.08, "cmap":cmaps["tmean"], "lname":"Temperature",
			"attrs":{'long_name':"Trend", "units":r"$^{o}$C yr$^{-1}$"}})
	elif time == "annual":
		setup["ppt"]   = ({"vmin":-4., "vmax":4, "cmap":cmaps["ppt"], "lname":"Precipitation",
			"attrs":{'long_name':"Trend", "units":r"mm yr$^{-1}$"}})
		setup["tmean"] = ({"vmin":-0.06, "vmax":0.06, "cmap":cmaps["tmean"], "lname":"Temperature",
			"attrs":{'long_name':"Trend", "units":r"$^{o}$C yr$^{-1}$"}})
	return setup

def _cmapsfun():
	"""
	Funtion to make the colourmaps 

	"""
	
	pptcmap = mpc.ListedColormap(palettable.colorbrewer.diverging.BrBG_11.mpl_colors)
	pptcmap.set_bad('dimgrey',1.)
	
	# tmncmap = mpc.ListedColormap(palettable.colorbrewer.diverging.PuOr_11_r.mpl_colors)
	# tmncmap = mpc.ListedColormap(palettable.colorbrewer.diverging.RdBu_11_r.mpl_colors)
	tmncmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_19.mpl_colors)
	tmncmap.set_bad('dimgrey',1.)
	return {"ppt":pptcmap, "tmean":tmncmap}




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
	elif sysname == 'DESKTOP-T77KK56':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		backpath = "/mnt/f/fireflies"
		chunksize = 500
	elif sysname == 'DESKTOP-KMJEPJ8':
		dpath = "./data"
		# backpath = "/mnt/g/fireflies"
		cpath = "/mnt/i/Data51/Climate/TerraClimate/"
	elif sysname == 'arden-worstation':
		# WHRC linux distro
		dpath = "./data"
		cpath= "/media/arden/SeagateMassStorage/Data51/Climate/TerraClimate/"
		# dpath= "/media/arden/Harbinger/Data51/BurntArea"
	elif sysname == 'LAPTOP-8C4IGM68':
		dpath     = "./data"
		backpath = "/mnt/d/fireflies"
	else:
		ipdb.set_trace()
	
	return dpath, cpath

# ==============================================================================

if __name__ == '__main__':
	main()