"""
Make maps of the forest cover type

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

print("numpy version   : ", np.__version__)
print("pandas version  : ", pd.__version__)
print("xarray version  : ", xr.__version__)
print("cartopy version : ", ct.__version__)
print("sklearn version : ", skl.__version__)

#==============================================================================
def main():
	# ========== find the paths ==========
	dpath, cpath = syspath()
	ppath = "./plots/ShortPaper/PF07_Regions/"
	cf.pymkdir(ppath)
	pbounds = [10.0, 170.0, 70.0, 49.0]
	maskver = "Boreal"
	setup   = setupfunc()
	formats = [".png", ".tiff", ".eps"]# ".pdf"
	vnames  = ["LandCover", "TreeSpecies"]
	AnnualPlotmaker(setup, dpath, cpath, ppath, pbounds, maskver, formats, vnames)


#==============================================================================
def AnnualPlotmaker(setup, dpath, cpath, ppath, pbounds, maskver, formats, vnames, alpC  = ["a", "b", "c", "d"]):
	""" Function fo making the annual plot
	args:
		setup: Ordered dict 
			contains the cmap and vrng infomation
		dpath:	str
		cpath:	str
			path to the climate data
	"""
	nrows=len(vnames)
	# ========== load the mask ==========
	fnmask = f"{dpath}/masks/broad/Hansen_GFC-2018-v1.6_SIBERIA_ProcessedToTerraClimate.nc"
	fnBmask = f"./data/LandCover/Regridded_forestzone_TerraClimate.nc"
	bpath = "./data/LandCover/Bartalev"
	fnTree = f"{bpath}/Bartalev_TreeSpecies_ProcessedToTerraClimate.nc"

	with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask, xr.open_dataset(fnBmask).drop(["DinersteinRegions", "GlobalEcologicalZones", "LandCover"]) as Bmask:
		# breakpoint()
		if maskver == "Boreal":
			msk    = (dsmask.mask.isel(time=0)*((Bmask.BorealMask.isel(time=0)>0).astype("float32")))#.sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
		else:
			msk    = (dsmask.mask.isel(time=0)).astype("float32")
		
		msk = msk.values

		# +++++ Change the boolean mask to NaNs +++++
		msk[msk == 0] = np.NAN
		
		# print("Masking %s frame at:" % dsn, pd.Timestamp.now())
		# +++++ mask the frame +++++
		# breakpoint()
		# frame *= msk
	latiMid=np.mean([pbounds[2], pbounds[3]])
	longMid=np.mean([pbounds[0], pbounds[1]])
	# ========== set the mpl rc params ==========
	font = ({
		'weight' : 'bold',
		'size'   : 14, 
		})

	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})
	
	# ========== Create the figure ==========
	fig, axs = plt.subplots(
		nrows, 1, sharex=True, subplot_kw={'projection': ccrs.Orthographic(longMid,latiMid)}, 
		figsize=(15, nrows * 6)
		)
	for va, ax, let in zip(setup, axs.flatten(), alpC):
		if va == "LandCover":
			# ========== load the dataset and pull out the relevant parts ==========
			ds = xr.open_dataset(fnBmask)
		else:
			ds = xr.open_dataset(fnTree)
		# +++++ remap the values +++++
		for vrm in setup[va]['valmap']:	
			ds[va] = ds[va].where(~(ds[va] == vrm), setup[va]['valmap'][vrm])
		ds[va] = np.fabs(ds[va])
		ds[va].attrs = setup[va]["attrs"]
		# if setup[va]["mask"]:
		# 	ds[va] *= msk
		
		p  = ds[va].plot(
			cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
			transform=ccrs.PlateCarree(), ax=ax,
			    cbar_kwargs={
			    "pad": 0.015, "shrink":0.80, "extend":"neither"
			    })
		cbar = p.colorbar
		keys =  pd.DataFrame({va:setup[va]["kys"]}).reset_index()
		# cbar.set_ticklabels( keys.Code.values)  # horizontal colorbar

		cbar.set_ticks( keys["index"].values)
		cbar.set_ticklabels(keys[va].values)
		axtitle = f"{let})" #  {setup[va]['lname']}

		if va == "LandCover":
			# ========== bring in the locations ==========
			df_loc = setup[va]['places']
			ax.scatter(
				df_loc.lon.values, df_loc.lat.values, s=20, c='k', 
				facecolors='none', edgecolors="none",  
				alpha=1, transform=ccrs.PlateCarree())
			for index, row in df_loc.iterrows():

				ax.text(
					row.lon+row.lonoffset, row.lat+row.latoffset, row.Name, 
					horizontalalignment='left',transform=ccrs.PlateCarree(), 
					zorder=105, color="lightgray", path_effects=[pe.withStroke(linewidth=0.75, foreground="black")])


		ax.set_extent(pbounds, crs=ccrs.PlateCarree())
		gl = ax.gridlines(draw_labels= True, dms=True, x_inline=False, y_inline=False)#{"bottom": "x", "Top": "y"}
		gl.xlocator = mticker.FixedLocator([60, 120])
		gl.ylocator = mticker.FixedLocator([50, 60, 70])
		# gl.xlabels_top = False
		# gl.top_labels   = False
		# gl.right_labels = False
		# ax.gridlines()
		# plt.show()
		# breakpoint()
		coast = cpf.GSHHSFeature(scale="intermediate")
		# p.axes.add_feature(cpf.COASTLINE, , zorder=101)
		ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
		ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
		ax.add_feature(coast, zorder=101, alpha=0.5)
		ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		ax.add_feature(cpf.RIVERS, zorder=104)
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
		# ========== Set the titles ==========
		ax.set_title("")
		ax.set_title(axtitle, loc= 'left')

	# plt.subplots_adjust(top=0.96,bottom=0.013,left=0.008,right=0.993,hspace=0.090,wspace=0.0)
	plt.subplots_adjust(top=0.95,bottom=0.020,left=0.008,right=0.967,hspace=0.156,wspace=0.0)
	# ========== make the plot name ==========
	plotfname = ppath + f"PF07_ForestCoverV2"
	# if mask:
	# 	plotfname += "_ForestMask_V2"
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
	# ax.set_title(axtitle, loc= 'left')

#==============================================================================

def setupfunc(shrink = 0.90):
	cmaps = _cmapsfun()
	# ========== Build an ordered dict with key info ==========
	setup = OrderedDict()
	
	# ========== make the kes foir the figure ==========
	df_lc = pd.read_csv("./data/LandCover/glc2000_v1_1/Tiff/Global_Legend.csv")
	df_lc["GROUP"].replace(0, np.NaN,inplace=True)
	df_lc["GROUP"].replace(1, np.NaN,inplace=True)
	exc = OrderedDict()
	for val, gp in zip(np.flip(df_lc.VALUE.values), np.flip(df_lc.GROUP.values)):
		exc[val]= -gp
	# kys = ({0:"FBD", 1:"FCE", 2:"FCD", 3:"FMx", 4:"SHC", 5:"CMA", 6:"BG", 7:"WSI", 8:"Oth"})
	# breakpoint()
	kys = ({ 2:"BG", 3:"CMA", 4:"SHC", 5:"FMx", 6:"FCD", 7:"FCE", 8:"FBD"})#, 1:"WSI",

	setup["LandCover"] = ({"vmin":1.5, "vmax":8.5, "cmap":cmaps["LandCover"],"lname":"Land Cover",
		"valmap":exc, "kys":kys, "attrs":{'long_name':"Land Cover Class"}, "places": _locations(), 
		"shrink":shrink, "mask":False})
	# ========== Do the tree species ==========
	bpath = "./data/LandCover/Bartalev"
	tsfn = f"{bpath}/Land_cover_map_Bartalev/BartalevLookup.csv"
	df_ts = pd.read_csv(tsfn) 
	df_ts["Group"].replace(0, np.NaN,inplace=True)

	exct = OrderedDict()
	kyst = OrderedDict()
	for val, gp, sp in zip(df_ts["Value"].values, df_ts["Group"].values, df_ts["Species"].values):
		exct[val]= gp
		if gp > 0:
			if not gp in kyst.keys():
				kyst[gp] = sp
	
	setup["TreeSpecies"] = ({"vmin":.5, "vmax":9.5, "cmap":cmaps["TreeSpecies"],"lname":"Tree Species",
		"valmap":exct, "kys":kyst, "attrs":{'long_name':"Dominate Tree Species"}, "places": None, 
		"shrink":shrink, "mask":False})
			

	# ========== Deinstine regions ==========
	# setup["DinersteinRegions"] = ({"vmin":0, "vmax":7, "cmap":cmaps["pptC"],"lname":"Land Cover",
	# 	})


	return setup

def _cmapsfun():
	"""
	Funtion to make the colourmaps 

	"""
	pcmap = palettable.colorbrewer.diverging.BrBG_11.mpl_colormap
	cmapP = []
	for point  in np.arange(0, 1.01, 0.05):
		cmapP.append( pcmap(point)[0:-1])

	pptcmap = mpc.ListedColormap(cmapP)	
	# pptcmapC = mpc.ListedColormap(palettable.colorbrewer.diverging.BrBG_10.mpl_colors)

	pcmapC = palettable.colorbrewer.diverging.BrBG_10.mpl_colormap
	cmap = []
	for point  in np.arange(0, 1.01, 0.05):
		cmap.append( pcmapC(point)[0:-1])

	pptcmapC = mpc.ListedColormap(cmap)
	# breakpoint()

	pptcmap.set_bad('dimgrey',1.)
	pptcmapC.set_bad('dimgrey',1.)
	
	# tmncmap = mpc.ListedColormap(palettable.colorbrewer.diverging.PuOr_11_r.mpl_colors)
	# tmncmap = mpc.ListedColormap(palettable.colorbrewer.diverging.RdBu_11_r.mpl_colors)
	tmncmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_19.mpl_colors)
	tmncmap.set_bad('dimgrey',1.)

	tmncmapC = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors)
	tmncmapC.set_bad('dimgrey',1.)

	
	LCcmap = mpc.ListedColormap(palettable.cartocolors.qualitative.Prism_7_r.mpl_colors)
	LCcmap.set_bad('dimgrey',1.)
	
	TScmap = mpc.ListedColormap(palettable.cartocolors.qualitative.Bold_9.mpl_colors)
	TScmap.set_bad('dimgrey',1.)

	return {"ppt":pptcmap, "tmean":tmncmap, "pptC":pptcmapC, "tmeanC":tmncmapC, "LandCover":LCcmap, "TreeSpecies":TScmap}



def _locations():
	""" Fiunction to make a dataframe of different locations """
	Loc = OrderedDict()
	Loc[0] = {"Name":"Moscow",      "lat": 55.756303, "lon":37.616777,  "latoffset":-1.0, "lonoffset":2.0}
	Loc[1] = {"Name":"Chita",       "lat": 52.051446, "lon":113.471214, "latoffset":-1.0, "lonoffset":0.5}
	Loc[2] = {"Name":"Krasnoyarsk", "lat": 56.015283, "lon":92.893248,  "latoffset":-1.0, "lonoffset":0.5}
	Loc[3] = {"Name":"Irkutsk",     "lat": 52.286286, "lon":104.295314, "latoffset":-1.5, "lonoffset":0.0}
	Loc[4] = {"Name":"Yakutsk",     "lat": 62.039691, "lon":129.742219, "latoffset":-1.0, "lonoffset":0.5}
	Loc[5] = {"Name":"Chelyabinsk", "lat": 55.164521, "lon":61.437660,  "latoffset":-1.0, "lonoffset":1.0}
	Loc[6] = {"Name":"Vladivostok", "lat": 43.133297, "lon":131.911234, "latoffset":-0.25,   "lonoffset":-7.0}
	# Loc[7] = {"Name":"Stockholm",   "lat": 18.068581, "lon": 59.329324, "latoffset":-1.0, "lonoffset":2.0}
	# Loc[] = {"Name":"", "lat":, "lon":, "latoffset":-1.0, "lonoffset":0.5}
	# Loc[] = {"Name":"", "lat":, "lon":, "latoffset":-1.0, "lonoffset":0.5}
	# Loc[] = {"Name":"", "lat":, "lon":, "latoffset":-1.0, "lonoffset":0.5}
	return pd.DataFrame(Loc).T

def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	backpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "/mnt/d/Data51"
		cpath = "/mnt/g/Data51/Climate/TerraClimate/"
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
		dpath = "/media/ubuntu/Harbinger/Data51"
	# elif 'ccrc.unsw.edu.au' in sysname:
	# 	dpath = "/srv/ccrc/data51/z3466821"
	elif sysname == 'DESKTOP-N9QFN7K':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		cpath = "/mnt/f/Data51/Climate/TerraClimate/"
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
		# backpath = "/mnt/d/fireflies"
		cpath = "/mnt/d/Data51/Climate/TerraClimate/"
	else:
		ipdb.set_trace()
	
	return dpath, cpath

# ==============================================================================

if __name__ == '__main__':
	main()