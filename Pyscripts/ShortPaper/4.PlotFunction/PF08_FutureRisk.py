"""
Make maps of the the future risk cats

"""

#==============================================================================

__title__ = "Future Risk Calculator"
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
	# dpath, cpath = syspath()
	ppath = "./plots/ShortPaper/PF08_FutRisk/"
	cf.pymkdir(ppath)
	mask = True
	bounds = [10.0, 170.0, 70.0, 49.0]
	maskver = "Boreal"
	proj    = "polar"
	# setup   = setupfunc()
	formats = [".png", ".pdf"]


	# dsnams1 = ["GFED", "MODIS", "esacci", "COPERN_BA"]#, "HANSEN_AFmask", "HANSEN"]
	# dsnams2 = ["HANSEN_AFmask", "HANSEN", "SRfrac"]
	# dsnams3 = ["Risk"]
	scale   = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20, "Risk":20, "SRfrac":20, "FutureRisk":20})
	
	info = OrderedDict()
	info ["FutureRisk"] = ({
		"fnpre":"./results/ProjectSentinal/FRImodeling/S03_FRIdrivers_esacci_v0_100yr_trendPrediction_forests_sigclim.nc",
		"fnSRF":""
		})
	
	compath, backpath = syspath()
	TCF     = 10
	if TCF == 0:
		tcfs = ""
	else:
		tcfs = "_%dperTC" % np.round(TCF)

	dsinfo  = dsinfomaker(compath, backpath, 1, tcfs)
	datasets = OrderedDict()
	datasets["FutureRisk"] = dsinfo["FutureRisk"]['fname']

	plotmaker(dsinfo, datasets, 1, ppath, formats, mask, compath, backpath, proj, scale, bounds, maskver)
	breakpoint()

	# AnnualPlotmaker(setup, dpath, cpath, ppath, pbounds, maskver, formats,)

#==============================================================================

def plotmaker(dsinfo, datasets, mwb, plotdir, formats, mask, compath, backpath, 
	proj, scale, bounds, maskver, dsn="FutureRisk", region = "SIBERIA",):
	"""Function builds a basic stack of maps """

	# ========== make the plot name ==========
	plotfname = plotdir + "PF08_FutureForestLossRisk_MW_%02dDegBox_V2_%s_%s" % (mwb, proj, "_".join(datasets.keys()))
	if mask:
		plotfname += "_ForestMask_V2"

	# ========== Setup the font ==========
	# ========== set the mpl rc params ==========
	font = ({
		'weight' : 'bold',
		'size'   : 11, 
		})
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# mpl.rc('font', **font)
	# plt.rcParams.update({'axes.titleweight':"bold", }) #'axes.titlesize':mapdet.latsize
		
	# ========== setup the figure ==========
	if proj == "polar":
		latiMid=np.mean([bounds[2], bounds[3]])
		longMid=np.mean([bounds[0], bounds[1]])
		# if len(datasets) == 4:
		# 	yv = 2
		# 	xv = 2
		# 	shrink=0.80
		# else:
		yv = 1
		xv = 1
		shrink=0.95

		fig, ax = plt.subplots(
			yv, xv, figsize=(12,5*len(datasets)), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})

	if not os.path.isfile(dsinfo["FutureRisk"]['fname']):
		_FutRiskBuilder(dsinfo,compath, backpath, scale, xbounds = [-10.0, 180.0, 70.0, 40.0])

	frame = _fileopen(dsinfo, datasets, "FutureRisk", "FutureForestLossRisk", scale, 
		proj, mask, compath, region, bounds, maskver, func = "mean")

	# ========== Set the colors ==========
	cmap, norm, vmin, vmax, levels = _colours( "FutureForestLossRisk", 3.5, dsn)

	# ========== Create the Title ==========
	title = ""
	extend = "neither"

	# ========== Grab the data ==========
	if proj == "polar":
		# .imshow
		# breakpoint()

		im = frame.compute().plot(
			ax=ax, vmin=vmin, vmax=vmax, 
			cmap=cmap, norm=norm, 
			transform=ccrs.PlateCarree(),
			# add_colorbar=False,
			cbar_kwargs={"pad": 0.02, "extend":extend, "shrink":shrink, "ticks":levels, "spacing":"uniform"}
			) #
			# subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)}
		if dsn in ["FutureRisk", "Risk"]:
			cbar = im.colorbar
			keys =  pd.DataFrame(_riskkys()).T
			# cbar.set_ticklabels( keys.Code.values)  # horizontal colorbar
			cbar.set_ticklabels( keys.FullName.values)
			# 
		# breakpoint()
		ax.set_extent(bounds, crs=ccrs.PlateCarree())
		ax.gridlines()

	else:
		breakpoint()


	# ========== Add features to the map ==========
	coast_50m = cpf.GSHHSFeature(scale="high")
	ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(coast_50m, zorder=101, alpha=0.5)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.set_aspect('equal')
	plt.subplots_adjust(top=0.971,bottom=0.013,left=0.008,right=0.98,hspace=0.063,wspace=0.0)

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

	



def _FutRiskBuilder(dsinfo, compath, backpath, scale, xbounds = [-10.0, 180.0, 70.0, 40.0]): 

	# xbounds = [140, 150.0, 60.0, 50.0]
	# ========== Riskbuilder ==========
	ds_dsn = xr.open_dataset("./results/ProjectSentinal/FRImodeling/S03_FRIdrivers_esacci_v0_100yr_trendPrediction_forests_sigclim.nc")
	# ========== Get the data for the frame ==========
	frameAB = ds_dsn["AnBF_XGBoost_fut"].sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))

	

	frame = 1/frameAB
	FRI15 = (frame <=15).astype("int16")
	FRI30 = (frame <=30).astype("int16")
	frame = None
	
	# ========== work out the less than SR ==========
	ds_SR  = xr.open_dataset(dsinfo["SRfrac"]["fname"])
	frameSR = ds_SR["StandReplacingFireFraction"].sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))

	# ========== Fetch the  FRIsr ==========
	frameSR = frameSR.where(~(frameSR >1), 1)
	frameSR["time"] = frameAB["time"]
	SR_da   = 1/(frameAB.values * frameSR)


	SRI60  = (SR_da <= 60).astype("int16")
	SRI120 = (SR_da <=120).astype("int16")
	SR_da  = None

	# ========= Workout my fire risk catogeries ==========
	HR = (SRI120 * FRI30.values) # High Risk Fire
	CR = np.logical_or(FRI15.values, SRI60).astype("int16").where(HR == 1, 0)
	MR = np.logical_or(FRI30.values, SRI120).astype("int16")
	
	# +++++ Cleanup +++++
	SRI60  = None
	SRI120 = None
	ds_SRI = None

	# ========= Workout my Dist risk catogeries ==========
	# ds_DRI = xr.open_dataset(dsinfo["HANSEN"]["fname"])
	# DR_da  = ds_DRI["FRI"].sortby("latitude", ascending=False).sel(
	# 	dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
	# DR_da["time"] = frameAB["time"]
	# DRI60  = (DR_da <= 60).astype("int16")
	# DRI120 = (DR_da <=120).astype("int16")
	# DR_da  = None

	# HRD = np.logical_or(HR, (DRI120 * FRI30)) # High Risk all
	# CRD = np.logical_or(FRI15, DRI60).astype("int16").where(HRD == 1, 0)
	# # CRD = np.logical_or(FRI15, SRI60).astype("int16").where(HR == 1, 0)
	# MRD = np.logical_or(MR,(np.logical_or(FRI30, DRI120))).astype("int16")
	
	# DRI60  = None
	# DRI120 = None
	FRI15  = None

	def _quickplot (da, scale, dsn):
		
		with ProgressBar():
			dac = da.coarsen(
				{"latitude":scale[dsn]*2, "longitude":scale[dsn]*2
				}, boundary ="pad").max().compute()
		dac.plot(vmin=1)
		plt.show()
	# breakpoint()
	Risk = MR+HR+CR#CRD+MRD+HRD
	Risk["time"] = [pd.Timestamp("2100-12-31")]
	# breakpoint()
	# _quickplot((Risk==1).astype("int16"), scale, dsn)
	ds_risk = xr.Dataset({"FutureForestLossRisk":Risk})
	GlobalAttributes(ds_risk, fnameout=dsinfo["FutureRisk"]['fname'])
	ds_risk.to_netcdf(dsinfo["FutureRisk"]['fname'], format = 'NETCDF4', unlimited_dims = ["time"])
	print("Risk Dataset Built")
	_quickplot (Risk, scale, "FutureRisk")
	# breakpoint()


#==============================================================================

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
				}, boundary ="pad").mean().compute()
		elif func == "max":
			frame = frame.coarsen(
				{"latitude":scale[dsn], "longitude":scale[dsn]
				}, boundary ="pad").max().compute()
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

def _colours(var, vmax, dsn):
	norm=None
	levels = None

	if var ==  "ForestLossRisk":
		vmin = -0.5
		cmapHex = palettable.cartocolors.qualitative.Prism_9.hex_colors[2:]
		levels = [0, 1, 2, 3, 4, 5, 6]
		cmap    = mpl.colors.ListedColormap(cmapHex)
		cmap.set_bad('dimgrey',1.)
	elif var ==  "FutureForestLossRisk":
		vmin = -0.5
		cmapHex = palettable.cartocolors.qualitative.Prism_9.hex_colors[2:]
		levels = [0, 1, 2, 3]#, 4, 5, 6]
		# breakpoint()
		cmap    = mpl.colors.ListedColormap(np.array(cmapHex)[[0, 2, 4, 6]].tolist())
		cmap.set_bad('dimgrey',1.)
		# vmax=3.5
	elif var == "StandReplacingFireFraction":
		vmin = 0.01
		vmax = 100
		cmapHex = palettable.cmocean.diverging.Curl_20.hex_colors#[2:] #Matter_11
		# levels  = np.arange(vmin, vmax+0.05, 0.10)
		cmap    = mpl.colors.ListedColormap(cmapHex)#[1:-1])
		cmap.set_bad('dimgrey',1.)
		cmap.set_over(cmapHex[-1])
		cmap.set_under(cmapHex[0])
		norm=LogNorm(vmin = vmin, vmax = vmax)
	else:
		# ========== Set the colors ==========
		vmin = 0.0
		vmax = 0.20

		# +++++ create the colormap +++++
		# cmapHex = palettable.matplotlib.Inferno_10.hex_colors
		# cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors
		cmapHex = palettable.colorbrewer.sequential.OrRd_9.hex_colors
		

		cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
		cmap.set_over(cmapHex[-1] )
		cmap.set_bad('dimgrey',1.)
		breakpoint()

	return cmap, norm, vmin, vmax, levels

def _riskkys(full=False):
	keys = OrderedDict()
	if full:
		keys[0] = {"Code":"LR",   "FullName":"Low Risk"}
		keys[1] = {"Code":"MRd",  "FullName":"Mod. Risk (dist)"}
		keys[2] = {"Code":"MRf",  "FullName":"Mod. Risk (fire)"}
		keys[3] = {"Code":"HRd",  "FullName":"High Risk (dist)"}
		keys[4] = {"Code":"HRf",  "FullName":"High Risk (fire)"}
		keys[5] = {"Code":"VHRd", "FullName":"Extreme Risk (dist)"}
		keys[6] = {"Code":"VHRf", "FullName":"Extreme Risk (fire)"}
	else:
		keys[0] = {"Code":"LR",   "FullName":"Low Risk"}
		keys[1] = {"Code":"MRf",  "FullName":"Mod. Risk (fire)"}
		keys[2] = {"Code":"HRf",  "FullName":"High Risk (fire)"}
		keys[3] = {"Code":"VHRf", "FullName":"Extreme Risk (fire)"}


	return keys
	

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

def GlobalAttributes(ds, fnameout=""):
	"""
	Creates the global attributes for the netcdf file that is being written
	these attributes come from :
	https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/metadata/DataDiscoveryAttConvention.html
	args
		ds: xarray ds
			Dataset containing the infomation im intepereting
		fnout: str
			filename out 
	returns:
		attributes 	Ordered Dictionary cantaining the attribute infomation
	"""
	# ========== Create the ordered dictionary ==========
	if ds is None:
		attr = OrderedDict()
	else:
		attr = ds.attrs

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["FileName"]            = fnameout
	attr["title"]               = "RiskFramework"
	attr["summary"]             = "BorealForestLossRisk" 
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. " % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
	
	if not ds is None:
		attr["history"]            += ds.history

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "Woodwell"
	attr["date_created"]        = str(pd.Timestamp.now())
	ds.longitude.attrs['units'] = 'degrees_east'
	ds.latitude.attrs['units']  = 'degrees_north'

	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	# attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	# attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	return attr	
# ==============================================================================
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
	dsinfo["FutureRisk"]    = ({"alias":"Forest Loss Risk"})
	dsinfo["SRfrac"]        = ({"alias":"Stand Replacing Fire Percentage", "long_name":f'FRI$_{{{"SR"}}}$ %'})

	for dsnm in dsinfo:
		if dsnm.startswith("H"):
			# +++++ make a path +++++
			ppath = compath + "/BurntArea/HANSEN/FRI/"
			fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, tcfs, mwb)
			# fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		elif dsnm in ["FutureRisk", "Risk"]:
			ppath = compath + "/BurntArea/Risk/FRI/"
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
			cf.pymkdir(ppath)
		else:
			# fname = "Hansen_GFC-2018-v1.6_regrided_esacci_FRI_%ddegMW_SIBERIA" % (mwb)
			ppath = compath + "/BurntArea/%s/FRI/" %  dsnm
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		# +++++ open the datasets +++++
		dsinfo[dsnm]["fname"] = ppath+fname


	return dsinfo
# ==============================================================================

if __name__ == '__main__':
	main()