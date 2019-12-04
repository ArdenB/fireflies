"""
Script goal, 

Build evaluation maps of GEE data

"""
#==============================================================================

__title__ = "GEE Movie Fixer"
__author__ = "Arden Burrell"
__version__ = "v1.0(04.04.2019)"
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
import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob

from collections import OrderedDict
from scipy import stats
from numba import jit
from dask.diagnostics import ProgressBar

# Import the Earth Engine Python Package
# import ee
# import ee.mapclient
# from ee import batch

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 

# import fiona
# fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
# fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default


# import moviepy.editor as mpe
# import skvideo.io     as skv
# import skimage as ski
# from moviepy.video.io.bindings import mplfig_to_npimage


# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
import ipdb

# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

# ==============================================================================

def main():

	# # ========== Read in the Site data ==========
	# df = pd.read_csv("./results/ProjectSentinal/Fire sites and forest loss.csv") 
	# df = renamer(df)

	# ========== Read in the Site data ==========
	cordname    = "./data/other/GEE_sitelist.csv"
	site_coords = pd.read_csv(cordname, index_col=0)

	# ========== Find the save lists ==========
	spath, dpath = syspath()
	data         = datasets(dpath)
	
	# ========== Determine if i show the plots ==========
	plot = False
	# plot = True


	# ========== Set the site up ==========
	for index, dfg in site_coords.iterrows():
		site = dfg["name"]
		print(site, " Starting at: ", pd.Timestamp.now())
		# ========== Get infomation about that site ==========
		path = SiteInfo(site, dpath, spath)
		# cf.pymkdir(path)

		# ========== Make the hansen forest maps ==========
		for dsn in ["esacci", "MODIS"]:
			MODISbox(dsn, path, dfg, site, data, dpath, plot)
			# ipdb.set_trace()

		# esacci(path, dfg, site, data, dpath, plot)

		Hansen("HansenGFL", path, dfg, site, data, dpath, plot)
		# plot = True
		copern("COPERN_BA", path, dfg, site, data, dpath, plot)
	ipdb.set_trace()

# ==============================================================================
# ====================== BA product functions functions ========================
# ==============================================================================
def MODISbox(dsn, path, dfg, site, data, dpath, plot):
	"""
	function to make the copernicous BA product maps
	"""
	# ========== Open the data ==========
	if "*" in data[dsn]["fname"]:
		fnames = glob.glob(data[dsn]["fname"])
		# ===== open the dataset =====
		ds = xr.open_mfdataset(fnames, chunks=data[dsn]["chunks"], combine='nested', concat_dim="time")
		# ========== fiz a time issue ==========
		if dsn == "COPERN_BA":
			ds["time"] = pd.date_range('2014-06-30', periods=6, freq='A')
		elif dsn == "esacci":
			ds = ds.sortby("latitude", ascending=False)
	else:
		# ========== open the dataset ==========
		ds = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
	
	with ProgressBar():
		ds_in = ds[data[dsn]["var"]].sel(dict(latitude =slice(dfg.loc["latr_max"], dfg.loc["latr_min"]), 
			longitude=slice(dfg.loc["lonr_min"], dfg.loc["lonr_max"]))).astype("float32").compute()
	
	# ========== Loop over each year ==========
	print(dsn)
	for date in ds_in.time.values:
		ds_sel = ds_in.sel(time=date).astype(float)
		if ds_sel.sum() == 0:
			continue
	
		# ========== Set up the plot ==========
		fig, ax = plt.subplots(1, figsize=(11,10))
		# plt.title("MODIS %d" % pd.Timestamp(date).year)
		ds_sel.plot.imshow(
			ax=ax,
			vmin=0, 
			vmax=1, 
			)
		ax.scatter(dfg.loc["lon"], dfg.loc["lat"], 5, c='r', marker='+')
		rect = mpl.patches.Rectangle(
			(dfg.loc["lonb_MOD_min"],dfg.loc["latb_MOD_min"]),
			dfg.loc["lonb_MOD_max"]-dfg.loc["lonb_MOD_min"],
			dfg.loc["lonb_MOD_max"]-dfg.loc["lonb_MOD_min"],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		# ax.set_title(None)#"%s %s" % (info.satellite, info.date.split(" ")[0]))
		plt.axis('scaled')

		fnout = "%sBAimages/%s_%s_%s_%d.png" % (path, dsn, site, data[dsn]["var"], pd.Timestamp(date).year) 
		plt.savefig(fnout)
		if plot:
			plt.show()

		ax.clear()
		plt.close()
		# ipdb.set_trace()

	# ipdb.set_trace()

def copern(dsn, path, dfg, site, data, dpath, plot):
	"""
	function to make the copernicous BA product maps
	"""

	# data["COPERN_BA"] = ({
	# 	'fname':"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/M0044633/c_gls_BA300_201812200000_GLOBE_PROBAV_V1.1.1.nc",
	# 	'var':"BA_DEKAD", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
	# 	"start":2014, "end":2019,"rasterio":False, "chunks":None, 
	# 	"rename":{"lon":"longitude", "lat":"latitude"}
	# 	})
	# for yr in range(2014, 2020):
	

	# ========== Open the data ==========
	if "*" in data[dsn]["fname"]:
		fnames = glob.glob(data[dsn]["fname"])
		# ===== open the dataset =====
		ds = xr.open_mfdataset(fnames, chunks=data[dsn]["chunks"], combine='nested', concat_dim="time")
		# ========== fiz a time issue ==========
		if dsn == "COPERN_BA":
			ds["time"] = pd.date_range('2014-06-30', periods=6, freq='A')
	else:
		# ========== open the dataset ==========
		ds = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
	
	with ProgressBar():
		ds_in = ds[data[dsn]["var"]].sel(dict(latitude =slice(dfg.loc["latr_max"], dfg.loc["latr_min"]), 
			longitude=slice(dfg.loc["lonr_min"], dfg.loc["lonr_max"]))).astype("float32").compute()

	
	print(dsn)
	for date in ds_in.time.values:
		ds_sel = ds_in.sel(time=date).astype(float)
		if ds_sel.sum() == 0:
			continue
	
		# ========== Set up the plot ==========
		fig, ax = plt.subplots(1, figsize=(11,10))

		ds_sel.plot.imshow(
			ax=ax,
			vmin=0, 
			vmax=1, 
			)

		ax.scatter(dfg.loc["lon"], dfg.loc["lat"], 5, c='r', marker='+')
		rect = mpl.patches.Rectangle(
			(dfg.loc["lonb_COP_min"],dfg.loc["latb_COP_min"]),
			dfg.loc["lonb_COP_max"]-dfg.loc["lonb_COP_min"],
			dfg.loc["lonb_COP_max"]-dfg.loc["lonb_COP_min"],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		# ax.set_title(None)#"%s %s" % (info.satellite, info.date.split(" ")[0]))
		plt.axis('scaled')
		
		fnout = "%sBAimages/PROBAV_%s_%s_%d.png" % (path, site, data[dsn]["var"], pd.Timestamp(date).year) 
		plt.savefig(fnout)
		if plot:
			plt.show()

		ax.clear()
		plt.close()
		# ipdb.set_trace()

	# ipdb.set_trace()
	
def Hansen(dsn, path, dfg, site, data, dpath, plot):
	
	if "*" in data[dsn]["fname"]:
		fnames = glob.glob(data[dsn]["fname"])
		# ===== open the dataset =====
		ds = xr.open_mfdataset(fnames, chunks=data[dsn]["chunks"], combine='nested', concat_dim="time")
		# ========== fiz a time issue ==========
		if dsn == "COPERN_BA":
			ds["time"] = pd.date_range('2014-06-30', periods=6, freq='A')
	else:
		# ========== open the dataset ==========
		ds = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
	
	with ProgressBar():
		ds_in = ds[data[dsn]["var"]].sel(dict(latitude =slice(dfg.loc["latr_max"], dfg.loc["latr_min"]), 
			longitude=slice(dfg.loc["lonr_min"], dfg.loc["lonr_max"]))).astype("float32").compute()
		ds_in += 2000
	# ========== fiz the values ==========
	# ds_in = ds_in.where(site_dm["lossyear"].values == 1)
	ds_in = ds_in.where(ds_in >= 2001, 0)

	# ========== Create the plot ==========
	cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_18.mpl_colors)
	cmap.set_under('w')
	cmap.set_bad('dimgrey',1.)
	tks, counts = np.unique(ds_in.values, return_counts=True)
	ticks = tks[np.logical_and((counts > 100), (tks>0))]
	
	# ========== Set up the plot ==========
	fig, ax = plt.subplots(1, figsize=(11,10))
	ax.clear()
	ds_in.rename(None).isel(time=0).plot.imshow(
		ax=ax,
		vmin=2000.5, 
		vmax=2018.5, 
		cmap=cmap, 
		cbar_kwargs={'ticks':ticks} )

	ax.scatter(dfg.lon, dfg.lat, 5, c='r', marker='+')
	rect = mpl.patches.Rectangle(
			(dfg.loc["lonb_MOD_min"],dfg.loc["latb_MOD_min"]),
			dfg.loc["lonb_MOD_max"]-dfg.loc["lonb_MOD_min"],
			dfg.loc["lonb_MOD_max"]-dfg.loc["lonb_MOD_min"],linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)
	ax.set_title(None)#"%s %s" % (info.satellite, info.date.split(" ")[0]))
	plt.axis('scaled')
	
	fnout = "%sBAimages/HANSEN_BAFL_%s_.png" % (path, site) 
	plt.savefig(fnout)
	if plot:
		plt.show()
	ax.clear()
	plt.close()

# ==============================================================================
# ============================= General functions ==============================
# ==============================================================================

def SiteInfo(site, dpath, spath):
	"""
	Function takes a site name and retunrs the infomation about the site
	args:
		site:		str
			the name of the site being tested.  

	"""
	# ========== Scale Factor ==========
	path = spath + "%s/"	% site
	cf.pymkdir(path+"BAimages/")
	return path

def datasets(dpath):
	# ========== set the filnames ==========
	data= OrderedDict()
	data["HansenGFL"] = ({
		"fname":dpath + "HANSEN/lossyear/Hansen_GFC-2018-v1.6_lossyear_SIBERIA.nc",
		'var':"lossyear", "gridres":"25m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': 10000, 'latitude': 10000},
		"rename":None, 
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	data["COPERN_BA"] = ({
		'fname':dpath + "COPERN_BA/processed/COPERN_BA_gls_*_SensorGapFix.nc",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":{'time':1,'longitude': 1000, 'latitude': 10000}, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	data["MODIS"] = ({
		"fname":dpath + "MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc",
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'longitude': 1000, 'latitude': 10000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
		})
	data["esacci"] = ({
		"fname":dpath + "esacci/processed/esacci_FireCCI_*_burntarea.nc",
		'var':"BA", "gridres":"250m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': 1000, 'latitude': 10000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	return data

def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-CSHARFM':
		# LAPTOP
		spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/e"
		ipdb.set_trace()
	elif sysname == "owner":
		spath = "/mnt/c/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/mnt/d/Data51/BurntArea/"
	elif sysname == "ubuntu":
		# Work PC
		dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/"
		spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	elif sysname == 'arden-H97N-WIFI':
		spath = "/mnt/FCBE3028BE2FD9C2/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/media/arden/Harbinger/Data51/BurntArea/"
	else:
		ipdb.set_trace()
	return spath, dpath

# ==============================================================================

if __name__ == '__main__':
	main()