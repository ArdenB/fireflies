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
	# ========== Set the site up ==========
	# site   = "G10T1-50" #
	# site   = "TestBurn
	site   = "G5T1-50"
	region = "SIBERIA"

	for site in ["G5T1-50", "TestBurn", "G10T1-50"]:
		print(site)
		# ========== Get infomation about that site ==========
		fnames, SF, dft, dfg = SiteInfo(site)
		path = "./results/movies/%s/"	% site
		cf.pymkdir(path)

		# ========== Make the hansen forest maps ==========
		esacci(path, fnames, SF, dft, dfg, region, site)
		Hansen(path, fnames, SF, dft, dfg, region, site)
		copern(path, fnames, SF, dft, dfg, region, site)

# ==============================================================================
# ====================== BA product functions functions ========================
# ==============================================================================

def esacci(path, fnames, SF, dft, dfg, region, site):
	"""Function  to build the cci maps"""
	# ========== Setup the path ==========
	ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/"
	var   = "BA"
	for yr in range(2001, 2019):
		# ========== make the file name ==========
		ANfn = ppath + "esacci_FireCCI_%d_burntarea.nc" % yr

		# ========== check if the file exists  ==========
		if not os.path.isfile(ANfn):
			print("Unable to locate processed esacci data for ", yr)
			continue

		# ========== open the file  ==========
		ds = xr.open_dataset(ANfn, chunks={'latitude': 1000}).sel(dict(
			latitude =slice(dfg.latr_max[0], dfg.latr_min[0]), 
			longitude=slice(dfg.lonr_min[0], dfg.lonr_max[0]))).compute()

		# ========== build the plot  ==========
		fig, ax = plt.subplots(1, figsize=(11,10))

		ds[var].isel(time=0).plot.imshow(
			ax=ax,
			vmin=0, 
			vmax=1, 
			)
			# cmap=cmap, 
			# cbar_kwargs={'ticks':ticks} 

		ax.scatter(dfg.lon[0], dfg.lat[0], 5, c='r', marker='+')
		rect = mpl.patches.Rectangle(
			(dfg.lonb_min[0],dfg.latb_min[0]),
			dfg.lonb_max[0]-dfg.lonb_min[0],
			dfg.lonb_max[0]-dfg.lonb_min[0],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		# ax.set_title(None)#"%s %s" % (info.satellite, info.date.split(" ")[0]))
		plt.axis('scaled')
		
		fnout = "%sesacci_%s_%s_%d.png" % (path, site, var, yr) 
		plt.savefig(fnout)
		plt.show()
		ax.clear()



def copern(path, fnames, SF, dft, dfg, region, site):
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
	

	files = []
	# ========== get the file neames ==========
	fnames = glob.glob("/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/M0044633/*.nc")
	var = "BA_DEKAD"
	# var = "FDOB_SEASON"
	for fn in fnames:
		ds_yin = xr.open_dataset(fn).rename({"lon":"longitude", "lat":"latitude"}).sel(dict(
			latitude =slice(dfg.latr_max[0], dfg.latr_min[0]), 
			longitude=slice(dfg.lonr_min[0], dfg.lonr_max[0]))).drop(["CP_DEKAD", "crs", "FDOB_DEKAD","FDOB_SEASON"]).compute()

		date = pd.to_datetime(ds_yin.attrs["time_coverage_end"][:10]).to_datetime64()
		try:
			ds_yin = ds_yin.expand_dims("time")
			ds_yin = ds_yin.assign_coords(time=[date])
		
		except:pass
		files.append(ds_yin)
	
	# ========== convert to smart xarray dataset ==========
	ds_in  = xr.concat(files, dim="time")

	ds_sum = ds_in.groupby('time.year').sum("time")
	

	for year in ds_sum.year.values:
		# ========== Set up the plot ==========
		fig, ax = plt.subplots(1, figsize=(11,10))

		ds_sum[var].sel(year=year).plot.imshow(
			ax=ax,
			vmin=0, 
			vmax=1, 
			)
			# cmap=cmap, 
			# cbar_kwargs={'ticks':ticks} 

		ax.scatter(dfg.lon[0], dfg.lat[0], 5, c='r', marker='+')
		rect = mpl.patches.Rectangle(
			(dfg.lonb_min[0],dfg.latb_min[0]),
			dfg.lonb_max[0]-dfg.lonb_min[0],
			dfg.lonb_max[0]-dfg.lonb_min[0],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		# ax.set_title(None)#"%s %s" % (info.satellite, info.date.split(" ")[0]))
		plt.axis('scaled')
		
		fnout = "%sPROBAV_%s_%s_%d.png" % (path, site, var, year) 
		plt.savefig(fnout)
		plt.show()
		ax.clear()
		# ipdb.set_trace()

	# ipdb.set_trace()
	
def Hansen(path, fnames, SF, dft, dfg, region, site):
	# ========== Set up the filename and global attributes =========
	ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN"
	pptex = ({"treecover2000":"FC2000", "lossyear":"lossyear", "datamask":"mask"})
	fpath        = "%s/FRI/" %  ppath
	cf.pymkdir(fpath)

	# ========== Setup the paths ==========
	def _Hansenfile(ppath, pptex, ft, region):
		dpath  = "%s/%s/" % (ppath, pptex[ft])
		datafn = "%sHansen_GFC-2018-v1.6_%s_%s.nc" % (dpath, ft, region)
		# fnout  = "%sHansen_GFC-2018-v1.6_forestmask_%s.nc" % (dpath, region)
		return xr.open_dataset(datafn, chunks={'latitude': 100})

	# ========== get the datatsets ==========
	ds_tc = _Hansenfile(ppath, pptex, "treecover2000", region)
	ds_ly = _Hansenfile(ppath, pptex, "lossyear", region)
	ds_dm = _Hansenfile(ppath, pptex, "datamask", region)

	# ========== subset out the relevant values ==========
	site_ly = ds_ly.sel(dict(
		latitude =slice(dfg.latr_max[0], dfg.latr_min[0]), 
		longitude=slice(dfg.lonr_min[0], dfg.lonr_max[0]))).compute() + 2000

	site_dm = ds_dm.sel(dict(
		latitude =slice(dfg.latr_max[0], dfg.latr_min[0]), 
		longitude=slice(dfg.lonr_min[0], dfg.lonr_max[0]))).compute()

	site_tc = ds_tc.sel(dict(
		latitude =slice(dfg.latr_max[0], dfg.latr_min[0]), 
		longitude=slice(dfg.lonr_min[0], dfg.lonr_max[0]))).compute()

	# ========== fiz the values ==========
	site_ly = site_ly.where(site_dm["datamask"].values == 1)
	site_ly = site_ly.where(site_ly >= 2001, 0)

	# ========== Create the plot ==========
	cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_18.mpl_colors)
	cmap.set_under('w')
	cmap.set_bad('dimgrey',1.)
	tks, counts = np.unique(site_ly["lossyear"].values, return_counts=True)
	ticks = tks[np.logical_and((counts > 100), (tks>0))]
	
	# ========== Set up the plot ==========
	fig, ax = plt.subplots(1, figsize=(11,10))
	ax.clear()
	site_ly["lossyear"].rename(None).isel(time=0).plot.imshow(
		ax=ax,
		vmin=2000.5, 
		vmax=2018.5, 
		cmap=cmap, 
		cbar_kwargs={'ticks':ticks} )

	ax.scatter(dfg.lon[0], dfg.lat[0], 5, c='r', marker='+')
	rect = mpl.patches.Rectangle(
		(dfg.lonb_min[0],dfg.latb_min[0]),
		dfg.lonb_max[0]-dfg.lonb_min[0],
		dfg.lonb_max[0]-dfg.lonb_min[0],linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)
	ax.set_title(None)#"%s %s" % (info.satellite, info.date.split(" ")[0]))
	plt.axis('scaled')
	
	fnout = "%sHANSEN_BAFL_%s_.png" % (path, site) 
	plt.savefig(fnout)
	plt.show()


# ==============================================================================
# ============================= General functions ==============================
# ==============================================================================

def SiteInfo(site):
	"""
	Function takes a site name and retunrs the infomation about the site
	args:
		site:		str
			the name of the site being tested.  

	"""
	# Scale Factor
	SF  = 0.0001 

	if site == "TestBurn":
			fnames = sorted(glob.glob("/home/ubuntu/Downloads/TestSite/*.tif"))
	else:
		fnames = sorted(glob.glob("/home/ubuntu/Downloads/FIREFLIES_geotifs/*.tif"))
		# fnames = sorted(glob.glob("/mnt/c/Users/user/Google Drive/FIREFLIES_geotifs/*.tif"))


	# ========== load the additional indomation ==========
	try:
		dft = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_RGB_timeinfo.csv" % site, index_col=0, parse_dates=True)
		dfg = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_RGB_gridinfo.csv" % site, index_col=0, parse_dates=True)      
		
	except:
		dft = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_NRGB_timeinfo.csv" % site, index_col=0, parse_dates=True)
		dfg = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_NRGB_gridinfo.csv" % site, index_col=0, parse_dates=True)     
	return fnames, SF, dft, dfg

# ==============================================================================

if __name__ == '__main__':
	main()