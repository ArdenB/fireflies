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
	site   = "TestBurn"
	region = "SIBERIA"

	# ========== Get infomation about that site ==========
	fnames, SF, dft, dfg = SiteInfo(site)

	# ========== Make the hansen forest maps ==========
	Hansen(fnames, SF, dft, dfg, region)

# ==============================================================================
# ====================== BA product functions functions ========================
# ==============================================================================

def Hansen(fnames, SF, dft, dfg, region):
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
	fig, ax = plt.subplots(1)
	ax.clear()
	site_ly["lossyear"].isel(time=0).plot.imshow(
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

	ipdb.set_trace()


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
	dft = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_RGB_timeinfo.csv" % site, index_col=0, parse_dates=True)
	dfg = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_%s_RGB_gridinfo.csv" % site, index_col=0, parse_dates=True)      
	return fnames, SF, dft, dfg

# ==============================================================================

if __name__ == '__main__':
	main()