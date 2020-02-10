
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Hansen Active fire version 2"
__author__ = "Arden Burrell"
__version__ = "v1.0(20.11.2019)"
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
#==============================================================================
import numpy as np
import pandas as pd
import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
from netCDF4 import Dataset, num2date, date2num 
from scipy import stats
import rasterio
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats

import shutil
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import itertools

# Import debugging packages 
import ipdb

# ========== Import specific packages  ==========
# from rasterio.warp import transform
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio import features
from affine import Affine
# import fiona as fi
# import regionmask as rm
# import matplotlib.path as mplPath
from rasterio import features
from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================

def main():
	# ==========
	force = False
	
	# ========== Get the path ==========
	dpath, chunksize = syspath()

	# ========== Make the files ==========
	fnames = ActiveFireMask(dpath, force)


	# ========== Loop over the datasets ==========
	HansenMasker(fnames, force, dpath, ymin=2001, ymax=2019, )
	ipdb.set_trace()


#==============================================================================

def ActiveFireMask(dpath, force, ymin=2001, ymax=2019):
	def _annualfire(actfire, yr):
		# ========== Convert to an equal area projection ==========
		print("starting equal area reprojection at:	", pd.Timestamp.now())
		actfire = actfire.to_crs({'init': 'epsg:3174'})

		# ========== Add a 4km buffer ==========
		print("starting buffer at:				", pd.Timestamp.now())
		actfire["geometry"] = actfire.geometry.buffer(4000)

		# ========== Convert back to projection ==========
		print("starting latlon reprojection at:   	", pd.Timestamp.now())
		actfire = actfire.to_crs({'init': 'epsg:4326'})

		# ========== Disolve by year ==========
		print("starting dissolve at:			", pd.Timestamp.now())
		actfire = actfire.dissolve(by='year')

		# ========== Save the results ==========
		print("starting data write at:			", pd.Timestamp.now())
		actfire.to_file(outfn)
	
	# ========== Build the results ==========
	path  = dpath + "BurntArea/MODIS_ActiveFire/DL_FIRE_M6_85602/fire_archive_M6_85602.shp"
	


	fnames = [dpath + "BurntArea/MODIS_ActiveFire/AnnualActiveFire%d.shp" % yr for yr in range(ymin, ymax)]

	if not all([os.path.isfile(fn) for fn in fnames]) or force:
		# ========== Load the data ==========
		print("starting active fire data read at:	", pd.Timestamp.now())
		afr = gpd.read_file(path)

		# ========== Add a new column ==========
		years = pd.Series([pd.Timestamp(date).year for date in afr.ACQ_DATE.values])
		afr["year"] = years
	
		# ========== Subset the data ==========
		afr = afr[afr.CONFIDENCE >= 30]
		afr = afr[['year', 'geometry']]

		# ========== Export the reActive fire masks ==========
		for yr, outfn in zip(range(ymin, ymax), fnames):
			if not os.path.isfile(outfn) or force:	
				print("Building new active fire shapefile for ", yr)
				# ========== Pull out the relevant year ==========
				actfire = afr[afr.year == yr].copy()

				# ========== Make the new shapefile ==========
				_annualfire(actfire, yr)

				# ========== Free the memory ==========
				actfire = None


			else:
				print("Shapefile already exists for ", yr)

	# ========== Load the data ==========
	return fnames
#==============================================================================

def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-UA7CT9Q':
		# dpath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h/Data51"
		# clpath = "/mnt/d/Data51/climate/TerraClimate"
		dpath = "/mnt/d/Data51"
		chunksize = 50
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51"
		dpath = "/media/ubuntu/Harbinger/Data51"
		chunksize = 50
		
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	elif 'ccrc.unsw.edu.au' in sysname:
		dpath  = "/srv/ccrc/data51/z3466821"
		# clpath = "/srv/ccrc/data51/z3466821/Input_data/TerraClimate"
		chunksize = 20
	else:
		ipdb.set_trace()
	return dpath, chunksize	

#==============================================================================

if __name__ == '__main__':
	main()