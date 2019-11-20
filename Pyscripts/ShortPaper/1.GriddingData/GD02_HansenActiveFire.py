
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Hansen Active fire"
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
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================
def main():

	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-CSHARFM':
		# LAPTOP
		spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"

	elif sysname == "owner":
		spath = "/mnt/d/Data51/"
	elif sysname == "ubuntu":
		# Work PC
		spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"


	# path = spath + "BurntArea/MODIS_ActiveFire/DL_FIRE_M6_85602/fire_archive_M6_85602.shp"
	path = spath + "BurntArea/MODIS_ActiveFire/DL_FIRE_M6_87877/fire_archive_M6_87877.shp"

	# ========== Load the data ==========
	actfire = gpd.read_file(path)
	
	# ========== Add a new column ==========
	years = pd.Series([pd.Timestamp(date).year for date in actfire.ACQ_DATE.values])
	actfire["year"] = years

	# ========== Subset the data ==========
	# actfire = actfire[['ACQ_DATE', 'CONFIDENCE', 'geometry']]
	actfire = actfire[['year', 'geometry']]


	# ========== Convert to an equal area projection ==========
	print("starting equal area reprojection at:", pd.Timestamp.now())
	actfire = actfire.to_crs({'init': 'epsg:3174'})

	# ========== Add a 4km buffer ==========
	actfire["geometry"] = actfire.geometry.buffer(4000)

	# ========== Convert back to projection ==========
	print("starting latlon reprojection at:", pd.Timestamp.now())
	actfire = actfire.to_crs({'init': 'epsg:4326'})

	# ========== Disolve by year ==========
	print("starting dissolve at:", pd.Timestamp.now())
	actfire = actfire.dissolve(by='year')

	# ========== Save the results ==========



	

	ipdb.set_trace()
#==============================================================================

if __name__ == '__main__':
	main()