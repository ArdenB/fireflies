"""
Thi script will compate variables to FRI to recaalculate the results
"""
#==============================================================================

__title__ = "FRI vs variables"
__author__ = "Arden Burrell"
__version__ = "v1.0(21.08.2019)"
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
import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
from netCDF4 import Dataset, num2date, date2num 
from scipy import stats
# import rasterio
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats
import glob
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import regionmask as rm
# import itertools
# Import debugging packages 
import ipdb
# from rasterio.warp import transform
from shapely.geometry import Polygon
# import geopandas as gpd
# from rasterio import features
# from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================

def main():
	# ========== Setup the paths ==========
	dpath, chunksize = syspath()

	# ========== Load the data ==========

#==============================================================================
def Content_Compare(dpath):
	# ========== Open the datasets =========
	# pre = 
	ipdb.set_trace()


#==============================================================================


def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h/Data51"
		dpath = "/mnt/d/Data51"
		chunksize = 50
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51"
		dpath = "/media/ubuntu/Harbinger/Data51"
		chunksize = 50
		
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	elif 'ccrc.unsw.edu.au' in sysname:
		dpath  = "/srv/ccrc/data51/z3466821"
		clpath = "/srv/ccrc/data51/z3466821/Input_data/TerraClimate"
		chunksize = 20
	else:
		ipdb.set_trace()
	return dpath, clpath, chunksize	

#==============================================================================

if __name__ == '__main__':
	main()