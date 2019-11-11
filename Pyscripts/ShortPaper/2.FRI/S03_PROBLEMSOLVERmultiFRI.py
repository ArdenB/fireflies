"""
Script exists to fix some of the weird data issues i've noticed the FRI results
"""

"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "FRI calculator Issue fixer"
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
import rasterio
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
import geopandas as gpd
from rasterio import features
from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================

def main():
	# ========== Setup the paths ==========
	data = datasets()
	
	# ========== select and analysis scale ==========
	mwbox = [1, 2, 5]#, 1, 10] #in decimal degrees
	force = True
	# force = True
	for dsn in data:
		# ========== Set up the filename and global attributes =========
		ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/%s/FRI/" %  dsn
		cf.pymkdir(ppath)
		
		# ========== Get the dataset =========
		ds, mask = dsloader(data, dsn, ppath, force)
		ipdb.set_trace()

		# ========== Calculate the annual burn frewuency =========
		# force = False
		# ds_ann = ANNcalculator(data, dsn, ds, mask, force, ppath)

		# force = True
		# ========== work out the FRI ==========
		# FRIcal(ds_ann, mask, dsn, force, ppath, mwbox, data)
		
	ipdb.set_trace()


#==============================================================================
def dsloader(data, dsn, ppath, force):
	"""Takes in infomation about datasets and loads a file
	args
		data: Ordered dict
		dsn:	str of the dataset name
	returns:
		ds
	"""
	# ========== check if the name needs to be globbed  ==========
	if "*" in data[dsn]["fname"]:
		# ========== get all the file names ==========
		fnames = glob.glob(data[dsn]["fname"])
		lat = []	#a test to make sure the sizes are correct
		for fn in fnames:
			dsin = xr.open_dataset(fn, chunks=data[dsn]["chunks"])
			lat.append(dsin["BA"].shape[1] )

		# ========== open the dataset ==========
		ds = xr.open_mfdataset(fnames, concat_dim="time", chunks=(data[dsn]["chunks"]))
		
		# ========== Add a simple dataset check ==========
		if not np.unique(lat).shape[0] == 1:
			warn.warn("the datasets have missmatched size, going interactive")
			ipdb.set_trace()
			sys.exit()
	else:
		ds = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
	
	mask = landseamaks(data, dsn, ppath, ds, force )

	return ds, mask

def landseamaks(data, dsn, ppath, ds, force, chunks=None ):

	# ========== create the mask fielname ==========
	masknm = "%s_landseamask.nc" % dsn

	if dsn == "esacci":
		chunks = data[dsn]["chunks"]

	raw_mask = xr.open_dataset(ppath+masknm, chunks=chunks)
	return raw_mask

def datasets():
	# ========== set the filnames ==========
	data= OrderedDict()
	data["COPERN_BA"] = ({
		'fname':"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/COPERN_BA/processed/COPERN_BA_gls_*.nc",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":None, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	# data["MODIS"] = ({
	# 	"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc",
	# 	'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
	# 	"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1,'longitude': 1000, 'latitude': 10000},
	# 	"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
	# 	})
	# data["esacci"] = ({
	# 	"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_FireCCI_*_burntarea.nc",
	# 	'var':"BA", "gridres":"250m", "region":"Asia", "timestep":"Annual", 
	# 	"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': 1000, 'latitude': 1000},
	# 	"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
	# 	# "rename":{"band":"time","x":"longitude", "y":"latitude"}
	# 	})
	# data["GIMMS"] = ({
	# 	"fname":"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc",
	# 	'var':"ndvi", "gridres":"8km", "region":"global", "timestep":"Annual", 
	# 	"start":1982, "end":2017, "rasterio":False, "chunks":{'time': 36},
	# 	"rename":None
	# 	})
	# data["COPERN"] = ({
	# 	'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
	# 	'var':"NDVI", "gridres":"1km", "region":"Global", "timestep":"AnnualMax",
	# 	"start":1999, "end":2018,"rasterio":False, "chunks":{'time':1}, 
	# 	"rename":{"lon":"longitude", "lat":"latitude"}
	# 	})
	return data


#==============================================================================
if __name__ == '__main__':
	main()