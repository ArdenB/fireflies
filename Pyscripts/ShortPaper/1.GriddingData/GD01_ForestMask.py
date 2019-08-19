
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Boreal Forest Mask"
__author__ = "Arden Burrell"
__version__ = "v1.0(19.08.2019)"
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
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import regionmask as rm
# Import debugging packages 
import ipdb
# from rasterio.warp import transform
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio import features
from affine import Affine
# +++++ Import my packages +++++
# import MyModules.CoreFunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf


def main():
	# ========== open the forest cover file ==========
	LonM = 110
	LatM =  60
	

	fcda = Forest2000(LonM, LatM)

	bada = Dataset()

	# ========== calculate the bounding box of the pixel ==========
	lonres = bada.attrs["res"][0]/2
	latres = bada.attrs["res"][1]/2

	# ========== Subset out a region of interest ==========
	bar =  bada.sel(dict(latitude=slice(LatM - latres, LatM-10+latres), longitude=slice(LonM+lonres, LonM+10-latres)))
	# far =  fcda.sel(dict(latitude=slice(LatM - latres, LatM-10+latres), longitude=slice(LonM+lonres, LonM+10-latres)))
	meansmth = fcda.rolling({"longitude":9}, center = True).mean().rolling({"latitude":9}, center = True).mean() 
	reind = meansmth.reindex_like(bar, method="nearest")    
	sumnof = (fcda==0).rolling({"longitude":9}, center = True).sum().rolling({"latitude":9}, center = True).sum() 
	reindS = sumnof.reindex_like(bar, method="nearest")
	ipdb.set_trace()

	# setup a test pixel


	# t0 = pd.Timestamp.now()
	# for ilon in range(0, bar.longitude.shape[0]):
	# 	BAlon = bar.isel(dict(longitude=ilon))
	# 	fcLon = FC = fcda.sel(dict(
	# 		longitude=slice(BAlon.longitude.values-lonres, BAlon.longitude.values+lonres))).persist()
	# 	array = lonforest(BAlon, fcLon, lonres, latres)
	# 	break
	# tdelta = pd.Timestamp.now() - t0
	# print(tdelta)
	# print(tdelta* bar.latitude.shape[0])
	# ipdb.set_trace()

#==============================================================================

# @jit
def Isforest(tp, FC, maxNF = 0.40, minCC = 0.3):

	
	# ===== Mean canapy cover ==========
	fcM = FC.mean().compute()
	ipdb.set_trace()
	sys.exit()
	# ===== Fraction of non forest ==========
	fcB =  (FC == 0).sum().compute()/np.prod(FC.shape).astype(float)
	if (fcM >= minCC) and (fcB <= maxNF):
		return 1
	else:
		return 0 	

def lonforest(BAlon, fcLon, lonres, latres):
	"""
	Function to loop over each row 
	"""
	array = np.zeros(BAlon.latitude.shape[0])
	for ilat in range(0, BAlon.latitude.shape[0]):
		tp = BAlon.isel(latitude=ilat)
		FC = 	fcLon.sel(dict(	latitude=slice(tp.latitude.values+latres, tp.latitude.values-latres)))
		array[ilat] = Isforest(tp, FC, maxNF = 0.40, minCC = 0.3)
	return array
	
#==============================================================================
def Forest2000(LonM, LatM):
	"""
	Function takes a lon and a lat and them opens the appropiate 2000 forest 
	cover data
	args:
		LonM: int
			must be divisible by 10
		LatM: int
			must be divisible by 10
	returns:
		ds: xarray dataset
			processed xr dataset 
	"""
	if LonM >= 0:
		lon = "%03dE" % LonM
	else:
		lon = "%03dW" % abs(LonM)
	if LatM >= 0:
		lat = "%02dN" % LatM
	else:
		lon = "%02dS" % abs(LatM)
	# ========== Create the path ==========
	path = "./data/Forestloss/2000Forestcover/Hansen_GFC-2018-v1.6_treecover2000_%s_%s.tif" % (lat, lon)


	da = xr.open_rasterio(path)
	da = da.rename({"x":"longitude", "y":"latitude"}) 
	# da = da.chunk({'latitude': 4444})   
	da = da.chunk({'latitude': 1000, "longitude":1000})   

	return da


def Dataset():

	fname = "./data/BurntArea/20010101-ESACCI-L3S_FIRE-BA-MODIS-AREA_4-fv5.1-JD.tif"
	da    = xr.open_rasterio(fname)
	da    = da.rename({"x":"longitude", "y":"latitude"})
	da = da.chunk({'latitude': 4453})    

	return da

#==============================================================================
if __name__ == '__main__':
	main()