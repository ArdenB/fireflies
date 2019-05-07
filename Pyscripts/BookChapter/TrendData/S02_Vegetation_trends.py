"""
Script goal, to produce trends in netcdf files
This script can also be used in P03 if required

"""
#==============================================================================

__title__ = "Global Vegetation Trends"
__author__ = "Arden Burrell"
__version__ = "v1.0(28.03.2019)"
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
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
import glob
from scipy import stats
import statsmodels.stats.multitest as smsM
import myfunctions.PlotFunctions as pf

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import socket
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)
#==============================================================================
def main():
	# ========== Get the relevant datasets ==========
	data = datasets()
	# ========== loop over the datasets ==========
	for dsn in data:
		# ========== open the dataset ==========
		ds = xr.open_dataset(data[dsn]["fname"])#, chunks={"latitude":480})
		if dsn == 'COPERN':
			ds = ds.drop(["crs", "time_bnds"])
		# ========== check if the file uses the correct names ==========
		try:
			nl = ds.latitude.values.shape[0]
		except AttributeError:
			# rename the lats and lons
			ds    = ds.rename({"lat":"latitude", "lon":"longitude"})
			nl = ds.latitude.values.shape[0]
		

		if ".ccrc.unsw.edu.au" in socket.gethostbyaddr(socket.gethostname())[0]:
			Lcnks = 20
			nlats = ds.latitude.values.shape[0]
			nlons = ds.longitude.values.shape[0]
			dsc = ds.chunk({
				"latitude":int(nlats/Lcnks), 
				"longitude":int(nlons/Lcnks)})
		else:
			# ========== subset for smaller ram ==========
			Lcnks = 40
			mapdet = pf.mapclass("boreal")
			dsc = ds.loc[dict(
				longitude=slice(mapdet.bounds[0], mapdet.bounds[1]),
				latitude=slice(mapdet.bounds[2], mapdet.bounds[3]))]
			# ========== Set the number of chunks ==========
			nlats = dsc.latitude.values.shape[0]
			nlons = dsc.longitude.values.shape[0]
			dsc = dsc.chunk({
				"latitude":int(nlats/Lcnks), 
				"longitude":int(nlons/Lcnks)})

		ipdb.set_trace()
		

		with ProgressBar():
		        dsout = nonparmetric_correlation(dsc, 'time').compute()
		ipdb.set_trace()

#==============================================================================
# ============================ xarray nonparmetric ============================
#==============================================================================
# @jit(nogil=True)
def scipyTheilSen(array, retpar):
	"""
	Function for rapid TheilSen slop estimation with time. 
	the regression is done with  an independent variable 
	rangeing from 0 to array.shape to make the intercept 
	the start which simplifies calculation
	
	args:
		array 		np : numpy array of annual max VI over time 
	return
		result 		np : slope, intercept
	"""
	try:
		# if bn.allnan(array):
		# 	return np.array([np.NAN, np.NAN, np.NAN, np.NAN])

		slope, intercept, _, _ = stats.mstats.theilslopes(
			array, np.arange(array.shape[0]))
		rho, pval = stats.spearmanr(
			array, np.arange(array.shape[0]))
		# change = (slope*array.shape[0])
		return np.array([slope, intercept, rho, pval])

	except Exception as e:
		# print(e)
		# warn.warn("unhandeled Error has occured")
		return np.array([np.NAN, np.NAN, np.NAN, np.NAN])
		# return np.NAN


def nonparmetric_correlation(array, retpar, dim='time' ):
    return xr.apply_ufunc(
        scipyTheilSen, array, 
        input_core_dims=[[dim]],
        vectorize=True,
        dask="allowed",#'parallelized',
        output_dtypes=[float, float, float, float],
        output_core_dims=[['slope'], ['intercept'], ['rho'], ['pvalue']]
        )

#==============================================================================
# ========================== Other usefull functions ==========================
#==============================================================================
def datasets():
	"""
	Create the summary of the datasets to be analyised
	"""


	data= OrderedDict()

	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
		'var':"NDVI", "gridres":"COPERN", "region":"Global", "Periods":["AnnualMax"]
		})
	data["GIMMS31v11"] = ({
		'fname':"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc",
		'var':"ndvi", "gridres":"GIMMS", "region":"Global", "Periods":["AnnualMax"]
		})
	data["MODISaqua"] = ({
		'fname': sorted(glob.glob("./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc"))[1:],
		'var':"ndvi", "gridres":"MODIS", "region":"Siberia", "Periods":["All"]
		})
	data["GIMMS31v10"] = ({
		'fname':"./data/veg/GIMMS31g/3.GLOBAL.GIMMS31.1982_2015_AnnualMax.nc",
		'var':"ndvi", "gridres":"GIMMS", "region":"Global", "Periods":["AnnualMax"]
		})
	return data


#==============================================================================
# ========================== Call the main functions ==========================
#==============================================================================

if __name__ == '__main__':
	main()