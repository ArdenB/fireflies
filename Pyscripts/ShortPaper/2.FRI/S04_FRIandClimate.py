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
import statsmodels.api as sm
import statsmodels.formula.api as smf


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
	dpath, clpath, chunksize = syspath()
	data = datasets(dpath, 100)

	# ========== Load the data ==========
	for dsn in data:
		Content_Compare(dpath, clpath, dsn, data)
		ipdb.set_trace()

#==============================================================================
def Content_Compare(dpath, clpath, dsn, data):
	# ========== Open the datasets =========
	pre = xr.open_dataset(
		clpath+"/TerraClimate_SIBERIA_ppt_1958to2018.nc", chunks={"latitude": 100, "longitude": 1000})
	tas = xr.open_dataset(
		clpath+"/TerraClimate_SIBERIA_tmean_1958to2018.nc", chunks={"latitude": 100, "longitude": 1000})
	
	fri = xr.open_dataset(dpath+"/BurntArea/%s/FRI/%s_annual_burns_MW_1degreeBox_REMAPBIL.nc" %(dsn, dsn))
	# pull out the areas whe the FRI is way to high
	# fri = fri.where(fri.FRI < 500)

	BFm = xr.open_dataset(dpath+"/masks/broad/Hansen_GFC-2018-v1.6_SIBERIA_ProcessedToTerraClimate.nc")
	BFm = BFm.where(BFm.datamask == 1)
	

	# ========== Group the data =========
	seasons = ["JJA", "Annual", "DJF", "MAM",  "SON"]
	for per in seasons:
		if seasons == "Annual":
			# ========== time subset the data the datasets =========
			pre_sub = pre.sel(dict(time=slice(pd.to_datetime("%d-01-01" % data[dsn]["start"]), None)))
			tas_sub = tas.sel(dict(time=slice(pd.to_datetime("%d-01-01" % data[dsn]["start"]), None)))
			# ========== Build the precipitation =========
			# Resample to annual then get the annual mean
			pre_mean = pre_sub.resample(time="1Y").sum().mean(dim='time')
			pre_mean = pre_mean.where(pre_mean > 0)

			# ========== Build the Temperature =========
			tas_mean = tas_sub.mean(dim='time')
		else:
			# ========== time subset the data the datasets =========
			pre_sub = pre.sel(dict(time=slice(pd.to_datetime("%d-12-01" % (data[dsn]["start"]-1)), None)))
			tas_sub = tas.sel(dict(time=slice(pd.to_datetime("%d-12-01" % (data[dsn]["start"]-1)), None)))

			pre_mean = pre_sub.resample(time="QS-DEC").sum()
			pre_mean = pre_sub.groupby('time.season').mean()
			pre_mean = pre_mean.sel(season=per).drop("season")
			
			tas_mean = tas_sub.resample(time="QS-DEC").mean()
			tas_mean = tas_mean.groupby('time.season').mean()
			tas_mean = tas_mean.sel(season=per).drop("season")

			# man_annual = ds[var].where(ds[var]['time.season'] == season).groupby('time.year')

		print("Starting ppt mean calculation at" , pd.Timestamp.now())
		with ProgressBar():
			pre_mean = pre_mean.compute()
		
		pre_mean = pre_mean.expand_dims("time")
		pre_mean *= BFm.datamask.values
		pre_mean["time"] = fri.time


		# pre_mean = pre_mean.where(pre_mean > 0)
		
		print("Starting tas mean calculation at" , pd.Timestamp.now())
		with ProgressBar():
			tas_mean = tas_mean.compute()

		tas_mean = tas_mean.expand_dims("time")
		tas_mean["time"] = fri.time

		# ========== subset the valyes ==========
		dfinput = OrderedDict()
		dfinput["FRI"]           = fri.FRI.values.reshape(-1)
		dfinput["AnBF"]          = fri.AnBF.values.reshape(-1)
		dfinput["temperature"]   = tas_mean.tmean.values.reshape(-1)
		dfinput["precipitation"] = pre_mean.ppt.values.reshape(-1)
		dfinput["treecover"]     = BFm.treecover2000.values.reshape(-1)

		df = pd.DataFrame(dfinput)
		df.dropna(axis=0, inplace=True)
		# df.plot.scatter(y="FRI", x="precipitation")

		results = smf.ols('FRI ~ temperature * precipitation', data=df).fit()
		print(results.summary())
		# results = smf.ols('FRI ~ treecover + temperature * precipitation', data=df).fit()
		# print(results.summary())
		ipdb.set_trace()
		df = df[df.FRI<500]

		sns.jointplot(x="temperature", y="FRI", data=df,  kind="hex")
		plt.show()
		ipdb.set_trace()



#==============================================================================

def datasets(dpath, chunksize):
	# ========== set the filnames ==========
	data= OrderedDict()
	data["MODIS"] = ({
		"fname":dpath+"/BurntArea/MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc",
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1,'longitude': chunksize, 'latitude': chunksize},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
		})
	data["COPERN_BA"] = ({
		'fname':dpath+"/BurntArea/COPERN_BA/processed/COPERN_BA_gls_*_SensorGapFix.nc",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":{'time':1, 'longitude': chunksize, 'latitude': chunksize}, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})

	data["esacci"] = ({
		"fname":dpath+"/BurntArea/esacci/processed/esacci_FireCCI_*_burntarea.nc",
		'var':"BA", "gridres":"250m", "region":"Asia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': chunksize, 'latitude': chunksize},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	return data
def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h/Data51"
		clpath = "/mnt/d/Data51/climate/TerraClimate"
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