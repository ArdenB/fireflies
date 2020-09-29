"""
Script goal, 

To Calculate the relationship between vars that determin FRI

"""
#==============================================================================

__title__ = "FRI Drivers"
__author__ = "Arden Burrell"
__version__ = "v1.0(23.09.2020)"
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
# import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import shutil
import time
from dask.diagnostics import ProgressBar
import dask.array as dska
from numba import  jit

from collections import OrderedDict
# from scipy import stats
# from numba import jit


# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl

import palettable 

# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# ========== Import ml packages ==========
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
from sklearn.utils import shuffle
# from scipy.stats import spearmanr
from scipy import stats
from scipy.cluster import hierarchy
import xgboost as xgb
from sklearn import preprocessing
# import statsmodels.api as sm
from sklearn import linear_model
# from sklearn.decomposition import PCA
import statsmodels.stats.multitest as smsM



import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

# ==============================================================================

def main():
	# =================================
	# ========== Input Params =========
	# =================================
	box   = [-10.0, 180, 40, 70]
	stdt  = pd.Timestamp("1985-01-01")
	fndt  = pd.Timestamp("2015-12-31")

	tstep = "annual"
	force = False
	cpath = "/mnt/e/Data51/Climate/TerraClimate/"
	
	# ========= Loop over the configerations ==========
	for var, fnt in zip(["ppt", "tmean"], [bn.nansum, bn.nanmean]):
		# if var == "ppt":
		# 	warn.warn("REMOVE ASAP")
		# 	continue
		for tstep in ["DJF","MAM","JJA","SON", "annual"]:
			fn = ClimateChangeCal(box, stdt, fndt, tstep, cpath, force, var, fnt)

# ==============================================================================
def ClimateChangeCal(box, stdt, fndt, tstep, cpath, force, var, fnt):
	"""
	function to calculate the annual climate change component 
	"""
	# ========== Read in the climate data ==========
	fnout = cpath + f"TerraClim_{var}_{tstep}trend_{stdt.year}to{fndt.year}.nc"
	if os.path.isfile(fnout) and not force:
		print(f"A file alread exists for {tstep} {var}")
		return fnout
	else:
		print(f"Starting {tstep} {var} trend calculation at {pd.Timestamp.now()}")

	ds_cli = xr.open_mfdataset(
		cpath +f"TerraClimate_{var}_*.nc", 
		drop_variables=["station_influence", "crs"]).rename(
		{"lat":"latitude","lon":"longitude"})
			# ========== subset the data to the box ==========
	ds_cli = ds_cli.sel(dict(
		# time=slice(stdt, fndt),
		latitude=slice(box[3], box[2]), 
		longitude=slice(box[0], box[1])))
	# ========== select and reduce the dataset ==========
	with ProgressBar():
		if tstep == "annual":
			ds_re = ds_cli.resample(time="Y").reduce(fnt).compute()
		else:
			ds_re = ds_cli.resample(time="QS-DEC").reduce(fnt).compute()
			# Shift things into the correct year 
			ds_re = ds_re.assign_coords(time=ds_re.time + pd.Timedelta(31, unit="d"))
			ds_re = ds_re.isel(time=(ds_re["time.season"] == tstep))
	
	# ========== Roll the dataset ==========
	ds_smoothed = ds_re.rolling(
		{"time":20}, min_periods=10).mean()
	ds_smoothed = ds_smoothed.where( ~(ds_smoothed == 0))
	ds_set = ds_smoothed.sel(dict(time=slice(stdt, fndt)))#.chunk({"longitude":285})
	
	# ========== Convert to 2d numpy array and apply ==========
	arr = ds_set[var].stack(pos=("latitude", "longitude")).values
	nonna = arr[:, ~bn.anynan(arr, axis=0)]

	# start the loop 
	print (f"Starting {tstep} {var} apply along axis at: {pd.Timestamp.now()}")
	# da_NN 
	t0  = pd.Timestamp.now()
	res = np.apply_along_axis(TheilSen, 0, nonna)
	t1  = pd.Timestamp.now()
	print(t1-t0)

	# ========== convert back to ==========
	# Build a numpy array 
	arr_out =  np.zeros([5, arr.shape[1]])
	arr_out[:] = np.nan
	# ===== Work out multisig =====
	pvalue_adj, _,_,_ =  smsM.multipletests(res[-4, :], method="fdr_bh", alpha=0.10)
	arr_out[:, ~bn.anynan(arr, axis=0)] = np.vstack([res,pvalue_adj.astype(float)])

	# ========== loop over the variables and put them into a dict ==========
	vnames = ["slope", "intercept", "rho", "pval", "FDRsig"]
	OD = OrderedDict()
	for nm, vn in enumerate(vnames):
		OD[vn] = (
			ds_set[var].stack(pos=("latitude", "longitude")).isel(time=-1).drop("time")).copy(data=arr_out[nm, :])
	
	# ========== Convert OD to dataset and fix attrs ==========
	ds     = xr.Dataset(OD).unstack().expand_dims({"time":[fndt]})
	try:
		ds.attrs = GlobalAttributes(ds, var, fnout, tstep, stdt, fndt)
		ds.longitude.attrs = ds_cli.longitude.attrs
		ds.latitude.attrs  = ds_cli.latitude.attrs

		ds_test = tempNCmaker(ds, fnout, vnames, var)
	except Exception as er:
		warn.warn(str(er))
		print("Failed on save, going manual")
		breakpoint()
	return fnout

#==============================================================================

def GlobalAttributes(ds, var, fnout, tstep, stdt, fndt):
	"""
	Creates the global attributes for the netcdf file that is being written
	these attributes come from :
	https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/metadata/DataDiscoveryAttConvention.html
	args
		ds: xarray ds
			Dataset containing the infomation im intepereting
		fnout: str
			filename out 
	returns:
		attributes 	Ordered Dictionary cantaining the attribute infomation
	"""
	# ========== Create the ordered dictionary ==========
	attr = ds.attrs

	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["FileName"]            = fnout
	attr["title"]               = f"{tstep}{var}trend"
	attr["summary"]             = f"Terraclimate {var} trend from {stdt} to {fndt}.  Trend:Theilsen, Sig:Spearnmans RHo, FDR:fdr_bh" 
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s." % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)


	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "Woodwell Climate Research Center"
	attr["date_created"]        = str(pd.Timestamp.now())
	
	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	# attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	# attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	return attr	



def tempNCmaker(ds, fnout, vnames, var, readchunks={'longitude': 1000}):

	""" Function to save out a tempary netcdf """
	

	delayed_obj = ds.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		unlimited_dims = ["time"],
		compute=False)

	print("Starting write of %s data at" % var, pd.Timestamp.now())
	with ProgressBar():
		results = delayed_obj.compute()

	dsout = xr.open_dataset(fnout, chunks=readchunks) 
	return dsout

def TheilSen(array):
	"""
	funtion to be applied to the blocks
	"""
	if bn.anynan(array):
		return np.array([np.nan, np.nan, np.nan,np.nan])
	else:
		slope, intercept, _, _ = stats.mstats.theilslopes(
					array, np.arange(array.size))
		rho, pval = stats.spearmanr(
			array, np.arange(array.size))
		# change = (slope*array.shape[0])
		return np.array([slope, intercept, rho, pval])

def alternate():
    pass
		# with ProgressBar():
	# 	ds_theil = xr.apply_ufunc(
	# 	        TheilSen,
	# 	        ds_set,
	# 	        vectorize=True,
	# 	        input_core_dims=[["time"]],
	# 	        dask="parallelized",
	# 	        output_dtypes=[float],
	# 	        output_core_dims=[['theil']], 
	# 	        output_sizes=({"theil":4})
	# 	    ).compute()
	# breakpoint()	
	        # output_core_dims=[["slope", "intercept", "rho", "pval"]]

#==============================================================================

if __name__ == '__main__':
	main()