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
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import xgboost as xgb
from sklearn import preprocessing
import statsmodels.api as sm

import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	dpath, chunksize	= syspath()
	# ========== Load the datasets ==========
	dsn  = "GFED"
	tcfs = ""
	mwb  = 1
	region = "SIBERIA"
	box = [-10.0, 180, 40, 70]

	# ========== Build the dataset ==========
	df = dfloader(dsn, box, mwb, dpath, tcfs)

	# ====================================================
	# ========== Perform soem ML on the dataset ==========
	# ====================================================
	# ========== split the data	========== 
	va = "FRI"
	X  = df.drop([ "AnBF", "FRI"], axis=1)
	y  = df[va]

	Xs = sm.add_constant(X)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2)

	model = sm.OLS(y_train, sm.add_constant(X_train))
	olsre = model.fit()
	yv2   = olsre.predict(sm.add_constant(X_test))
	print('r squared score:',         sklMet.r2_score(y_test, yv2))

	# ========== Calculate the regression ==========
	# breakpoint()
	regressor = xgb.XGBRegressor(objective ='reg:squarederror', tree_method='hist', 
		colsample_bytree = 0.3, learning_rate = 0.1,
		max_depth = 10, n_estimators =2000,
	    num_parallel_tree=10)

	eval_set = [(X_test.values, y_test.values.ravel())]
	regressor.fit(X_train.values, y_train.values.ravel(), early_stopping_rounds=15, verbose=True, eval_set=eval_set)
	# breakpoint()


	# ========== Testing out of prediction ==========
	print("starting regression prediction at:", pd.Timestamp.now())
	y_pred = regressor.predict(X_test.values)

	print('r squared score:',         sklMet.r2_score(y_test, y_pred))

	result = permutation_importance(regressor, X_test.values, y_test.values.ravel(), n_repeats=5)
	# ========== make a list of names ==========
	clnames = X_train.columns.values
	# ========== Convert Feature importance to a dictionary ==========
	FI = OrderedDict()

	for fname, f_imp in zip(clnames, result.importances_mean): 
		FI[fname] = f_imp

	dfpi = pd.DataFrame({"PermutationImportance":pd.Series(FI), "FeatureImportance":regressor.feature_importances_})
	print(dfpi)
	breakpoint()







#==============================================================================
def dfloader(dsn, box, mwb, dpath, tcfs,
	stdt=pd.Timestamp("1980-01-01"), fndt=pd.Timestamp("2018-12-31")):
	"""
	Function to load and preprocess into a single dataframe
	args:
		dsn:		str
			The name of the dataset
		box:		list of len 4
			spatial coords of bounding box
		mwb:		int
			moving window box size
		dpath:		str
			path to the data
		stdt:		pd time
			start date
		fndt:		pd time
			end date
	returns Dataframe of valud pixels
	"""
	# ##########################################
	# ========== Read in the FRI data ==========
	# ##########################################
	ppath = dpath + f"/BurntArea/{dsn}/FRI/"
	fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsn, tcfs, mwb)

	ds_bf = xr.open_dataset(ppath+fname)
	# /// Build some indexers \\\
	latin = np.arange(
		np.floor(ds_bf.latitude.values[0]), 
		np.floor(ds_bf.latitude.values[-1]-1), -1)
	lonin = np.arange(
		np.ceil(ds_bf.longitude.values[0]), 
		np.ceil(ds_bf.longitude.values[-1]+1))
	# ========== Pull out only a subset ove the data to avoid spatial autocorrelation ==========
	ds_sub = ds_bf.reindex({"latitude":latin, "longitude":lonin}, method = "nearest")
	
	# ========== Convert to dataframe, drop time and reset index ==========
	df_sub =  ds_sub.to_dataframe()
	df     = df_sub.reset_index().drop("time", axis=1).set_index(["latitude","longitude"]).copy()
	
	# =================================
	# ========== Climate data =========
	# =================================

	cpath  = "/mnt/e/Data51/Climate/TerraClimate/"
	for var in ["ppt", "tmean"]:
		# ========== Read in the climate data ==========
		ds_cli = xr.open_mfdataset(
			cpath +f"TerraClimate_{var}_*.nc", 
			drop_variables=["station_influence", "crs"]).rename(
			{"lat":"latitude","lon":"longitude"})

		# ========== subset the data to the box ==========
		ds_cli = ds_cli.sel(dict(
			time=slice(stdt, fndt),
			latitude=slice(box[3], box[2]), 
			longitude=slice(box[0], box[1])))

		pix =  abs(np.unique(np.diff(ds_cli.latitude.values))[0]) 
		SF  = np.round(mwb /pix).astype(int)
		# ========== Group and roll the data ==========
		ds_sea = ds_cli.groupby("time.season").mean()
		print(f"Loading {dsn} {var} data into ram at", pd.Timestamp.now())
		with ProgressBar():
			dsan_lons = ds_sea.rolling(
				{"longitude":SF}, center = True, min_periods=1).mean() 
			ds_out = dsan_lons.rolling(
				{"latitude":SF}, center = True, min_periods=1).mean().compute()
		# ========== Convert to dataframe and collapse index ==========
		ds_psu = ds_out.reindex({"latitude":latin, "longitude":lonin}, method = "nearest")
		df_psu =  ds_psu.to_dataframe().unstack()
		df_psu.columns = [''.join(col).strip() for col in df_psu.columns.values]

		df = df.merge(df_psu, left_index=True, right_index=True)
	
	# ======================================================
	# ========== Read in the forest fraction data ==========
	# ======================================================
	ds_mask = xr.open_dataset(f"./data/masks/broad/Hansen_GFC-2018-v1.6_SIBERIA_ProcessedTo{dsn}.nc")
	print(f"Loading {dsn} treecover data into ram at", pd.Timestamp.now())
	with ProgressBar():
		dsml = ds_mask.rolling(
			{"longitude":SF}, center = True, min_periods=1).mean() 
		ds_msk = dsml.rolling(
			{"latitude":SF}, center = True, min_periods=1).mean().compute()
		ds_msk["datamask"] = (ds_msk["datamask"] >.3).astype("float32")

	
	ds_msu = ds_msk.reindex({"latitude":latin, "longitude":lonin}, method = "nearest")
	df_msu = ds_msu.to_dataframe().reset_index().drop("time", axis=1).set_index(["latitude","longitude"])
	df = df.merge(df_msu, left_index=True, right_index=True)
	# ======================================================
	# ========== Do the final round of processing ==========
	# ======================================================
	df["AnBF"][df.AnBF <= 0.0001] = np.NaN

	df.dropna(inplace=True)
	return df



#==============================================================================
def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h/Data51"
		dpath = "/mnt/d/Data51"
		chunksize = 20
		# chunksize = 5000
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51"
		dpath = "/media/ubuntu/Harbinger/Data51"
		chunksize = 50
		
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	elif 'ccrc.unsw.edu.au' in sysname:
		dpath = "/srv/ccrc/data51/z3466821"
		chunksize = 20
		# chunksize = 5000
	elif sysname in ['burrell-pre5820', 'LAPTOP-8C4IGM68']:
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51"
		dpath = "./data"
		chunksize = 300
	elif sysname == 'arden-Precision-5820-Tower-X-Series':
		# WHRC linux distro
		# dpath= "/media/arden/Harbingerq/Data51"
		dpath = "./data"
		chunksize = 300
	else:
		ipdb.set_trace()
	return dpath, chunksize	
#==============================================================================
if __name__ == '__main__':
	main()