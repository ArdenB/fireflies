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
from itertools import islice 
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
# import statsmodels.api as sm
from sklearn import linear_model
from sklearn.decomposition import PCA


import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	dpath, cpath, chunksize	= syspath()
	# ========== Load the datasets ==========
	# dsn  = "GFED"
	# dsn  = "esacci"
	dsn  = "MODIS"

	tmpath = "./results/ProjectSentinal/FRImodeling/"
	cf.pymkdir(tmpath)
	cf.pymkdir(tmpath+"tmp/")

	tcfs = ""
	mwb  = 1
	region = "SIBERIA"
	box = [-10.0, 180, 40, 70]
	# va = "FRI"
	va = "AnBF"

	stdt  = pd.Timestamp("1985-01-01")
	fndt  = pd.Timestamp("2015-12-31")
	drop  = ["AnBF", "FRI", "datamask"]#, "treecover2000"]#, "pptDJF", "tmeanDJF"]#"pptDJF", "pptJJA",
	BFmin = 0.0001
	DrpNF = True # False
	sub   = 1 #subsampling interval in deg lat and long
	transform = "QFT" #None 
	sens  =  60
	rammode="complex" #"full"


	# ========== Build the dataset ==========
	df, mask, ds_bf, latin, lonin = dfloader(dsn, box, mwb, dpath, cpath, tcfs, stdt, fndt, va, BFmin, DrpNF, sub)

	# ========== Calculate the ML models ==========
	models = MLmodeling(df, va, drop, BFmin, DrpNF, trans = transform)

	# ========== Calculate the future ==========

	FuturePrediction(df, dsn, models, box, mwb,dpath, cpath, tcfs, stdt, fndt, 
		mask, ds_bf, va, drop, BFmin, DrpNF, latin, lonin, tmpath, fmode="trend", 
		rammode=rammode, sen=sens)

	FuturePrediction(df, dsn, models, box, mwb,dpath, cpath, tcfs, stdt, fndt, 
		mask, ds_bf, va, drop, BFmin, DrpNF, latin, lonin, tmpath, fmode="TCfut", 
		rammode="full")
	breakpoint()

#==============================================================================
def FuturePrediction(df_org, 
	dsn, models, box, mwb, 
	dpath, cpath, tcfs, stdt, fndt, 
	mask, ds_bf, va, drop, BFmin, 
	DrpNF, latin, lonin, tmpath, fmode="TCfut",
	sen=4, rammode = "simple", fut="", splits = 10
	):
	"""
	Function to get predictions of FRI based on future climate.
	args:
		models:		dict
			container of ml models
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
		mask:		xr ds
			ds with mask and treecover
		ds_bf:		xr ds
			ds with observed FRI
		va:			str
			name of the variable being assessed 
		sen:		int
			degrees of warming in the future senario 

	"""
	# ========== specify the climate data path ==========
	
	if rammode == "simple":
		# /// Simple only uses one point per mwb, is way less ram instensive \\\
		# .reindex({"latitude":latin, "longitude":lonin}, method = "nearest")
		df_obs = df_org
		lats   = latin
		lons   = lonin
	else:
		lats = ds_bf.latitude.values
		lons = ds_bf.longitude.values

	dfX    = ds_bf[va].reindex(
		{"latitude":lats, "longitude":lons}, 
		method = "nearest").squeeze("time",drop=True).to_dataframe()
	msk    = mask.reindex(
		{"latitude":lats, "longitude":lons}, 
		method = "nearest").squeeze("time",drop=True).to_dataframe()
	
	# ========== make the observed dataset ==========
	df_obs = _Obsclim(tmpath, dsn, dfX, cpath, mwb, stdt, fndt, msk, ds_bf, va, drop, lats, lons, rammode, splits=splits)

	# breakpoint()
	df_obs.drop(drop, axis=1, errors='ignore', inplace=True)

	# ========== make the future datasets ==========
	print(f"Loading in the future climate data at: {pd.Timestamp.now()}")
	if fmode=="TCfut":
		df_pre   = _futurePre(dsn, cpath, box, mwb, tcfs, stdt, fndt, ds_bf, lats, lons, sen)
		df_tmean = _futureTemp(dsn, cpath, box, mwb, tcfs, stdt, fndt, ds_bf, lats, lons, sen)
	else:
		# ========== Loop over the trends ==========
		df_pre, df_tmean = _ctrend_cal(cpath, stdt, fndt, mwb, df_obs, sen, lats, lons, rammode, splits=splits)

	# ========== Build a dataframe ==========
	df = dfX.merge(df_pre, left_index=True, right_index=True)
	df = df.merge(df_tmean, left_index=True, right_index=True)
	df = df.merge(msk, left_index=True, right_index=True)

	print(f"Loading in the future climate complete at: {pd.Timestamp.now()}")
	# =============================================
	# ========== Estimate the future FRI ==========
	# =============================================
	# ========== Filter the dataset ==========
	if va == "FRI":
		sss = (df.FRI<= 1/BFmin)
	else:
		sss = (df.AnBF>= BFmin)
	# /// maks non forests ]]]
	if DrpNF:
		subs = np.logical_and(sss, df["datamask"]==1)
	else:
		subs = sss
	breakpoint()
	X      = df[subs].copy().drop(drop, axis=1, errors='ignore')
	df_obs = df_obs[subs]
	dfX[va][~subs] = np.NaN

	# ========== loop over the models ==========
	for mod in models['models']:
		# ========== Calculate the future estimates ==========
		regressor = models['models'][mod]
		for Xdf, modi in zip([df_obs, X],["cur", "fut"]):
			print(f"starting {mod} {modi} prediction at: {pd.Timestamp.now()}")
			# breakpoint()
			if not models["transformer"] is None:
				Xdf = models["transformer"].transform(Xdf)
			else:
				Xdf = Xdf.to_numpy()
			y_pred = regressor.predict(Xdf)
			# y_pred[y_pred<0] = 0 #Remove places that make no sense

			# ========== Create a nue column in the table ==========
			dfX[f"{va}_{mod}_{modi}"] = np.NaN
			dfX[f"{va}_{mod}_{modi}"][subs] = y_pred

	# ========== Covert to dataset ==========
	dsX = dfX.to_xarray()
	warn.warn("To DO, Save the file here to stop super slow recalculation process")
	breakpoint()
	# def Plotmaker(dsX)
	# ========== Build a plot ==========
	if va == "FRI":
		cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors
	else:
		cmapHex = palettable.matplotlib.Viridis_11.hex_colors
	cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
	cmap.set_over(cmapHex[-1] )
	cmap.set_bad('dimgrey',1.)
	for var in dfX.columns:
		plt.figure(var)
		if va =="FRI":

			dsX[var].plot(vmin=0, vmax=500, cmap=cmap, cbar_kwargs={"extend":"max"})#1/BFmin)
		else:
			dsX[var].plot(vmin=0, vmax=.1, cmap=cmap, cbar_kwargs={"extend":"max"})
	for exper in ["OLS", "XGBoost"]:
		plt.figure(exper)
		(dsX[f"AnBF_{exper}_fut"] - dsX[f"AnBF_{exper}_cur"]).plot(vmin=-0.05, vmax=.05, cbar_kwargs={"extend":"both"})
	plt.show()


	# ========== Plot the FIR ==========
	for var in dfX.columns:
		cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors
		cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
		cmap.set_over(cmapHex[-1] )
		cmap.set_bad('dimgrey',1.)

		plt.figure(var)
		(1./dsX[var]).plot(vmin=0, vmax=100, cmap=cmap, cbar_kwargs={"extend":"max"})#1/BFmin)

	for exper in ["OLS", "XGBoost"]:
		plt.figure(exper)
		# +++++ pull out the values and mask bad values ==========
		fut  = (1/((dsX[f"AnBF_{exper}_fut"]).where(~(dsX[f"AnBF_{exper}_fut"]<=BFmin), BFmin)))
		cur  = (1/((dsX[f"AnBF_{exper}_cur"]).where(~(dsX[f"AnBF_{exper}_cur"]<=BFmin), BFmin)))
		delt = (fut - cur)
		delt.plot(vmin=-500, vmax=500, cmap = plt.get_cmap('PiYG'), cbar_kwargs={"extend":"both"})
	plt.show()
	breakpoint()
	ipdb.set_trace()


def MLmodeling(df, va, drop, BFmin, DrpNF, trans = "QFT"):
	"""
	take in the dataset and builds dome ML models
	"""
	# ====================================================
	# ========== Perform soem ML on the dataset ==========
	# ====================================================
	# ========== split the data	========== 
	# X  = df.drop(drop, axis=1)
	y  = df[va]
	X_t, X_ts, y_train, y_test = train_test_split(
		df, y, test_size=0.2, random_state=42)
	# ========== subset the dataset ==========
	X_tn  = X_t.drop(drop, axis=1)
	X_tst = X_ts.drop(drop, axis=1)
	# ========== perform some transforms ==========
	if trans == "QFT":	
		transformer = preprocessing.QuantileTransformer(random_state=0)
		if not "datamask" in X_tst.columns:
			X_train = transformer.fit_transform(X_tn)
			X_test  = transformer.transform(X_tst)

		else:
			breakpoint()
	elif trans is None:
		X_train = X_tn.values
		X_test  = X_tst.values
		transformer = None

	# ========== Create linear regression object ==========
	regr = linear_model.LinearRegression(n_jobs=-1)

	# Train the model using the training sets
	regr.fit(X_train, y_train)
	ryval      = regr.predict(X_test)
	resultsOLS = permutation_importance(regr, X_test, y_test, n_repeats=5)
	# breakpoint()

	# ========== XGBoost Regression ==========
	regressor = xgb.XGBRegressor(
		objective ='reg:squarederror', tree_method='hist', 
		colsample_bytree = 0.3, learning_rate = 0.1,
		max_depth = 20, n_estimators =2000,
	    num_parallel_tree=20, n_jobs=-1)

	eval_set = [(X_test, y_test)]
	regressor.fit(
		X_train, y_train, 
		early_stopping_rounds=15, verbose=True, eval_set=eval_set)
	# breakpoint()


	# ========== Testing out of prediction ==========
	print("starting regression prediction at:", pd.Timestamp.now())
	y_pred = regressor.predict(X_test)
	R2_OLS = sklMet.r2_score(y_test, ryval) 
	R2_XGB = sklMet.r2_score(y_test, y_pred)
	print(f'OLS r squared score: {R2_OLS}')
	print(f'XGB r squared score: {R2_XGB}')

	# ========== Check the performance over the masked zone ==========
	if not DrpNF:
		# ===== Subset to only boreal forest =====
		inbf = (X_ts.datamask == 1).values
		print("Performance in boreal forests:")
		print(f'BF OLS r squared score: {sklMet.r2_score(y_test[inbf], ryval[inbf])}')
		print(f'BF XGB r squared score: {sklMet.r2_score(y_test[inbf], y_pred[inbf])}')
		# breakpoint()

	# ========== make a list of names and performance metrics ==========
	resultXGB  = permutation_importance(regressor, X_test, y_test, n_repeats=5)
	featImpXGB = regressor.feature_importances_
	clnames    = X_tst.columns.values
	
	# ========== Convert Feature importance to a dictionary ==========
	FI = OrderedDict()
	for loc, fname in enumerate(clnames): 
		FI[fname] = ({
			"XGBPermImp":resultXGB.importances_mean[loc], 
			"XGBFeatImp":featImpXGB[loc], 
			"OLSPermImp":resultsOLS.importances_mean[loc]})

	dfpi = pd.DataFrame(FI).T
	print(dfpi)

	return ({"models":{"OLS":regr, "XGBoost":regressor}, 
		"transformer":transformer, "Importance":dfpi,
		"performance":{"OLS":R2_OLS, "XGBoost":R2_XGB}})

def dfloader(dsn, box, mwb, dpath, cpath, tcfs, stdt, fndt, va, BFmin, DrpNF, sub):
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
	if dsn == "esacci":
		ds_bf = ds_bf.sortby("latitude", ascending=False)
	# dfX = ds_bf.to_dataframe().reset_index().drop("time", axis=1).set_index(
	# 	["latitude","longitude"])
	# /// Build some indexers \\\
	latin = np.arange(
		np.floor(ds_bf.latitude.values[0]), 
		np.floor(ds_bf.latitude.values[-1]-1), -(mwb/sub))
	lonin = np.arange(
		np.ceil(ds_bf.longitude.values[0]), 
		np.ceil(ds_bf.longitude.values[-1]+(mwb/sub)))
	
	# ========== Pull out only a subset ove the data to avoid spatial autocorrelation ==========
	ds_sub = ds_bf.reindex({"latitude":latin, "longitude":lonin}, method = "nearest")
	
	# ========== Convert to dataframe, drop time and reset index ==========
	df_sub = ds_sub.to_dataframe()
	df     = df_sub.reset_index().drop("time", axis=1).set_index(["latitude","longitude"]).copy()
	# =================================
	# ========== Climate data =========
	# =================================

	# cpath  = "/mnt/e/Data51/Climate/TerraClimate/"
	cf.pymkdir(cpath+"smoothed/")
	for var in ["ppt", "tmean"]:
		# ========== Read in the climate data ==========
		fnout = f"{cpath}smoothed/TerraClimate_{var}_{mwb}degMW_SeasonalClimatology_{stdt.year}to{fndt.year}.nc"
		if os.path.isfile(fnout):
			print(f"A climatology file alread exists for {var}")
			ds_out = xr.open_dataset(fnout) 
			pix    =  abs(np.unique(np.diff(ds_out.latitude.values))[0]) 
			SF     = np.round(mwb /pix).astype(int)
		else:
			ds_cli = xr.open_mfdataset(
				cpath +f"TerraClimate_{var}_*.nc", 
				drop_variables=["station_influence", "crs"]).rename(
				{"lat":"latitude","lon":"longitude"})

			# ========== subset the data to the box ==========
			ds_cli = ds_cli.sel(dict(
				# time=slice(stdt, fndt),
				latitude=slice(box[3], box[2]), 
				longitude=slice(box[0], box[1])))

			# ========== Roll the dataset and save it ==========
			ds_out, SF   = _roller(mwb, ds_cli, dsn, var, times=[stdt, fndt])
			ds_out.attrs = GlobalAttributes(
				ds_out, var, fnout, "seasonal", 
				stdt, fndt, mwb, typ = "Climatology", ogds="TerraClimate")
			ds_out       = tempNCmaker(ds_out, fnout, var)
		
		# ========== Resample, convert to dataframe and collapse index ==========
		ds_psu = ds_out.reindex({"latitude":latin, "longitude":lonin}, method = "nearest")
		df_psu =  ds_psu.to_dataframe().unstack()
		df_psu.columns = [''.join(col).strip() for col in df_psu.columns.values]
		clorder = [f"{var}{ses}" for ses in ["DJF","MAM","JJA","SON"]]
		df = df.merge(df_psu[clorder], left_index=True, right_index=True)
		
		# Do the same for full res 
		# dsX_psu = ds_out.reindex({
		# 	"latitude":ds_bf.latitude.values, 
		# 	"longitude":ds_bf.longitude.values}, method = "nearest")
		
		# # ========== Convert to dataframe ==========
		# dfX_psu =  dsX_psu.to_dataframe().unstack()
		# dfX_psu.columns = [''.join(col).strip() for col in dfX_psu.columns.values]
		# dfX = dfX.merge(dfX_psu, left_index=True, right_index=True)
	
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

	# breakpoint()
	ds_msu = ds_msk.reindex({"latitude":latin, "longitude":lonin}, method = "nearest")
	df_msu = ds_msu.to_dataframe().reset_index().drop("time", axis=1).set_index(["latitude","longitude"])
	df = df.merge(df_msu, left_index=True, right_index=True)
	# ========== Convert to dataframe ==========
	# dsX_msu = ds_msk.reindex({
	# 	"latitude":ds_bf.latitude.values, 
	# 	"longitude":ds_bf.longitude.values}, method = "nearest")
	# dfX_msu = dsX_msu.to_dataframe().unstack()
	# dfX_msu = dsX_msu.to_dataframe().reset_index().drop("time", axis=1).set_index(["latitude","longitude"])
	# # dfX_msu.columns = [''.join(col).strip() for col in dfX_msu.columns.values]
	# dfX = dfX.merge(dfX_msu, left_index=True, right_index=True)

	# ======================================================
	# ========== Do the final round of processing ==========
	# ======================================================
	try:
		df["AnBF"][df.AnBF <= BFmin]       = np.NaN
		# dfX["AnBF"][dfX.AnBF <= BFmin]       = np.NaN
		# df.drop("datamask", axis=1, inplace=True)
		if DrpNF:
			df["datamask"][df["datamask"] == 0] = np.NaN
			# dfX["datamask"][dfX["datamask"] == 0] = np.NaN
		df.dropna(inplace=True)
		# dfX.dropna(inplace=True)
		# dfX.drop("datamask", axis=1, inplace=True)
	except Exception as er:
		warn.warn(str(er))
		breakpoint()
	return df, ds_msk, ds_bf, latin, lonin

#==============================================================================
def _Obsclim(tmpath, dsn, dfX, cpath, mwb, stdt, fndt, msk, ds_bf, va, drop, lats, lons, rammode, splits = 10):
	"""
	Function to build the observed climate. It just takes all the local arguments in the future prediction 
	function.  
	"""
	# breakpoint()
	# fname = f"{tmpath}S03_FRIdrivers_OBSclim_at_{dsn}.csv"
	# if os.path.isfile(fname):
	# 	print(f"Loading existing full resolution climate data at: {pd.Timestamp.now()}. This might be slow")
	# 	breakpoint()
	# 	df_obs = pd.read_csv(fname, index_col=[0, 1], engine="c", sep=',', dtype=np.float32)
	# 	return df_obs
	# ========== mBuild a new file if needed
	if rammode == "full":
		df_obs = dfX.copy()
		# ========== add the climate columns ==========
		for var in ["ppt", "tmean"]:
			print(f"Loading in the full {var} at: {pd.Timestamp.now()}")
			# ========== Read in the climate data ==========
			fnout  = f"{cpath}smoothed/TerraClimate_{var}_{mwb}degMW_SeasonalClimatology_{stdt.year}to{fndt.year}.nc"
			ds_out = xr.open_dataset(fnout, chunks = {"longitude":265}) 
			df_out =  _reindexer(ds_out, lats, lons, var, dsn = dsn)
			df_obs = df_obs.merge(df_out, left_index=True, right_index=True)

		# ========== Add the mask columns ==========
		# df_obs = df_obs.merge(msk, left_index=True, right_index=True)
		# df_obs.dropna(inplace=True)
	elif rammode == "complex":
		df_obs = dfX.copy()
		
		# ========== add the climate columns ==========
		for var in ["ppt", "tmean"]:
			print(f"Using complex loading with {splits} chunks for {var} at: {pd.Timestamp.now()}")
			# ========== Read in the climate data ==========
			fnout  = f"{cpath}smoothed/TerraClimate_{var}_{mwb}degMW_SeasonalClimatology_{stdt.year}to{fndt.year}.nc"
			ds_out = xr.open_dataset(fnout) 
			ds_out = ds_out.chunk({"latitude": int(ds_out.latitude.size / splits)})
				
			# ========== build a subset of the lats ==========
			sizes      = np.zeros(splits).astype(int)
			sizes[:]   = len(lats)//splits
			sizes[-1] += len(lats)%splits
			# breakpoint()
			Inputt  = iter(lats) 
			Output  = [list(islice(Inputt, elem)) for elem in sizes] # subst of the lats
			
			# ========== loop through the sections ==========
			def _subslice(ds_out, lats, lons, var, dsn, gpnum):
				# function for slicing 
				print(f"{var} group {gpnum} started at: {pd.Timestamp.now()}")
				df_out =  _reindexer(ds_out, lats, lons, var, dsn = dsn)
				return df_out
			df_list = [_subslice(ds_out, latsub, lons, var, dsn, gpnum) for gpnum, latsub in enumerate(Output)]
			df_out  = pd.concat(df_list)
			
			# ========== Check the indexing and merge the dataframes ==========
			if not df_out.index.equals(df_obs.index):
				warn.warn("Dataframes have different indexes. Going interactive")
				breakpoint()


			# breakpoint()
			# df_obs = df_obs.merge(df_out, left_index=True, right_index=True)
			for cl in  df_out.columns.values: 
				df_obs[cl] = df_out[cl].copy()

	# ========== Check the indexing and add the treecover datasets ==========
	assert df_obs.index.equals(msk.index), "Mask and the observed datasets have different indexes"
	# df_obs = df_obs.merge(msk, left_index=True, right_index=True)
	for clx in  msk.columns.values: 
		df_obs[clx] = msk[clx].copy()
	return df_obs

def _ctrend_cal(cpath, stdt, fndt, mwb, df_obs, sen, lats, lons, rammode, splits=10):
	"""
	funtion to calculate future climate an arbitary number of years into the 
	future
	args:
		cpath: path to climate data
		stdt:	start date 
		fndt:	end date
		mwb:	moving window box
		df_obs: the observed version of the dataset
		sen: number of years
	"""
	df_ot = []
	for cvar in ["ppt", "tmean"]:
		# ========== create a dataframe ==========
		df_cli =  df_obs[[col for col in df_obs if col.startswith(cvar)]].copy()
		# ========== loop over the months ==========
		for sea in ["DJF","MAM","JJA","SON"]:
			print(f"calculating {sen}year {sea}{cvar} data at: {pd.Timestamp.now()}")
			if not f"{cvar}{sea}" in df_obs.columns:
				continue
			# ========== make the fn and loat the file ==========
			fn  = cpath + f"TerraClim_{cvar}_{sea}trend_{stdt.year}to{fndt.year}.nc"
			ds_seas = xr.open_dataset(fn, chunks = {"longitude":265}).drop(
				["intercept", "rho", "pval", "FDRsig"])#.squeeze("time",drop=True)

			ds_co, SF = _roller(mwb, ds_seas, f"{cvar}{sea}trend", "trend", times = None)
			ds_co = ds_co.chunk({"longitude":265})
			if rammode in  ["simple", "full"]:
				vals = _reindexer(ds_co, lats, lons, "trend")
			elif rammode == "complex":
				# ========== build a subset of the lats ==========
				sizes      = np.zeros(splits).astype(int)
				sizes[:]   = len(lats)//splits
				sizes[-1] += len(lats)%splits
				# breakpoint()
				Inputt  = iter(lats) 
				Output  = [list(islice(Inputt, elem)) for elem in sizes] # subst of the lats
				
				# ========== loop through the sections ==========
				def _subslice(ds_out, lats, lons, var, gpnum):
					# function for slicing 
					print(f"{var} group {gpnum} started at: {pd.Timestamp.now()}")
					df_out =  _reindexer(ds_out, lats, lons, var)
					return df_out
				df_list = [_subslice(ds_co, latsub, lons, "trend", gpnum) for gpnum, latsub in enumerate(Output)]
				vals  = pd.concat(df_list)

			# ========== Build a temp dataframe to match the indexs and provent issues ==========
			dfp = pd.DataFrame(df_cli[f"{cvar}{sea}"])
			assert dfp.index.equals(vals.index), "Dataframes have different indexes"
			dfp["trend"] = vals["trend"].copy()

			# ========== fix the nans ==========
			dfp["trend"][np.logical_and(dfp["trend"].isnull(), ~dfp[f"{cvar}{sea}"].isnull() )] = 0

			# ========== add the trend  ==========
			df_cli[f"{cvar}{sea}"] = dfp[f"{cvar}{sea}"] + (dfp["trend"] * sen)
		# ========== append the data ==========
		breakpoint()
		df_ot.append(df_cli.copy())
	return df_ot


def _reindexer(ds_out, lats, lons, var, dsn = None):
	# ========== Internal function to reindex data quickly ==========
	with ProgressBar():
		ds_psu = ds_out.reindex(
			{"latitude":lats, "longitude":lons}, 
			method = "nearest").compute()
	# if dsn is None: #or dsn  == "GFED"
	df_psu = ds_psu.to_dataframe().unstack()
	# else:
	# 	df_psu = ds_psu.sel(season="DJF").to_dataframe().unstack()
	# 	breakpoint()

	if var in ["ppt", "tmean"]:
		df_psu.columns = [''.join(col).strip() for col in df_psu.columns.values]
		clorder = [f"{var}{ses}" for ses in ["DJF","MAM","JJA","SON"]]
		return df_psu[clorder].sort_index(ascending=[False, True])
	elif var =="trend":
		# return 1d numpy array
		df_psu.columns = [var]
		return df_psu.sort_index(ascending=[False, True])#.to_numpy().ravel()
	else:
		breakpoint()

def _roller(mwb, ds_cli, dsn, var, times = None):
	# /// Function to calculate spatial moving windows \\\
	# ========== Work out pixel size and scale factors ==========
	pix =  abs(np.unique(np.diff(ds_cli.latitude.values))[0]) 
	SF  = np.round(mwb /pix).astype(int)

	# breakpoint()
	# ========== Make sure the seasons match the method used for climate trend calculation =========
	if var in ["ppt", "tmean"]:
		with ProgressBar():
			if var == "ppt":
				ds_re = ds_cli.resample(time="QS-DEC").sum().compute()
			elif var == "tmean":
				ds_re = ds_cli.resample(time="QS-DEC").sum().compute()
			else:
				breakpoint()
		# /// Shift things into the correct year (from december to jan) \\\
		ds_re = ds_re.assign_coords(time=ds_re.time + pd.Timedelta(31, unit="d"))
		# ========== Temporal subset the data ==========
		if not times is None:
			ds_re = ds_re.sel(dict(time=slice(times[0], times[1])))
		# ds_re = ds_re.isel(time=(ds_re["time.season"] == tstep))

		ds_sea = ds_cli.groupby("time.season").mean()
	else:
		ds_sea = ds_cli
	# ========== Group and roll the data ==========
	print(f"Loading {dsn} {var} data into ram at", pd.Timestamp.now())
	with ProgressBar():
		dsan_lons = ds_sea.rolling(
			{"longitude":SF}, center = True, min_periods=1).mean() 
		ds_out = dsan_lons.rolling(
			{"latitude":SF}, center = True, min_periods=1).mean().compute()
	return ds_out, SF

def _futurePre(dsn, cpath, box, mwb, tcfs, stdt, fndt, ds_bf, lats, lons, sen):
	# =================================================
	# ========== Load the precipitation data ==========
	# =================================================
	var = "ppt"
	fnout = f"{cpath}smoothed/TerraClimate_fut{sen}deg_{var}_{mwb}degMW_SeasonalClimatology_{stdt.year}to{fndt.year}.nc"
	if os.path.isfile(fnout):
		ds_out = xr.open_dataset(fnout, chunks = {"longitude":265})
	else:
		# /// Load and process future precip \\\
		ds_cli = xr.open_mfdataset(
			cpath +f"{sen}deg/TerraClimate_{sen}c_{var}_*.nc", 
			drop_variables=["station_influence", "crs"]).rename(
			{"lat":"latitude","lon":"longitude"})

		# ========== subset the data to the box ==========
		ds_cli = ds_cli.sel(dict(
			# time=slice(stdt, fndt),
			latitude=slice(box[3], box[2]), 
			longitude=slice(box[0], box[1])))

		# ========== roll the dataset ==========
		ds_out, SF = _roller(mwb, ds_cli, dsn, var, times=[stdt, fndt])
		ds_out.attrs = GlobalAttributes(
			ds_out, "ppt", fnout, f"seasonal_{sen}deg", 
			stdt, fndt, mwb, typ = "FutureClimatology", ogds="TerraClimate")
		ds_out       = tempNCmaker(ds_out, fnout, var)
	# ========== Resample the dataset to match the BA ==========
	with ProgressBar():
		ds_msu = ds_out.reindex({
			"latitude" :lats, 
			"longitude":lons}, method = "nearest")
	# ========== Convert to dataframe ==========
	df_msu =  ds_msu.to_dataframe().unstack()
	df_msu.columns = [''.join(col).strip() for col in df_msu.columns.values]
	return df_msu


def _futureTemp(dsn, cpath, box, mwb, tcfs, stdt, fndt, ds_bf, lats, lons, sen):
	# ===============================================
	# ========== Load the temperature data ==========
	# ===============================================
	var = "tmean"
	fnout = f"{cpath}smoothed/TerraClimate_fut{sen}deg_{var}_{mwb}degMW_SeasonalClimatology_{stdt.year}to{fndt.year}.nc"
	if os.path.isfile(fnout):
		ds_out = xr.open_dataset(fnout, chunks = {"longitude":265})
	else:
		# /// Load and process future precip \\\
		# ========== ==========
		ds_tmax = xr.open_mfdataset(
			cpath +f"{sen}deg/TerraClimate_{sen}c_tmax_*.nc", 
			drop_variables=["station_influence", "crs"]).rename(
			{"lat":"latitude","lon":"longitude"})#.rename({"tmax":"tmean"})
		ds_tmin = xr.open_mfdataset(
			cpath +f"{sen}deg/TerraClimate_{sen}c_tmin_*.nc", 
			drop_variables=["station_influence", "crs"]).rename(
			{"lat":"latitude","lon":"longitude"})#.rename({"tmax":"tmean"})
	
		# ========== do the algebra ==========
		ds = xr.merge([ds_tmax, ds_tmin])
		ds["tmean"] = (ds["tmax"] + ds["tmin"])/2
		ds = ds.drop(["tmin", "tmax"])

		# ========== subset the data to the box ==========
		ds = ds.sel(dict(
			# time=slice(stdt, fndt),
			latitude=slice(box[3], box[2]), 
			longitude=slice(box[0], box[1])))

		# ========== roll the dataset ==========
		ds_out, SF = _roller(mwb, ds, dsn, "tmean")
		ds_out.attrs = GlobalAttributes(
			ds_out, "ppt", fnout, f"seasonal_{sen}deg", 
			stdt, fndt, mwb, typ = "FutureClimatology", ogds="TerraClimate")
		ds_out       = tempNCmaker(ds_out, fnout, var)

	# ========== Resample the dataset to match the BA ==========
	with ProgressBar():
		ds_msu = ds_out.reindex({
			"latitude" :lats, 
			"longitude":lons}, method = "nearest")
	# ========== Convert to dataframe ==========
	df_msu =  ds_msu.to_dataframe().unstack()
	df_msu.columns = [''.join(col).strip() for col in df_msu.columns.values]
	return df_msu
	
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
	elif sysname == 'burrell-pre5820':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51"
		dpath = "./data"
		chunksize = 300
		cpath  = "/mnt/d/Data51/Climate/TerraClimate/"
	elif sysname =='LAPTOP-8C4IGM68':
		dpath = "./data"
		chunksize = 300
		cpath  = "/mnt/e/Data51/Climate/TerraClimate/"
	elif sysname in ['arden-worstation', 'arden-Precision-5820-Tower-X-Series']:
		# WHRC linux distro
		cpath= "/media/arden/SeagateMassStorage/Data51/Climate/TerraClimate/"
		dpath = "./data"
		chunksize = 300
	else:
		ipdb.set_trace()
	return dpath, cpath, chunksize	

def GlobalAttributes(ds, var, fnout, tstep, stdt, fndt, mwb, typ = "Climatology", ogds="TerraClimate"):
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
	attr["title"]               = f"{tstep}{var}{typ}"
	attr["summary"]             = f"{ogds} {var} {typ} from {stdt} to {fndt}." 
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

def tempNCmaker(ds, fnout, var):

	""" Function to save out a tempary netcdf """
	

	delayed_obj = ds.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		unlimited_dims = ["time"],
		compute=False)

	print("Starting write of %s data at" % var, pd.Timestamp.now())
	with ProgressBar():
		results = delayed_obj.compute()

	dsout = xr.open_dataset(fnout) 
	return dsout
#==============================================================================

if __name__ == '__main__':
	main()