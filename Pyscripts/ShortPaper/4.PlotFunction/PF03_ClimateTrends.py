"""
Script goal, 

Make the climate maps

"""
#==============================================================================

__title__ = "Climate stat calculator"
__author__ = "Arden Burrell"
__version__ = "v1.0(11.11.2020)"
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
import subprocess as subp
from dask.diagnostics import ProgressBar
import dask

from collections import OrderedDict
# from cdo import *

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
import seaborn as sns

# import seaborn as sns
import cartopy as ct
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
import matplotlib.colors as mpc
import matplotlib.patheffects as pe
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from statsmodels.stats.weightstats import DescrStatsW
import pickle
from sklearn import metrics as sklMet

# ========== Import ml packages ==========
import sklearn as skl
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
import ipdb

print("numpy version   : ", np.__version__)
print("pandas version  : ", pd.__version__)
print("xarray version  : ", xr.__version__)
print("cartopy version : ", ct.__version__)
print("sklearn version : ", skl.__version__)

#==============================================================================
def main():
	# ========== find the paths ==========
	dpath, cpath = syspath()
	ppath = "./plots/ShortPaper/PF03_Climate/"
	cf.pymkdir(ppath)
	pbounds = [10.0, 170.0, 70.0, 49.0]
	maskver = "Boreal"

	# ========== Build the annual plots ==========
	AnnualPlotmaker(setupfunc("annual", rmaps = False), dpath, cpath, ppath, pbounds, maskver, rmaps = False)
	# ========== Make a figure of the model output ==========
	ModelPrediction(ppath)
	# ========== Build the seasonal plots ==========
	Seasonalplotmaker(setupfunc("seasonal", rmaps = True), dpath, cpath, ppath, pbounds, maskver, "Climatology")
	Seasonalplotmaker(setupfunc("seasonal"), dpath, cpath, ppath, pbounds, maskver, "trend")
	# AnnualPlotmaker(setupfunc("annual", rmaps = True), dpath, cpath, ppath, pbounds, maskver, rmaps = True)
	# breakpoint()

	


	# ModelPrediction(ppath, model="OLS")

	# g = sns.FacetGrid(df, col="Method",  hue="Dataset")
	# g.map(sns.barplot, "Predictor", "Score", order=df.Predictor.unique().tolist())
	
def ModelPrediction(ppath, model = 'XGBoost'):
	altnames = ({"GFED":"GFED4", "MODIS":"MCD64A1", "esacci":"FireCCI51", "COPERN_BA":"CGLS-BA", "HANSEN_AFmask":"HansenGFC-MAF", "HANSEN":"HansenGFC"}) 
	# ========== set the mpl rc params ==========
	sns.set_style("whitegrid")
	font = ({
		'weight' : 'bold',
		'size'   : 11, 
		})
	# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	

	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})


	# ========== model loader ==========
	Varimp = []
	perfm  = OrderedDict()
	for dsn in ["MODIS","GFED", "esacci", "COPERN_BA"]:
		try:
				# vp, df = ModelLoadter(dsn=dsn)
			for mod, sen in enumerate([30, 60, 100]):
				# vp, df = ModelLoadter(dsn=dsn, )
				vp, df =ModelLoadter(dsn=dsn, sen=sen, mod=mod, model=model)
				Varimp.append(vp) #sen=sen, mod=mod
				
				if mod == 0: 
					perfm[altnames[dsn]] = df
		except Exception as er:
			warn.warn(str(er))
			breakpoint()

	df = pd.concat(Varimp)#.reset_index().rename({"index":"Predictor"}, axis=1)

	# =======================================
	# ========== Build the figures ==========
	# =======================================

	# ========== Setup the axes ==========
	fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12,14), gridspec_kw={'height_ratios':[1.8, 1.8, 1]})#, constrained_layout=True)

	# ========== build the skill plots ==========
	for dsn, alp, ax, ytl, xtl in zip(perfm, ["a", "b", "c", "d"], axs.flat[:4],[True, False, True, False],[False, False, True, True]):
		# +++++ setup the colormap +++++
		# cmap = mpc.ListedColormap(['#FFFFFF'] + palettable.colorbrewer.sequential.YlOrRd_9.hex_colors)
		# cmap = mpc.ListedColormap(['#FFFFFF'] + palettable.cmocean.sequential.Matter_9.mpl_colors)
		cmap = mpc.ListedColormap(palettable.cmocean.sequential.Matter_10.mpl_colors)
		mask = perfm[dsn] == 0
		
		# +++++ make the figure +++++
		sns.heatmap(perfm[dsn], mask=mask, annot=False, vmin=0, vmax=1,  cbar=(not ytl),
			square=True, cmap=cmap, ax = ax, yticklabels=ytl, xticklabels=xtl, linewidths=.25, linecolor="whitesmoke")
		ax.set_title(f"{alp}) {dsn}", loc= 'left')
		# ax.grid()
		# breakpoint()
		# ========== Alternate title approach ===========
		# txt = f"{alp}) {dsn}"
		# ax.text(-0.15, 1.05, txt, transform=ax.transAxes, )
					# size=8, weight='bold')#, zorder=106)
		# breakpoint()
		# X tick labels 
		ax.set_xticks(np.arange(perfm[dsn].shape[0] +1))
		ax.set_xticklabels([0, 15, 30, 60, 120, 500, 1000, 3000, 10000])
		if xtl:
			ax.set(xlabel="Observed FRI")
		# Y tick labels 
		ax.set_yticks(np.arange(perfm[dsn].shape[0] +1))
		ax.set_yticklabels(np.flip([0, 15, 30, 60, 120, 500, 1000, 3000, 10000]), rotation = 45)
		if ytl:
			ax.set_ylabel("Predicted FRI", labelpad=2)
		
		ax.plot(np.flip(np.arange(perfm[dsn].shape[0] +1)), np.arange(perfm[dsn].shape[0] +1), "black", alpha=0.5)
		# ax.spines['left'].set_color('black')
		# ax.spines['bottom'].set_color('black')
		# ax.invert_yaxis()

		# breakpoint()
	# breakpoint()
	for alp, ax, meth, ytl in zip(["e", "f"],axs[-1,], ["Permutation Importance", "Feature Importance"], [True, False]):
		# sns.barplot(y="Predictor", x="Score", hue="Dataset", data= df[df.Method == meth], ax=ax, orient="h")#, yticklabels=ytl)#, kind="bar", col="Method")
		# ax.set_xlim(0, 0.5)
		chex = palettable.cartocolors.qualitative.Vivid_4.hex_colors
		sns.barplot(x="Predictor", y="Score", hue="Model", ci=None,  data= df[df.Method == meth], ax=ax, palette=chex)#, legend=(not ytl))#, yticklabels=ytl)#, kind="bar", col="Method")
		ax.set(ylabel=None)
		ax.set(xlabel=None)
		ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha="right")
		ax.set_ylim(0, 0.5)
		
		if not ytl:
			ax.set(yticklabels=[])
		else:
			ax.legend().set_visible(False)
		ax.set_title(f"{alp}) {meth}", loc= 'left')
		# txt = f"{alp}) {meth}"
		# ax.text(-0.15, 1.05, txt, transform=ax.transAxes, )
		# 	ax.set(yticklabels=[])
		

	# sns.barplot(x="Predictor", y="Score")
	fig.tight_layout()

	plotfname = f"{ppath}PF03_ModelPrediction_{model}."
	# breakpoint()
	for fmt in ["tiff", "eps", "png"]:# "pdf"
		print(f"Starting {fmt} plot save at:{pd.Timestamp.now()}")
		plt.savefig(plotfname+fmt)#, dpi=dpi)
	
	print("Starting plot show at:", pd.Timestamp.now())
	plt.show()

	if not (plotfname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotfname, gitinfo]
		cf.writemetadata(plotfname, infomation)


	plt.show()
	# breakpoint()

	# sns.catplot(x="Predictor", y="Score", hue="Dataset", data=df, kind="bar", col="Method")


#==============================================================================
def ModelLoadter(dsn="esacci", sen=30, version=0, model = 'XGBoost', mod=0):
	"""
	Load in the models 
	"""
	altnames = ({"GFED":"GFED4", "MODIS":"MCD64A1", "esacci":"FireCCI51", "COPERN_BA":"CGLS-BA", "HANSEN_AFmask":20, "HANSEN":20}) 

	va    = "AnBF"
	drop  = ["AnBF", "FRI", "datamask"]

	tmpath = "./results/ProjectSentinal/FRImodeling/"
	modfn  = f"{tmpath}models/S03_FRIdrivers_{dsn}_v{version}_{sen}yr_trendPrediction.dat" 
	ttfn  = f"{tmpath}models/TestTrain/S03_FRIdrivers_{dsn}_v{version}_{sen}yr_TestTrain.csv"

	# ========== load the model and the dataset ==========
	models = pickle.load(open(modfn, "rb"))
	df     = pd.read_csv(ttfn, index_col=0).set_index(["latitude","longitude"])

	# ========== split the data	========== 
	y  = df[va]
	X_t, X_ts, y_train, y_test = train_test_split(
		df, y, test_size=0.2, random_state=42)
	# ========== subset the dataset ==========
	X_tn  = X_t.drop(drop, axis=1)
	X_tst = X_ts.drop(drop, axis=1)
	
	# ========== perform some transforms ==========
	if not  models['transformer'] is None:	
		X_train = models['transformer'].fit_transform(X_tn)
		X_test  = models['transformer'].transform(X_tst)
	
	# ========== do the prediction and get the performance ==========
	y_pred = models['models'][model].predict(X_test)
	R2_XGB = sklMet.r2_score(y_test, y_pred)

	pr_df = pd.DataFrame({"Observed":y_test.values,"Predicted":y_pred})
	# "Residual":y_pred-y_test.values
	dfT = 1/pr_df
	pr_df["Residual"]= pr_df.Predicted -pr_df.Observed
	dfT["Residual"]= dfT.Predicted -dfT.Observed

	print (dsn, R2_XGB, models['performance'])
	
	if not (model == "XGBoost"):
		warn.warn("Importance does not work for OLS yet")

	modim = models["Importance"][["XGBPermImp",  "XGBFeatImp"]].reset_index().melt(id_vars="index", value_vars=["XGBPermImp",  "XGBFeatImp"])
	# models["Importance"][["XGBPermImp",  "XGBFeatImp"]].reset_index().rename(
	# 	{"XGBPermImp":"Permutation Importance",  "XGBFeatImp":"Feature Importance"}, axis=1).melt()
	modim.replace({"XGBPermImp":"Permutation Importance",  "XGBFeatImp":"Feature Importance"}, inplace=True)
	modim = modim.rename({"index":"Predictor", "variable":"Method", "value":"Score"}, axis=1)
	modim["Model"] = altnames[dsn]
	modim["Version"] = mod


	# ========== Get the observed values ==========
	split    = np.array([0, 15, 30, 60, 120, 500, 1000, 3000, 10000])
	df_class = pd.DataFrame({"Observed":(1/y_test.values), "Estimated":(1/y_pred) })
	expsize  = len(split) -1 # df_class.experiment.unique().size
	
	if (df_class.Estimated.max()>10000) or (df_class.Estimated.max()<0):
		 # getting some bad values
		df_class.where(df_class <0, 0)
		df_class.where(df_class >10000, 10000)
		# breakpoint()

	df_class["ObservedC"]  =  pd.cut(df_class["Observed"], split, labels=np.arange(expsize))
	df_class["EstimatedC"] =  pd.cut(df_class["Estimated"], split, labels=np.arange(expsize))
	df_class.dropna(inplace=True)

	# cMat  = sklMet.confusion_matrix(
	# 	df_class["ObservedC"], df_class["EstimatedC"], labels=df_class["ObservedC"].cat.categories).astype(int) 
	# cCor  = np.tile(
	# 	df_class.groupby("ObservedC").count()["EstimatedC"].values.astype(float), (cMat.shape[0], 1)).T
	# # breakpoint()
	# conM =  ( cMat/cCor)#.T
	# conM[np.logical_and((cCor == 0), (cMat==0))] = 0.#.T
	# ========== Create a dataframe and set the column and rownames ==========
	# df_cm = pd.DataFrame(conM, index = [int(i) for i in np.arange(expsize)], columns = [int(i) for i in np.arange(expsize)])

	try:
		df_cm  = pd.DataFrame(
			sklMet.confusion_matrix(df_class["ObservedC"], df_class["EstimatedC"],  labels=df_class["ObservedC"].cat.categories,  normalize='true'),  
			index = [int(i) for i in np.arange(expsize)], columns = [int(i) for i in np.arange(expsize)]).T.sort_index(ascending=False)
		
	except Exception as er:
		warn.warn(str(er))
		breakpoint()

	# breakpoint()
	return modim, df_cm#.iloc[::-1] # reversed order


def Seasonalplotmaker(setup, dpath, cpath, ppath, pbounds, maskver, cli):
	""" 
	Function fo making the Seasonal plots
	args:
		setup: Ordered dict 
			contains the cmap and vrng infomation
		dpath:	str
		cpath:	str
			path to the climate data
	"""
	# ========== load the mask ==========
	# ========== load the mask ==========
	fnmask = f"{dpath}/masks/broad/Hansen_GFC-2018-v1.6_SIBERIA_ProcessedToTerraClimate.nc"
	fnBmask = f"./data/LandCover/Regridded_forestzone_TerraClimate.nc"

	with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask, xr.open_dataset(fnBmask).drop(["DinersteinRegions", "GlobalEcologicalZones", "LandCover"]) as Bmask:
		# breakpoint()
		if maskver == "Boreal":
			msk    = (dsmask.mask.isel(time=0)*((Bmask.BorealMask.isel(time=0)>0).astype("float32")))#.sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
		else:
			msk    = (dsmask.mask.isel(time=0)).astype("float32")
		msk = msk.values

		# +++++ Change the boolean mask to NaNs +++++
		msk[msk == 0] = np.NAN
	
	# ========== set the mpl rc params ==========
	font = ({
		'weight' : 'bold',
		'size'   : 11, 
		})
	
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})
	latiMid=np.mean([pbounds[2], pbounds[3]])
	longMid=np.mean([pbounds[0], pbounds[1]])

	# ========== Create the figure ==========
	fig, axs = plt.subplots(
		4, 2, sharex=True, 
		subplot_kw={'projection': ccrs.Orthographic(longMid,latiMid)}, 
		figsize=(14,13)
		)

	# ========== Loop over the rows ==========
	for sea, (row, raxes) in zip(["DJF", "MAM", "JJA", "SON"], enumerate(axs)):
		# ========== Loop over the variables ==========
		for va, ax in zip(setup, raxes):
			# ========== Read in the data and mask the boreal zone ==========
			if cli == "trend":
				ds = xr.open_dataset(f"{cpath}TerraClim_{va}_{sea}trend_1985to2015.nc")
				ds = ds.where(msk == 1)
				ds.slope.attrs = setup[va]["attrs"]

				p  = ds.slope.isel(time=0).plot(
					cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
					transform=ccrs.PlateCarree(), ax=ax,
					    cbar_kwargs={"pad": 0.02, "shrink":0.85, "extend":"both"})

				# ========== work out the stippling ==========
				slats, slons = _stippling(ds, squeeze=30, nanfrac = 0.15, sigfrac=0.5)
				ax.scatter(
					slons, slats, s=4, c='k', marker='X', 
					facecolors='none', edgecolors="none",  
					alpha=0.35, transform=ccrs.PlateCarree())
				print(sea, va, ds.slope.quantile([0.01,0.05, 0.50,0.95,0.99]))
			else:
				ds = xr.open_dataset(f"{cpath}smoothed/TerraClimate_{va}_1degMW_SeasonalClimatology_1985to2015.nc")
				ds = ds.where(msk == 1)
				# breakpoint()
				ds[va].attrs = setup[va]["attrs"]
				

				p  = ds[va].sel(season=sea).plot(
					cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
					transform=ccrs.PlateCarree(), ax=ax,
					    cbar_kwargs={"pad": 0.02, "shrink":0.85, "extend":setup[va]['extend']})
				print(sea, va, ds[va].sel(season=sea).quantile([0.01,0.05, 0.50,0.95,0.99]))


			ax.set_extent(pbounds, crs=ccrs.PlateCarree())
			# ax.gridlines()
			gl = ax.gridlines(draw_labels= True, dms=True, x_inline=False, y_inline=False)#{"bottom": "x", "Top": "y"}
			gl.xlocator = mticker.FixedLocator([60, 120])
			gl.ylocator = mticker.FixedLocator([50, 60, 70])

			# p.axes.add_feature(cpf.COASTLINE, , zorder=101)
			coast = cpf.GSHHSFeature(scale="high")
			ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
			ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
			ax.add_feature(coast, zorder=101, alpha=0.5)
			ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
			ax.add_feature(cpf.RIVERS, zorder=104)
			
			# ========== Set the center titles ==========
			if row == 0:
				vanm = setup[va]["lname"]
				ax.set_title(f"{vanm}")	
			else:
				ax.set_title("")
			# ========== Set the left titles ==========
			if va == "ppt":
				ax.set_title(f"{sea}", loc= 'left')
			

	# plt.subplots_adjust(top=0.971,bottom=0.013, left=0.011, right=0.97, hspace=0.0, wspace=0.0)
	plt.subplots_adjust(top=0.94, bottom=0.020,left=0.011,right=0.967,hspace=0.2,wspace=0.0)
	# plt.subplots_adjust(top=0.971, bottom=0.013, left=0.012, right=0.988, hspace=0.063, wspace=0.2)
	# ========== Save the plots ==========
	if cli == "trend":
		plotfname = f"{ppath}PF03_SeasonalClimateTrend."
	else:
		plotfname = f"{ppath}PF03_SeasonalClimatology."

	for fmt in [ "png", "tiff", "eps" ]:#,"pdf"
		print(f"Starting {fmt} plot save at:{pd.Timestamp.now()}")
		plt.savefig(plotfname+fmt)#, dpi=dpi)
	
	print("Starting plot show at:", pd.Timestamp.now())
	plt.show()

	if not (plotfname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotfname, gitinfo]
		cf.writemetadata(plotfname, infomation)
	breakpoint()

def AnnualPlotmaker(setup, dpath, cpath, ppath, pbounds, maskver, rmaps = False):
	""" Function fo making the annual plot
	args:
		setup: Ordered dict 
			contains the cmap and vrng infomation
		dpath:	str
		cpath:	str
			path to the climate data
	"""
	# ========== load the mask ==========
	fnmask = f"{dpath}/masks/broad/Hansen_GFC-2018-v1.6_SIBERIA_ProcessedToTerraClimate.nc"
	fnBmask = f"./data/LandCover/Regridded_forestzone_TerraClimate.nc"


	with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask, xr.open_dataset(fnBmask).drop(["DinersteinRegions", "GlobalEcologicalZones", "LandCover"]) as Bmask:
		# breakpoint()
		if maskver == "Boreal":
			msk    = (dsmask.mask.isel(time=0)*((Bmask.BorealMask.isel(time=0)>0).astype("float32")))#.sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
		else:
			msk    = (dsmask.mask.isel(time=0)).astype("float32")
		
		msk = msk.values

		# +++++ Change the boolean mask to NaNs +++++
		msk[msk == 0] = np.NAN
		
		# print("Masking %s frame at:" % dsn, pd.Timestamp.now())
		# +++++ mask the frame +++++
		# breakpoint()
		# frame *= msk
	latiMid=np.mean([pbounds[2], pbounds[3]])
	longMid=np.mean([pbounds[0], pbounds[1]])
	# ========== set the mpl rc params ==========
	font = ({
		'weight' : 'bold',
		'size'   : 11, 
		})

	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})
	if rmaps:
		nrows = 3
		alpC  = ["a", "b", "c", "d", "e", "f"]
	else:
		nrows = 2
		alpC  = ["a", "b", "c", "d"]
	# ========== Create the figure ==========
	fig, axs = plt.subplots(
		nrows, 2, sharex=True, subplot_kw={'projection': ccrs.Orthographic(longMid,latiMid)}, 
		figsize=(17, nrows * 4)
		)

	# ========== load the datasets ==========
	for va, ax, let in zip(setup, axs.flatten(), alpC):
		if va in ["ppt", "tmean"]:
			cli = "Trend"
		elif va in ["pptC", "tmeanC"]:
		# for cli, ax, let in zip(["Climatology", "Trend"], axa, lets):
			cli = "Climatology"
		else:
			cli = "Land Cover"
		vanm = setup[va]["lname"]
		print(va, cli, vanm)
		# ========== Make the trend ==========
		if cli == "Trend":
			# ========== Read in the data and mask the boreal zone ==========
			ds = xr.open_dataset(f"{cpath}TerraClim_{va}_annualtrend_1985to2015.nc")
			ds.slope.attrs = setup[va]["attrs"]
			ds = ds.where(msk == 1)
			p  = ds.slope.isel(time=0).plot(
				cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
				transform=ccrs.PlateCarree(), ax=ax,
				    cbar_kwargs={
				    "pad": 0.015, "shrink":0.80, "extend":"both"
				    })
			# ========== work out the stippling ==========
			slats, slons = _stippling(ds, squeeze=10, nanfrac = 0.15, sigfrac=0.5)
			ax.scatter(
				slons, slats, s=4, c='k', marker='X', 
				facecolors='none', edgecolors="none",  
				alpha=0.35, transform=ccrs.PlateCarree())
			print(f"Annual trend {va}", ds.slope.quantile([0.01,0.05, 0.50,0.95,0.99]))
			ds = None
			axtitle = f"{let}) {vanm} {cli}"
		elif cli == "Climatology":
			extend =  'max'
			if va == "pptC":
				ds = xr.open_dataset(f"{cpath}smoothed/TerraClimate_ppt_1degMW_SeasonalClimatology_1985to2015.nc")
				ds = ds.sum(dim='season')
				vax = "ppt"
			else:
				ds = _annualtempmaker("tmean", cpath,  funb =bn.nanmax, func="max")
				vax = "tmean"
				# ds = _annualtempmaker(va, cpath,  funb =bn.nanmean, func="mean")
				# extend="both"
			# breakpoint()
			# ds = ds.where(np.squeeze(dsmask.datamask.values) == 1)
			ds = ds.where(msk == 1)

			ds[vax].attrs = setup[va]["attrs"]
			p  = ds[vax].plot(
				cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
				transform=ccrs.PlateCarree(), ax=ax,
				    cbar_kwargs={
				    "pad": 0.015, "shrink":0.80, "extend":extend
				    })
			print(f"Annual Climate {vax}", ds[vax].quantile([0.01,0.05, 0.50,0.95,0.99]))
			ds = None
			axtitle = f"{let}) {vanm} {cli}"
		else:
			# breakpoint()
			# ========== load the dataset and pull out the relevant parts ==========
			ds = xr.open_dataset(fnBmask)
			# +++++ remap the values +++++
			for vrm in setup[va]['valmap']:	
				ds[va] = ds[va].where(~(ds[va] == vrm), setup[va]['valmap'][vrm])
			ds[va] = np.fabs(ds[va])
			ds[va].attrs = setup[va]["attrs"]
			
			p  = ds[va].plot(
				cmap=setup[va]["cmap"], vmin=setup[va]["vmin"], vmax=setup[va]["vmax"],
				transform=ccrs.PlateCarree(), ax=ax,
				    cbar_kwargs={
				    "pad": 0.015, "shrink":0.80, "extend":"neither"
				    })
			cbar = p.colorbar
			keys =  pd.DataFrame({va:setup[va]["kys"]}).reset_index()
			# cbar.set_ticklabels( keys.Code.values)  # horizontal colorbar

			cbar.set_ticks( keys["index"].values)
			cbar.set_ticklabels(keys[va].values)
			axtitle = f"{let}) {cli}"

			# ========== bring in the locations ==========
			df_loc = setup[va]['places']
			ax.scatter(
				df_loc.lon.values, df_loc.lat.values, s=20, c='k', 
				facecolors='none', edgecolors="none",  
				alpha=1, transform=ccrs.PlateCarree())
			for index, row in df_loc.iterrows():

				ax.text(
					row.lon+row.lonoffset, row.lat+row.latoffset, row.Name, 
					horizontalalignment='left',transform=ccrs.PlateCarree(), 
					zorder=105, color="lightgray", path_effects=[pe.withStroke(linewidth=0.75, foreground="black")])


				# marker='X', 

			# breakpoint()
			# plt.show()



		ax.set_extent(pbounds, crs=ccrs.PlateCarree())
		# ax.gridlines()
		gl = ax.gridlines(draw_labels= True, dms=True, x_inline=False, y_inline=False)#{"bottom": "x", "Top": "y"}
		gl.xlocator = mticker.FixedLocator([60, 120])
		gl.ylocator = mticker.FixedLocator([50, 60, 70])

		coast = cpf.GSHHSFeature(scale="intermediate")
		# p.axes.add_feature(cpf.COASTLINE, , zorder=101)
		ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
		ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
		ax.add_feature(coast, zorder=101, alpha=0.5)
		ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		ax.add_feature(cpf.RIVERS, zorder=104)
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
		# ========== Set the titles ==========
		ax.set_title("")
		ax.set_title(axtitle, loc= 'left')
		
	# ========== Save the plots ==========

	# plt.subplots_adjust(top=0.971, bottom=0.013, left=0.012, right=0.988, hspace=0.001, wspace=0.000)
	# plt.subplots_adjust(top=0.971,bottom=0.013, left=0.011, right=0.97, hspace=0.0, wspace=0.0)
	plt.subplots_adjust(top=0.93, bottom=0.025,left=0.011,right=0.967,hspace=0.25,wspace=0.0)
	# +++++ remove unised ax +++++
	if len(setup)%2:
		axs[-1, -1].remove()

	# breakpoint()
	# plt.tight_layout()
	plotfname = f"{ppath}PF03_AnnualClimateAndTrend."
	# breakpoint()
	for fmt in ["png", "tiff", "eps"]:# "pdf"
		print(f"Starting {fmt} plot save at:{pd.Timestamp.now()}")
		plt.savefig(plotfname+fmt)#, dpi=dpi)
	
	print("Starting plot show at:", pd.Timestamp.now())
	plt.show()

	if not (plotfname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotfname, gitinfo]
		cf.writemetadata(plotfname, infomation)
	breakpoint()


# ==============================================================================
def _stippling(ds, squeeze=10, nanfrac = 0.25, sigfrac=0.5):
	"""
	function to change calculate the stipling
	"""
	# /// pull fraction sig \\\\
	da = (ds.FDRsig.squeeze("time").coarsen({"latitude":squeeze, "longitude":squeeze}, boundary ="pad").mean())>sigfrac#).astype(int)
	# /// pull the nan fraction \\\\
	da *= (ds.FDRsig.squeeze("time").isnull().coarsen({"latitude":squeeze, "longitude":squeeze}, boundary ="pad").mean()<nanfrac)
	da = da.astype(np.float32).where(da)
	# /// convert to a dataframe \\\\
	df = xr.Dataset({"sig":da}).to_dataframe().drop("time", axis=1).dropna().reset_index()
	return df.latitude.values, df.longitude.values

def setupfunc(time, rmaps = False):
	cmaps = _cmapsfun()
	# ========== Build an ordered dict with key info ==========
	setup = OrderedDict()
	if time == "seasonal":
		if not rmaps:
			setup["ppt"]   = ({"vmin":-2., "vmax":2, "cmap":cmaps["ppt"], "lname":"Precipitation",
				"attrs":{'long_name':"Trend", "units":r"mm yr$^{-1}$"}})
			setup["tmean"] = ({"vmin":-0.08, "vmax":0.08, "cmap":cmaps["tmean"], "lname":"Temperature",
				"attrs":{'long_name':"Trend", "units":r"$^{o}$C yr$^{-1}$"}})
		else:
			setup["ppt"]   = ({"vmin":0, "vmax":140, "cmap":cmaps["pptC"], "lname":"Precipitation",
				"attrs":{'long_name':"Annual total", "units":r"mm"}, "extend":"max"})
			setup["tmean"] = ({"vmin":-30, "vmax":30, "cmap":cmaps["tmeanC"], "lname":"Temperature",
				"attrs":{'long_name':"Monthly max", "units":r"$^{o}$C"}, "extend":"both"})


	elif time == "annual":
		shrink = 0.90
		setup["pptC"]   = ({"vmin":0, "vmax":400, "cmap":cmaps["pptC"], "lname":"Precipitation",
			"attrs":{'long_name':"Annual total", "units":r"mm"}, "shrink":shrink, "extend":"max"})
		setup["tmeanC"] = ({"vmin":10, "vmax":25, "cmap":cmaps["tmeanC"], "lname":"Temperature",
			"attrs":{'long_name':"Monthly max", "units":r"$^{o}$C"}, "shrink":shrink, "extend":"both"})

		setup["ppt"]   = ({"vmin":-4., "vmax":4, "cmap":cmaps["ppt"], "lname":"Precipitation",
			"attrs":{'long_name':"Trend", "units":r"mm yr$^{-1}$"}, "shrink":shrink, "extend":"both"})
		setup["tmean"] = ({"vmin":0, "vmax":0.06, "cmap":cmaps["tmean"], "lname":"Temperature",
			"attrs":{'long_name':"Trend", "units":r"$^{o}$C yr$^{-1}$"}, "shrink":shrink, "extend":"max"})
		
		if rmaps:
			# ========== make the kes foir the figure ==========
			df_lc = pd.read_csv("./data/LandCover/glc2000_v1_1/Tiff/Global_Legend.csv")
			df_lc["GROUP"].replace(0, np.NaN,inplace=True)
			df_lc["GROUP"].replace(1, np.NaN,inplace=True)
			exc = OrderedDict()
			for val, gp in zip(np.flip(df_lc.VALUE.values), np.flip(df_lc.GROUP.values)):
				exc[val]= -gp
			# kys = ({0:"FBD", 1:"FCE", 2:"FCD", 3:"FMx", 4:"SHC", 5:"CMA", 6:"BG", 7:"WSI", 8:"Oth"})
			# breakpoint()
			kys = ({ 2:"BG", 3:"CMA", 4:"SHC", 5:"FMx", 6:"FCD", 7:"FCE", 8:"FBD"})#, 1:"WSI",

			setup["LandCover"] = ({"vmin":1.5, "vmax":8.5, "cmap":cmaps["LandCover"],"lname":"Land Cover",
				"valmap":exc, "kys":kys, "attrs":{'long_name':"Land Cover Class"}, "places": _locations(), 
				"shrink":shrink,})
			
			

			# ========== Deinstine regions ==========
			# setup["DinersteinRegions"] = ({"vmin":0, "vmax":7, "cmap":cmaps["pptC"],"lname":"Land Cover",
			# 	})

	return setup

def _cmapsfun():
	"""
	Funtion to make the colourmaps 

	"""
	pcmap = palettable.colorbrewer.diverging.BrBG_11.mpl_colormap
	cmapP = []
	for point  in np.arange(0, 1.01, 0.05):
		cmapP.append( pcmap(point)[0:-1])

	pptcmap = mpc.ListedColormap(cmapP)	
	# pptcmapC = mpc.ListedColormap(palettable.colorbrewer.diverging.BrBG_10.mpl_colors)

	pcmapC = palettable.colorbrewer.diverging.BrBG_10.mpl_colormap
	cmap = []
	for point  in np.arange(0, 1.01, 0.05):
		cmap.append( pcmapC(point)[0:-1])

	pptcmapC = mpc.ListedColormap(cmap)
	# breakpoint()

	pptcmap.set_bad('dimgrey',1.)
	pptcmapC.set_bad('dimgrey',1.)
	
	# tmncmap = mpc.ListedColormap(palettable.colorbrewer.diverging.PuOr_11_r.mpl_colors)
	# tmncmap = mpc.ListedColormap(palettable.colorbrewer.diverging.RdBu_11_r.mpl_colors)
	tmncmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_19.mpl_colors)
	tmncmap.set_bad('dimgrey',1.)

	tmncmapC = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors)
	tmncmapC.set_bad('dimgrey',1.)

	
	LCcmap = mpc.ListedColormap(palettable.cartocolors.qualitative.Prism_7_r.mpl_colors)
	LCcmap.set_bad('dimgrey',1.)

	return {"ppt":pptcmap, "tmean":tmncmap, "pptC":pptcmapC, "tmeanC":tmncmapC, "LandCover":LCcmap}

def _annualtempmaker(va, cpath, funb =bn.nanmax, func="max",  box   = [-10.0, 180, 40, 70] ):
	#Function fo cumputing multi year means, sums and medians
	# breakpoint()
	fnout = f"{cpath}smoothed/TerraClim_{va}_meanannual{func}_1985to2015.nc"
	if os.path.isfile(fnout):
		ds = xr.open_dataset(fnout)
	else:
		with dask.config.set(**{'array.slicing.split_large_chunks': True}):
			with ProgressBar():
				ds = xr.open_mfdataset(
					f"{cpath}TerraClimate_tmean_*.nc", drop_variables=["station_influence", "crs"]).rename(
					{"lat":"latitude","lon":"longitude"}).sel(dict(
					time=slice('1985-01-01', '2015-12-31'), 
					latitude=slice(box[3], box[2]), 
					longitude=slice(box[0], box[1])
					))#.groupby('time.year').max().compute()	
				ds = ds.resample(time="Y").reduce(bn.nanmax).compute()
		ds = ds.mean(dim='time')
		attrs = GlobalAttributes(ds, va, fnout, func = "max", stdt = 1985, fndt = 2015)
		ds = tempNCmaker(ds, fnout, va)

	# breakpoint()
	return ds



def GlobalAttributes(ds, var, fnout, func = "max", stdt = 1985, fndt = 2015):
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
	attr["title"]               = f"{var} annual {func}"
	attr["summary"]             = f"Terraclimate {var} annual monthly {func} from {stdt} to {fndt}" 
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


def _locations():
	""" Fiunction to make a dataframe of different locations """
	Loc = OrderedDict()
	Loc[0] = {"Name":"Moscow",      "lat": 55.756303, "lon":37.616777,  "latoffset":-1.0, "lonoffset":2.0}
	Loc[1] = {"Name":"Chita",       "lat": 52.051446, "lon":113.471214, "latoffset":-1.0, "lonoffset":0.5}
	Loc[2] = {"Name":"Krasnoyarsk", "lat": 56.015283, "lon":92.893248,  "latoffset":-1.0, "lonoffset":0.5}
	Loc[3] = {"Name":"Irkutsk",     "lat": 52.286286, "lon":104.295314, "latoffset":-1.5, "lonoffset":0.0}
	Loc[4] = {"Name":"Yakutsk",     "lat": 62.039691, "lon":129.742219, "latoffset":-1.0, "lonoffset":0.5}
	Loc[5] = {"Name":"Chelyabinsk", "lat": 55.164521, "lon":61.437660,  "latoffset":-1.0, "lonoffset":1.0}
	Loc[6] = {"Name":"Vladivostok", "lat": 43.133297, "lon":131.911234, "latoffset":-0.25,   "lonoffset":-7.0}
	# Loc[7] = {"Name":"Stockholm",   "lat": 18.068581, "lon": 59.329324, "latoffset":-1.0, "lonoffset":2.0}
	# Loc[] = {"Name":"", "lat":, "lon":, "latoffset":-1.0, "lonoffset":0.5}
	# Loc[] = {"Name":"", "lat":, "lon":, "latoffset":-1.0, "lonoffset":0.5}
	# Loc[] = {"Name":"", "lat":, "lon":, "latoffset":-1.0, "lonoffset":0.5}
	return pd.DataFrame(Loc).T

def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	backpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "/mnt/d/Data51"
		cpath = "/mnt/g/Data51/Climate/TerraClimate/"
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
		dpath = "/media/ubuntu/Harbinger/Data51"
	# elif 'ccrc.unsw.edu.au' in sysname:
	# 	dpath = "/srv/ccrc/data51/z3466821"
	elif sysname == 'DESKTOP-N9QFN7K':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		cpath = "/mnt/f/Data51/Climate/TerraClimate/"
	elif sysname == 'DESKTOP-KMJEPJ8':
		dpath = "./data"
		# backpath = "/mnt/g/fireflies"
		cpath = "/mnt/i/Data51/Climate/TerraClimate/"
	elif sysname == 'arden-worstation':
		# WHRC linux distro
		dpath = "./data"
		cpath= "/media/arden/SeagateMassStorage/Data51/Climate/TerraClimate/"
		# dpath= "/media/arden/Harbinger/Data51/BurntArea"
	elif sysname == 'LAPTOP-8C4IGM68':
		dpath     = "./data"
		# backpath = "/mnt/d/fireflies"
		cpath = "/mnt/d/Data51/Climate/TerraClimate/"
	else:
		ipdb.set_trace()
	
	return dpath, cpath

# ==============================================================================

if __name__ == '__main__':
	main()