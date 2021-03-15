"""
Script goal, 

Predict future FRI based on current climate

"""
#==============================================================================

__title__ = "FRI Prediction status"
__author__ = "Arden Burrell"
__version__ = "v1.0(04.03.2021)"
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
from statsmodels.stats.weightstats import DescrStatsW
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
import cartopy as ct
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string


# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 


# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)
print("cartopy version : ", ct.__version__)

#==============================================================================

def main():

	dpath, cpath= syspath()
	tmpath = "./results/ProjectSentinal/FRImodeling/"

	# ========== Setup the params ==========
	TCF = 10
	mwbox   = [1]#, 2]#, 5]
	dsnams1 = ["esacci", "GFED", "COPERN_BA", "MODIS",  ]#  "GFED", ]# 
	dsinfo = datasets(dpath, dsnams1)
	# breakpoint()
	altnames = ({"GFED":"GFED4", "MODIS":"MCD64A1", "esacci":"FireCCI51", "COPERN_BA":"CGLS-BA", "HANSEN_AFmask":"HansenGFC-AFM", "HANSEN":"HansenGFC"}) 
	# scale = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20})
	BFmin = 0.0001
	DrpNF = True # False
	sub   = 1 #subsampling interval in deg lat and long
	transform = "QFT" #None 
	sens  =  [30, 60, 100]
	version = 0
	va = "AnBF"
	# ========== Setup the plot dir ==========
	plotdir = "./plots/ShortPaper/PF05_PredStats/"
	cf.pymkdir(plotdir)
	formats = [".png"]#, ".pdf"]

	sigmask = True
	fns = []
	for model in ["XGBoost", "OLS"]:
		for dsn in dsnams1:
			fnx = netcdfloader(dsn, model, dpath, cpath, plotdir, va, 
				tmpath, sub, sens, formats, sigmask, altnames, dsinfo, fmode="trend",
				 version=0, force = False, incTCfut=True)
			fns += fnx
		breakpoint()


def netcdfloader(
	dsn, model, dpath, cpath, plotdir, va, tmpath, 
	sub, sens, formats, sigmask, altnames, dsinfo, 
	fmode="trend", version=0, force = False, incTCfut=True,
	DrpNF=True, var ="FRI", bounds = [-10.0, 180.0, 70.0, 40.0], griddir = "./data/gridarea/"):

	# ========== Covert to dataset and save the results ==========
	fnames = []
	yrw    = []
	trnd   = []
	for sen in sens:
		fnout = f"{tmpath}S03_FRIdrivers_{dsn}_v{version}_{sen}yr_{fmode}Prediction_{'forests' if DrpNF else 'nomask'}_{'sigclim' if sigmask else ''}.nc"
		if os.path.isfile(fnout):
			fnames.append(fnout)
			yrw.append(pd.Timestamp(f"{2015 + sen}-12-31"))
			trnd.append(False)
		else:
			print(f"{dsn} {sen}yr file is missing")
		if sen == 100 and incTCfut:
			fnt = f"{tmpath}S03_FRIdrivers_{dsn}_v{version}_{sen}yr_TCfutPrediction_{'forests' if DrpNF else 'nomask'}.nc"
			if os.path.isfile(fnt):
				fnames.append(fnt)
				yrw.append(pd.Timestamp(f"{2015 + sen}-12-31 12:00:00"))
				trnd.append(True)
			else:
				print(f"{dsn} {sen}yr TCpred file is missing")

	# ========== Load the mask and grid area ==========
	print("Loading the grid", pd.Timestamp.now())
	gafn   = f"{griddir}{dsn}_gridarea.nc"
	if not os.path.isfile(gafn):
		subp.call(f"cdo gridarea {datasets[dsn]} {gafn}", shell=True)
	ds_ga = xr.open_dataset(gafn).sortby("latitude", ascending=False)#.astype(np.float32)
	with ProgressBar():	
		ds_ga = ds_ga.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0))).compute()
	ds_ga["cell_area"] *= 1e-6 # Convert from sq m to sq km
	if ds_ga["cell_area"].sum() == 0:
		print("Grid area failed, trying alternate method that is a bit slower")
		del ds_ga
		ds_ga = _gridcal (datasets, dsn, ds_dsn, gafn, var)


	print("Loading the mask", pd.Timestamp.now())
	mask = masker(dsn, dpath,  region="SIBERIA")
	# breakpoint()
	
	# ========== bring in the datasets ==========
	# dax = []
	statlist = []
	for fnout, tm, tskip  in zip(fnames, yrw, trnd):
		print(f"Starting the load for {dsn} {tm} prediction at: {pd.Timestamp.now()}")
		ds = xr.open_dataset(fnout).sortby("latitude", ascending=False).sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
		for tp in ["obs", "cur", "fut"]:
			if not (tm == pd.Timestamp('2045-12-31 00:00:00')): 
				if not tp =="fut":
					# Skip current predictions for TCpred datasets as they use the same model as tthe trend ones
					continue
			
			if tp =="obs":
				dsva = f"{va}"
				# breakpoint()
			else:
				dsva = f"{va}_{model}_{tp}"
			da = 1/ds[dsva].rename("FRI").squeeze(dim="time", drop=True).compute()#.coarsen()
			da *= mask
			# breakpoint()
			print(f"starting stat calculation for {dsn} {tm} {tp}")
			statlist.append(stamaker(da, ds_ga, tm, dsn, altnames, model, tskip, tp))

	keystats = pd.DataFrame(statlist)
	print(keystats)

	# ========== save the info out ==========
	# ========== Create the Metadata ==========
	try:
		Scriptinfo = "File saved from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, str(pd.Timestamp.now()))
		gitinfo = pf.gitmetadata()

		outpath = plotdir+"stats/"
		cf.pymkdir(outpath) 
		cf.writemetadata(outpath+f"PF05_{var}stats_{dsn}_{model}", [Scriptinfo, gitinfo])
		keystats.to_csv(outpath+f"PF05_{var}stats_{dsn}_{model}.csv")
	except Exception as e:
		print(str(e))
		breakpoint()
	
	# breakpoint()

	return statlist

#==============================================================================
def stamaker(frame, ds_ga, tm, dsn, altnames, model,  tskip, tp, var ="FRI"):
	# ========== Calculate the stats ==========
	stats = OrderedDict()
	stats["Dataset"] = altnames[dsn]
	if tp in ['cur', "obs"]:
		stats["date"] = pd.Timestamp('2015-12-31 00:00:00')
	else:
		stats["date"]    = tm
	
	if tp == "obs":
		stats["Method"]  = "Observed"
	else:
		stats["Method"]  = model
	
	if tskip:
		stats["ClimateData"] = "Tcpred"
	else:
		stats["ClimateData"] = "Trend"


	# ========== create the weights ==========
	weights        = np.cos(np.deg2rad(frame.latitude))
	weights.name   = "weights"
	
	# ========== calculate the number of nans and the number of  ==========
	# (frame.isnull()).weighted(weights).sum() / (~frame.isnull()).weighted(weights).sum()
	NN = ((~frame.isnull()).weighted(weights).sum()).values
	NA = ((frame.isnull()).weighted(weights).sum()).values
	stats["NonNan"] = NN / (NN+NA)
	stats["NonNansqkm"] = ((~frame.isnull().values) * ds_ga["cell_area"]).sum().values
	# breakpoint()

	# ========== Mask ouside the range ==========
	if var =="FRI":
		stats["OutRgnFrac"] = ((frame>10000.).weighted(weights).sum() / NN).values
		stats["OutRgnsqkm"] = ((frame>10000.).values * ds_ga["cell_area"]).sum().values

		# ========== Mask ouside the range ==========
		if tp == "obs":
			frame = frame.where(~(frame>10000.).values, 10001)
	elif var == "AnBF":
		stats["OutRgnFrac"] = ((frame<0.0001).weighted(weights).sum() / NN).values
		frame = frame.where(frame>0.0001)
	
	# ========== Use statsmodels to calculate the key statistis ==========
	# breakpoint()
	try:
		d1    = DescrStatsW(frame.values[~frame.isnull()], weights=ds_ga["cell_area"].values[~frame.isnull()])
	except Exception as err:
		print(str(err))
		breakpoint()
	stats[f"Mean{var}"] = d1.mean
	stats[f"std{var}"]  = d1.std
	if var =="FRI":
		stats["FRIsub15"] =  ((frame  < 15).weighted(weights).sum()/NN).values
		stats["FRIsub30"] =  (((frame < 30).weighted(weights).sum()/NN) - stats["FRIsub15"]).values
		stats["FRIsub60"] =  (((frame < 60).weighted(weights).sum()/NN) - (stats["FRIsub15"]+stats["FRIsub30"])).values
		stats["FRIsub15sqkm"] =  ((frame  < 15).values * ds_ga["cell_area"]).sum().values
		stats["FRIsub30sqkm"] =  (((frame < 30).values * ds_ga["cell_area"]).sum() - stats["FRIsub15sqkm"]).values
		stats["FRIsub60sqkm"] =  (((frame < 60).values * ds_ga["cell_area"]).sum() - (stats["FRIsub15sqkm"]+stats["FRIsub30sqkm"])).values
	# ========== Do the weighted quantiles ==========
	cquants = [0.001, 0.01,  0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
	quant   = d1.quantile(cquants)
	for cq in cquants:
		stats[f"{cq*100}percentile"] = quant[cq]
	# del frame, ds_ga
	# print(pd.Series(stats))
	# breakpoint()
	return pd.Series(stats)


def masker(dsn, dpath, region="SIBERIA"):
	stpath = dpath + "/masks/broad/"

	if not dsn.startswith("H"):
		fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsn)
	else:
		fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedToesacci.nc" % (region)

	# +++++ Check if the mask exists yet +++++
	if os.path.isfile(fnmask):
		with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask:
			
			msk    = dsmask.mask.isel(time=0).astype("float32")
			
			# if proj == "polar" and not dsn == "GFED":
			# 	msk = msk.coarsen({"latitude":scale[dsn], "longitude":scale[dsn]}, boundary ="pad").mean()
			
			msk = msk.values

			# +++++ Change the boolean mask to NaNs +++++
			msk[msk == 0] = np.NAN
			
			print("Masking %s loaded at:" % dsn, pd.Timestamp.now())
			# +++++ mask the frame +++++
			return msk
	else:
		breakpoint()


def _gridcal (datasets, dsn, ds_dsn, gafn, var, degmin= 111250.8709452735):
	# ========== import the python verion of cdo ==========
	cdo   = Cdo()
	# ========== Remove old file ==========
	os.remove(gafn)

	# ========= calculate the area ==========
	print(f"Starting python CDO gridarea at: {pd.Timestamp.now()}")
	da = cdo.gridarea(input=datasets[dsn], returnXArray="cell_area")
	data = xr.Dataset({"cell_area":da}).chunk({"latitude":500})
	with ProgressBar():
		data.to_netcdf(
			gafn, format = 'NETCDF4',)

	del data
	data = xr.open_dataset(gafn).astype(np.float32).sortby("latitude", ascending=False)
	if data["cell_area"].sum() == 0:
		raise ValueError("grid cell_area == 0")
	# data.longitude.attrs = {"long_name":"longitude", "units":"degrees_east"}
	# data.latitude.attrs  = {"long_name":"latitude", "units":"degrees_north"}
	# weights = np.cos(np.deg2rad(data.latitude))
	# data   *=weights
	# equtpix = (degmin*np.diff(data.longitude.values)[0]) * (degmin*np.diff(data.longitude.values)[0])
	# data *= equtpix
	return data 

def datasets(dpath, dsnames, mwb=1):
	datasets = OrderedDict()
	for dsnm in dsnames:
		if dsnm.startswith("H"):
			# +++++ make a path +++++
			ppath = dpath + "/BurntArea/HANSEN/FRI/"
			fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, tcfs, mwb)
			# fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		else:
			# fname = "Hansen_GFC-2018-v1.6_regrided_esacci_FRI_%ddegMW_SIBERIA" % (mwb)
			ppath = dpath + "/BurntArea/%s/FRI/" %  dsnm
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
		# +++++ open the datasets +++++
		# ipdb.set_trace()
		datasets[dsnm] = ppath+fname
	return datasets
def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	backpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "/mnt/d/Data51"
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
		dpath = "/media/ubuntu/Harbinger/Data51"
	# elif 'ccrc.unsw.edu.au' in sysname:
	# 	dpath = "/srv/ccrc/data51/z3466821"
	elif sysname == 'DESKTOP-T77KK56':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		backpath = "/mnt/f/fireflies"
		chunksize = 500
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
		backpath = "/mnt/d/fireflies"
	elif sysname == 'DESKTOP-N9QFN7K':
		# spath = "/mnt/c/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath     = "./data"
		cpath = "/mnt/f/Data51/Climate/TerraClimate/"


	else:
		ipdb.set_trace()
	
	return dpath, cpath

#==============================================================================

if __name__ == '__main__':
	main()
