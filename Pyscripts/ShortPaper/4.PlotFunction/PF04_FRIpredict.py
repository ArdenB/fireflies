"""
Script goal, 

Predict future FRI based on current climate

"""
#==============================================================================

__title__ = "FRI Prediction"
__author__ = "Arden Burrell"
__version__ = "v1.0(27.11.2019)"
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
	dsnams1 = ["esacci", "COPERN_BA", "MODIS", "GFED"]#  "GFED", ]# 
	altnames = ({"GFED":"GFED4", "MODIS":"MCD64A1", "esacci":"FireCCI51", "COPERN_BA":"CGLS-BA", "HANSEN_AFmask":"HansenGFC-AFM", "HANSEN":"HansenGFC"}) 
	scale = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20})
	BFmin = 0.0001
	DrpNF = True # False
	sub   = 1 #subsampling interval in deg lat and long
	transform = "QFT" #None 
	sens  =  [30, 60, 100]
	version = 0
	va = "AnBF"
	maskver = "Boreal"
	# ========== Setup the plot dir ==========
	plotdir = "./plots/ShortPaper/PF04_Predictions/"
	cf.pymkdir(plotdir)
	formats = [".png", ".tiff", ".eps",]# ".pdf"]
	bounds = [10.0, 170.0, 70.0, 49.0] # plot bounds

	for sigmask in [True, False]:
		for model in ["XGBoost", "OLS"]:
			for dsn in dsnams1:
				futurenetcdfloader(dsn, model, dpath, cpath, plotdir, va, 
					tmpath, sub, sens, scale, formats, sigmask, altnames, maskver,  fmode="trend",
					 version=0, force = False, incTCfut=True, bounds=bounds)
			
		# breakpoint()

def futurenetcdfloader(dsn, model, dpath, cpath, plotdir, va, tmpath, sub, sens, scale, 
	formats, sigmask, altnames, maskver,  fmode="trend",
	version=0, force = False, DrpNF=True, bounds = [-10.0, 180.0, 70.0, 40.0], 
	xbounds = [-10.0, 180.0, 70.0, 40.0], incTCfut=False, areacal=False):
	# ========== make the plot name ==========
	plotfname = f"{plotdir}PF04_FRIprediction_{dsn}_{model}" 
	if sigmask:
		plotfname += "_sigclim"	

	# ========== set the mpl rc params ==========
	font = {'weight' : 'bold'}
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

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

	dax = []
	dac = []
	# ========== Calculate the mask ==========
	if maskver == "Boreal":
		print(f"Starting the load for {dsn} boreal mask at: {pd.Timestamp.now()}")
		fnBmask = f"./data/LandCover/Regridded_forestzone_{dsn}.nc"

		Bmask = xr.open_dataset(fnBmask).drop(["DinersteinRegions", "GlobalEcologicalZones", "LandCover"]).sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1]))).transpose('time', 'latitude', 'longitude')
		# Bmask["time"] = da.time
		msk = ((Bmask.BorealMask>0).astype("float32"))
		if scale[dsn] > 1:
			msk = msk.coarsen({"latitude":scale[dsn], "longitude":scale[dsn]}, boundary ="pad").median()
	# ========== Check if a file exists ==========
	for fnout, tm, tskip  in zip(fnames, yrw, trnd):
		print(f"Starting the load for {dsn} {tm} prediction at: {pd.Timestamp.now()}")
		ds = xr.open_dataset(fnout).sortby("latitude", ascending=False).sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
		for tp in ["cur", "fut"]:
			if tskip and tp == "cur":
				# Skip current predictions for TCpred datasets as they use the same model as tthe trend ones
				continue
			da = ds[f"{va}_{model}_{tp}"].rename("FRI")#.coarsen()
			
			if scale[dsn] > 1:
				da = da.coarsen({"latitude":scale[dsn], "longitude":scale[dsn]}, 
					boundary ="pad").mean().compute().rename("FRI")#.squeeze().squeeze()
					# {"latitude":scale[dsn], "longitude":scale[dsn]}, 

			# Add the mask 
			if maskver == "Boreal":
				# breakpoint()
				da  = da.where((msk == 1).values)

			if tp == "fut":
				da["time"] = [tm]#pd.Timestamp(f"{pd.Timestamp(da.time.values[0]).year + sen}-12-31")
				# ========== Convert to FRI ==========
				dax.append(1.0/da)
			else:
				# ========== Convert to FRI ==========
				dac.append(1.0/da)
			# da["name"] = "FRI"
	# ========== Build a single file by taking the mean of the current, theen merging the rest ==========
	if len(dac) > 1:
		daM = xr.merge([xr.concat(dac, dim="ver").mean(dim="ver")] + dax)
	elif len(dac) == 0:
		print(f"{dsn} is missing all results, going to next dataset")
		return
	else:
		daM = xr.merge(dac + dax)

	daM.FRI.attrs = {'long_name':"FRI", "units":"years"}


	print(f"Starting {dsn} plot at: {pd.Timestamp.now()}")
	latiMid=np.mean([bounds[2], bounds[3]])
	longMid=np.mean([bounds[0], bounds[1]])

		# fig, ax = plt.subplots(
		# 	1, 1, figsize=(20,12), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})
		
	cmap, norm, vmin, vmax, levels = _colours("FRI", 10000)
	# breakpoint()
	px = daM.FRI.plot(x="longitude", y="latitude", col="time", col_wrap=2,
		vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, size = 4,  aspect=2,
		subplot_kws={"projection": ccrs.Orthographic(longMid, latiMid)},
		transform=ccrs.PlateCarree(),
		add_colorbar=False,
		cbar_kwargs={ "ticks":levels, "spacing":"uniform", "extend":"max","pad": 0.20,  "shrink":0.85}
		) #
	# breakpoint()
	for ax, tm, alp in zip(px.axes.flat, daM.time.values, string.ascii_lowercase):
		ax.set_aspect('equal')
		ax.gridlines()
		gl = ax.gridlines(draw_labels= True, dms=True, x_inline=False, y_inline=False)#{"bottom": "x", "Top": "y"}
		gl.xlocator = mticker.FixedLocator([60, 120])
		gl.ylocator = mticker.FixedLocator([50, 60, 70])

		coast = cpf.GSHHSFeature(scale="intermediate")
		ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
		ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
		ax.add_feature(coast, zorder=101, alpha=0.5)
		ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		ax.add_feature(cpf.RIVERS, zorder=104)		
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=105)	

		ax.set_title("")
		istrend = (pd.Timestamp(tm).hour == 0)
		ax.set_title(f"{alp}) {altnames[dsn]} - {model} {'ObsTrend' if istrend else 'TCpred'} ({pd.Timestamp(tm).year - 30} to {pd.Timestamp(tm).year})", loc= 'left')
		ax.set_extent(bounds, crs = ccrs.PlateCarree())
		# if dsn == "esacci":
		# 	# breakpoint()
	
	plt.subplots_adjust(top=.99, bottom=0.01, left=0.02, right=0.991, hspace=0.04, wspace=0.08)
	
	# "constrained_layout":True
	# fig = plt.gcf()

	px.add_colorbar(extend="max",pad= 0.010, shrink=0.65)
	plt.draw()
	# plt.show()
	# breakpoint()

	if not (formats is None): 
		# ========== loop over the formats ==========
		for fmt in formats:
			print(f"starting {fmt} plot save at:{pd.Timestamp.now()}")
			plt.savefig(plotfname+fmt)#, dpi=dpi)
	print("Starting plot show at:", pd.Timestamp.now())

	plt.show()
	# plt.close()
	# breakpoint()



#==============================================================================
def _colours(var, vmax):
	norm=None
	levels = None
	if var == "FRI":
		# +++++ set the min and max values +++++
		vmin = 0.0
		# +++++ create hte colormap +++++
		if vmax in [80, 10000]:
			# cmapHex = palettable.matplotlib.Viridis_9_r.hex_colors
			cmapHex = palettable.colorbrewer.diverging.Spectral_9.hex_colors
			levels = [0, 15, 30, 60, 120, 500, 1000, 3000, 10000, 10001]
		else:
			cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors

		cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
		
		if vmax == 10000:
			norm   = mpl.colors.BoundaryNorm([0, 15, 30, 60, 120, 500, 1000, 3000, 10000], cmap.N)

		cmap.set_over(cmapHex[-1] )
		# cmap.set_bad('dimgrey',1.)

	else:
		# ========== Set the colors ==========
		vmin = 0.0
		vmax = 0.20

		# +++++ create the colormap +++++
		# cmapHex = palettable.matplotlib.Inferno_10.hex_colors
		# cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors
		cmapHex = palettable.colorbrewer.sequential.OrRd_9.hex_colors
		

		cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
		cmap.set_over(cmapHex[-1] )
		cmap.set_bad('dimgrey',1.)
	return cmap, norm, vmin, vmax, levels

def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	cpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "./data"
		# cpath = "/mnt/d/Data51/Climate/TerraClimate/"
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
