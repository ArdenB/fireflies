"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

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
	dsnams1 = ["esacci", "MODIS", "GFED", ]# 
	scale = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20})
	BFmin = 0.0001
	DrpNF = True # False
	sub   = 1 #subsampling interval in deg lat and long
	transform = "QFT" #None 
	sens  =  [30, 60, 100]
	version = 0
	va = "AnBF"
	# ========== Setup the plot dir ==========
	plotdir = "./plots/ShortPaper/PF04_Predictions/"
	cf.pymkdir(plotdir)
	formats = [".png", ".pdf"]

	for dsn in dsnams1:
		for sigmask in [True, False]:
			for model in ["XGBoost", "OLS"]:
				futurenetcdfloader(dsn, model, dpath, cpath, plotdir, va, 
					tmpath, sub, sens, scale, formats, sigmask, fmode="trend",
					 version=0, force = False)
		# breakpoint()




def futurenetcdfloader(dsn, model, dpath, cpath, plotdir, va, tmpath, sub, sens, scale, formats, sigmask, fmode="trend",
	 version=0, force = False, DrpNF=True, bounds = [-10.0, 180.0, 70.0, 40.0]):
	# ========== make the plot name ==========
	plotfname = f"{plotdir}PF04_FRIprediction_{dsn}_{model}" 
	if sigmask:
		plotfname += "_sigclim"	

	# ========== set the mpl rc params ==========
	font = {'weight' : 'bold'}
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# ========== Covert to dataset and save the results ==========
	dax = []
	dac = []
	for sen in sens:
		fnout = f"{tmpath}S03_FRIdrivers_{dsn}_v{version}_{sen}yr_{fmode}Prediction"
		if DrpNF:
			fnout += "_forests"
		else:
			fnout += "_nomask"
		
		if sigmask:
			# plotfname += "_sigclim"	
			fnout     += "_sigclim"	
		fnout += ".nc"

		# ========== Check if a file exists ==========
		if os.path.isfile(fnout):
			print(f"Starting the load for {dsn} {sen}yr prediction at: {pd.Timestamp.now()}")
			ds = xr.open_dataset(fnout)
			for tp in ["cur", "fut"]:
				da = ds[f"{va}_{model}_{tp}"].rename("FRI")#.coarsen()
				if scale[dsn] > 1:
					da = da.coarsen(
						{"latitude":scale[dsn], "longitude":scale[dsn]}, 
						boundary ="pad", keep_attrs=True).mean().compute().rename("FRI")
				if tp == "fut":
					da["time"] = [pd.Timestamp(f"{pd.Timestamp(da.time.values[0]).year + sen}-12-31")]
					# ========== Convert to FRI ==========
					dax.append(1.0/da)
				else:
					# ========== Convert to FRI ==========
					dac.append(1.0/da)
		else:
			print(f"{dsn} {sen}yr file is missing")
				# da["name"] = "FRI"
	# ========== Build a single file by taking the mean of the current, theen merging the rest ==========
	if len(dac) > 1:
		daM = xr.merge([xr.concat(dac, dim="ver").mean(dim="ver")] + dax)
	elif len(dac) == 0:
		print(f"{dsn} is missing all results, going to next dataset")
		return
	else:
		daM = xr.merge(dac + dax)

	# breakpoint()
	daM.FRI.attrs = {'long_name':"FRI", "units":"years"}


	print(f"Starting {dsn} load at: {pd.Timestamp.now()}")
	latiMid=np.mean([70.0, 40.0])
	longMid=np.mean([-10.0, 180.0])

		# fig, ax = plt.subplots(
		# 	1, 1, figsize=(20,12), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})
		
	cmap, norm, vmin, vmax, levels = _colours("FRI", 10000)
	# breakpoint()
	px = daM.FRI.plot(x="longitude", y="latitude", col="time", col_wrap=2,
		vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, size = 8, #extent=bounds,
		transform=ccrs.PlateCarree(),
		add_colorbar=False,
		subplot_kws={"projection": ccrs.Orthographic(longMid, latiMid)},
		cbar_kwargs={ "ticks":levels, "spacing":"uniform", "extend":"max","pad": 0.20,  "shrink":0.85}
		) #
	for ax, tm in zip(px.axes.flat, daM.time.values):
		ax.gridlines()
		coast = cpf.GSHHSFeature(scale="intermediate")
		ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
		ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
		ax.add_feature(coast, zorder=101, alpha=0.5)
		ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		ax.add_feature(cpf.RIVERS, zorder=104)		
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)	

		ax.set_title("")
		ax.set_title(f"{dsn} - {model} ({pd.Timestamp(tm).year - 30} to {pd.Timestamp(tm).year})", loc= 'left')
		if dsn == "esacci":
			# breakpoint()
			ax.set_extent(bounds, crs = ccrs.PlateCarree())
	
	plt.subplots_adjust(top=.99, bottom=0.01, left=0.009, right=0.991, hspace=0.0, wspace=0.02)
	
	# "constrained_layout":True
	# fig = plt.gcf()
	# breakpoint()

	px.add_colorbar(extend="max",pad= 0.015, shrink=0.65)

	if not (formats is None): 
		# ========== loop over the formats ==========
		for fmt in formats:
			print(f"starting {fmt} plot save at:{pd.Timestamp.now()}")
			plt.savefig(plotfname+fmt)#, dpi=dpi)
	print("Starting plot show at:", pd.Timestamp.now())

	plt.show()
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
		cmap.set_bad('dimgrey',1.)

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
	else:
		ipdb.set_trace()
	
	return dpath, cpath

#==============================================================================

if __name__ == '__main__':
	main()
