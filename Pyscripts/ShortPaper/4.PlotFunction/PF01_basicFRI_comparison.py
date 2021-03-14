"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

"""
#==============================================================================

__title__ = "FRI Comparison"
__author__ = "Arden Burrell"
__version__ = "v1.0(08.11.2019)"
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
	warn.warn("\n TO DO: add to the variaable attrs so the plots have units. Also implement a better downsampling")
	# ========== Setup the params ==========
	TCF     = 10
	mwbox   = [1]#, 2]#, 5]
	dsnams1 = ["GFED", "MODIS", "esacci"]#, "COPERN_BA"]#, "HANSEN_AFmask", "HANSEN"]
	dsnams2 = ["HANSEN_AFmask", "HANSEN"]
	scale   = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20})
	dsts    = [dsnams1, dsnams2]
	dsinfo  = dsinfomaker()
	proj    = "polar"
	# proj = "latlon"
	# vmax    = 120
	# vmax    = 80
	# vmax    = 100
	for var in ["FRI"]:#, "AnBF"]:
		for dsnames, vmax in zip(dsts, [10000, 10000]):
			formats = [".png"]#, ".pdf"] # None 
			# mask    = True
			if TCF == 0:
				tcfs = ""
			else:
				tcfs = "_%dperTC" % np.round(TCF)


			# ========== Setup the plot dir ==========
			plotdir = "./plots/ShortPaper/PF01_FRI/"
			cf.pymkdir(plotdir)
			# compath = "/media/ubuntu/Seagate Backup Plus Drive"
			compath, backpath = syspath()

			for mwb in mwbox:
				# ========== Setup the dataset ==========
				datasets = OrderedDict()
				for dsnm in dsnames:
					if dsnm.startswith("H"):
						# +++++ make a path +++++
						ppath = compath + "/BurntArea/HANSEN/FRI/"
						fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, tcfs, mwb)
						# fname = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
					else:
						# fname = "Hansen_GFC-2018-v1.6_regrided_esacci_FRI_%ddegMW_SIBERIA" % (mwb)
						ppath = compath + "/BurntArea/%s/FRI/" %  dsnm
						fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
					# +++++ open the datasets +++++
					# ipdb.set_trace()
					datasets[dsnm] = ppath+fname #xr.open_dataset(ppath+fname)
					# ipdb.set_trace()
				
				for mask, bounds in zip([True, False], [[15.0, 170.0, 70.0, 49.0], [-10.0, 180.0, 70.0, 40.0]]):
					# testplotmaker(datasets, var, mwb, plotdir, formats, mask, compath, vmax, backpath, proj, scale)
					plotmaker(dsinfo, datasets, var, mwb, plotdir, formats, mask, compath, vmax, backpath, proj, scale, bounds)
					breakpoint()

				# ipdb.set_trace()

#==============================================================================
def testplotmaker(dsinfo, datasets, var, mwb, plotdir, formats, mask, compath, vmax, backpath, proj, scale, region = "SIBERIA"):
	# ========== Setup the font ==========
	font = {'weight' : 'bold', #,
			# 'size'   : mapdet.latsize
			}

	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", }) #'axes.titlesize':mapdet.latsize
		
	# ========== setup the figure ==========
	if proj == "polar":
		latiMid=np.mean([70.0, 40.0])
		longMid=np.mean([-10.0, 180.0])
		for dsn in datasets:
			print(f"{dsn} start at: {pd.Timestamp.now()}")
			fig, ax = plt.subplots(
				1, 1, figsize=(20,12), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})
			
			frame = _fileopen(datasets, dsn, var, scale, proj, mask, compath, region)
			# ========== Set the colors ==========
			cmap, norm, vmin, vmax, levels = _colours(var, vmax, )
			
			# ========== Creat the plot ==========
			im = frame.compute().plot(
				ax=ax, vmin=vmin, vmax=vmax, 
				cmap=cmap, norm=norm, #add_colorbar=False,
				transform=ccrs.PlateCarree(),
				cbar_kwargs={"pad": 0.02, "extend":"max", "shrink":0.97, "ticks":levels, "spacing":"uniform"}
				) #
				# subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)}
			ax.gridlines()
			coast = cpf.GSHHSFeature(scale="intermediate") #"high"
			ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
			ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
			ax.add_feature(coast, zorder=101, alpha=0.5)
			ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
			ax.add_feature(cpf.RIVERS, zorder=104)
			print(f"Starting testplot show for {dsn} at:{pd.Timestamp.now()}")
			plt.show()
			breakpoint()




def plotmaker(dsinfo, datasets, var, mwb, plotdir, formats, mask, compath, vmax, backpath, proj, scale, bounds):
	"""Function builds a basic stack of maps """

	# ========== make the plot name ==========
	plotfname = plotdir + "PF01_%s_MW_%02dDegBox_V2_%s_%s" % (var, mwb, proj, "_".join(datasets.keys()))
	if mask:
		plotfname += "_ForestMask_V2"

	# ========== Setup the font ==========
	# ========== set the mpl rc params ==========
	font = {'weight' : 'bold'}
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# mpl.rc('font', **font)
	# plt.rcParams.update({'axes.titleweight':"bold", }) #'axes.titlesize':mapdet.latsize
		
	# ========== setup the figure ==========
	if proj == "polar":
		latiMid=np.mean([bounds[2], bounds[3]])
		longMid=np.mean([bounds[0], bounds[1]])
		if len(datasets) == 4:
			yv = 2
			xv = 2
			shrink=0.80
		else:
			yv = len(datasets)
			xv = 1
			shrink=0.95
		fig, axs = plt.subplots(
			yv, xv, figsize=(20,12), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})
	else:
		latiMid=np.mean([bounds[2], bounds[3]])
		longMid=np.mean([bounds[0], bounds[1]])
		fig, axs = plt.subplots(
			len(datasets), 1, sharex=True, 
			figsize=(16,9), subplot_kw={'projection': ccrs.PlateCarree()})
		shrink = None
	# bounds = [-10.0, 180.0, 70.0, 40.0]
	# breakpoint()

	# ========== Loop over the figure ==========
	for (num, ax), dsn, in zip(enumerate(axs.flat), datasets):
		# make the figure
		im = _subplotmaker(dsinfo, num, ax, var, dsn, datasets, mask, compath, backpath, proj, scale, bounds, latiMid, longMid, vmax = vmax, shrink=shrink)
		# breakpoint()
		ax.set_aspect('equal')

	# ========== Make the final figure adjusments ==========
	# +++++ Get rid of the excess lats +++++
	if not proj == "polar":
		for ax in axs.flat:
			ax.label_outer()
		if vmax == 10000:
			# +++++ Add a single colorbar +++++
			levels = [0, 15, 30, 60, 120, 500, 1000, 3000, 10000, 10001]
			fig.colorbar(im, ax=axs.ravel().tolist(), extend="max", ticks = levels, spacing = "uniform")
		else:
			fig.colorbar(im, ax=axs.ravel().tolist(), extend="max")
	
	# ========== Change parms for the entire plot =========
	# plt.axis('scaled')
	if len (datasets) == 4:
		plt.subplots_adjust(top=0.99,bottom=0.010, left=0.010, right=0.97, hspace=0.00,wspace=0.0)
	else:
		plt.subplots_adjust(top=0.971,bottom=0.013,left=0.008,right=0.993,hspace=0.063,wspace=0.0)

	# print("Starting plot show at:", pd.Timestamp.now())
	# plt.show()
	# sys.exit()

	if not (formats is None): 
		# ========== loop over the formats ==========
		for fmt in formats:
			print(f"starting {fmt} plot save at:{pd.Timestamp.now()}")
			plt.savefig(plotfname+fmt)#, dpi=dpi)
	print("Starting plot show at:", pd.Timestamp.now())
	
	plt.show()
	if not (plotfname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotfname, gitinfo]
		cf.writemetadata(plotfname, infomation)

#==============================================================================
def dsinfomaker(SR="SR"):
	"""
	Contains infomation about the Different datasets
	"""
	dsinfo = OrderedDict()
	# ==========
	dsinfo["GFED"]          = ({"alias":"GFED4.1","long_name":"FRI", "units":"yrs"})
	dsinfo["MODIS"]         = ({"alias":"MCD64A1", "long_name":"FRI","units":"yrs", "version":"v006"})
	dsinfo["esacci"]        = ({"alias":"FireCCI5.1", "long_name":"FRI","units":"yrs"})
	dsinfo["COPERN_BA"]     = ({"alias":"CGLS", "long_name":"FRI","units":"yrs"})
	dsinfo["HANSEN_AFmask"] = ({"alias":"Hansen GFC & MCD14ML", "long_name":f'FRI$_{{{SR}}}$',"units":"yrs"})
	dsinfo["HANSEN"]        = ({"alias":"Hansen GFC", "long_name":"DRI","units":"yrs"})


	return dsinfo


def _subplotmaker(dsinfo, num, ax, var, dsn, datasets, mask,compath, backpath, proj,scale, bounds, latiMid, longMid, region = "SIBERIA", vmax = 80.0,shrink=0.85):
	"""
	Funstion to build subplots
	"""
	# ========== open the dataset ==========
	if not os.path.isfile(datasets[dsn]):
		# The file is not in the folder
		warn.warn(f"File {datasets[dsn]} could not be found")
		breakpoint()
	else:
		frame = _fileopen(dsinfo, datasets, dsn, var, scale, proj, mask, compath, region, bounds)

	
	# ========== Set the colors ==========
	cmap, norm, vmin, vmax, levels = _colours(var, vmax, dsn)


	# ========== Grab the data ==========
	if proj == "polar":
		# .imshow
		# breakpoint()

		im = frame.compute().plot(
			ax=ax, vmin=vmin, vmax=vmax, 
			cmap=cmap, norm=norm, 
			transform=ccrs.PlateCarree(),
			# add_colorbar=False,
			cbar_kwargs={"pad": 0.02, "extend":"max", "shrink":shrink, "ticks":levels, "spacing":"uniform"}
			) #
			# subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)}
		ax.set_extent(bounds, crs=ccrs.PlateCarree())
		ax.gridlines()
		# +++++ get rid of the excess lables +++++
		# gl.xlabels_top = False
		# gl.ylabels_right = False
		# if not dsn == [dss for dss in datasets][-1]:
			# Get rid of lables in the middle of the subplot
			# gl.xlabels_bottom = False
			# ax.axes.xaxis.set_ticklabels([])
		# ax.set_extent(bounds, crs=ccrs.Orthographic(longMid, latiMid))
	else:
		im = frame.plot.imshow(
			ax=ax, extent=bounds, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, 
			transform=ccrs.PlateCarree(), 
			add_colorbar=False,
			) #

		ax.set_extent(bounds, crs=ccrs.PlateCarree())
		# =========== Set up the gridlines ==========
		gl = ax.gridlines(
			crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, 
			linestyle='--', zorder=105)

		# +++++ get rid of the excess lables +++++
		gl.xlabels_top = False
		gl.ylabels_right = False
		if not dsn == [dss for dss in datasets][-1]:
			# Get rid of lables in the middle of the subplot
			gl.xlabels_bottom = False
			# ax.axes.xaxis.set_ticklabels([])


		gl.xlocator = mticker.FixedLocator(np.arange(bounds[0], bounds[1]+10.0, 20.0)+10)
		gl.ylocator = mticker.FixedLocator(np.arange(bounds[2], bounds[3]-10.0, -10.0))
		
		gl.xformatter = LONGITUDE_FORMATTER
		gl.yformatter = LATITUDE_FORMATTER
		ax.outline_patch.set_visible(False)


	# ========== Add features to the map ==========
	coast_50m = cpf.GSHHSFeature(scale="high")
	ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(coast_50m, zorder=101, alpha=0.5)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)


	# =========== Setup the subplot title ===========
	ax.set_title(f"{string.ascii_lowercase[num]}) {dsinfo[dsn]['alias']}", loc= 'left')
	# plt.show()
	# sys.exit()
	return im
#==============================================================================
def _fileopen(dsinfo, datasets, dsn, var, scale, proj, mask, compath, region, bounds):
	ds_dsn = xr.open_dataset(datasets[dsn])
	# xbounds [-10.0, 180.0, 70.0, 40.0]
	xbounds = [-10.0, 180.0, 70.0, 40.0]
	# ========== Get the data for the frame ==========
	frame = ds_dsn[var].isel(time=0).sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1]))).drop("time")
	
	if proj == "polar" and not dsn == "GFED":
		# ========== Coarsen to make plotting easier =========
		frame = frame.coarsen(
			{"latitude":scale[dsn], "longitude":scale[dsn]
			}, boundary ="pad", keep_attrs=True).mean().compute()

	
	frame.attrs = dsinfo[dsn]#{'long_name':"FRI", "units":"years"}

	# ========== mask ==========
	if mask:
		# +++++ Setup the paths +++++
		# stpath = compath +"/Data51/ForestExtent/%s/" % dsn
		stpath = compath + "/masks/broad/"

		if not dsn.startswith("H"):
			fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsn)
			fnBmask = f"./data/LandCover/Regridded_forestzone_{dsn}.nc"
		else:
			fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedToesacci.nc" % (region)
			fnBmask = f"./data/LandCover/Regridded_forestzone_esacci.nc"

		# +++++ Check if the mask exists yet +++++
		if os.path.isfile(fnmask):
			with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask, xr.open_dataset(fnBmask).drop(["DinersteinRegions", "GlobalEcologicalZones", "LandCover"]) as Bmask:
				# breakpoint()
				msk    = (dsmask.mask.isel(time=0)*((Bmask.BorealMask.isel(time=0)>0).astype("float32")))#.sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
				
				if proj == "polar" and not dsn == "GFED":
					msk = msk.coarsen({"latitude":scale[dsn], "longitude":scale[dsn]}, boundary ="pad").median()
				
				msk = msk.values

				# +++++ Change the boolean mask to NaNs +++++
				msk[msk == 0] = np.NAN
				
				print("Masking %s frame at:" % dsn, pd.Timestamp.now())
				# +++++ mask the frame +++++
				# breakpoint()
				frame *= msk

				# +++++ close the mask +++++
				msk = None
				print(f"masking complete, begining ploting at {pd.Timestamp.now()}")


		else:
			print("No mask exists for ", dsn)
			breakpoint()
	# breakpoint()
	return frame

def _colours(var, vmax, dsn):
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
		# dpath = "./data"
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
		backpath = "/mnt/g/fireflies"
		chunksize = 500
	elif sysname == 'arden-Precision-5820-Tower-X-Series':
		# WHRC linux distro
		dpath = "./data"
		breakpoint()
		# dpath= "/media/arden/Harbinger/Data51/BurntArea"
	elif sysname == 'LAPTOP-8C4IGM68':
		dpath     = "./data"
		backpath = "/mnt/d/fireflies"
	else:
		ipdb.set_trace()
	return dpath, backpath

#==============================================================================

if __name__ == '__main__':
	main()