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
import cartopy as ct
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from matplotlib.colors import LogNorm, Normalize


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
	# ========== Setup the params ==========
	TCF     = 10
	mwbox   = [1]#, 2]#, 5]
	dsnams1 = ["GFED", "MODIS", "esacci", "COPERN_BA"]#, "HANSEN_AFmask", "HANSEN"]
	dsnams2 = ["HANSEN_AFmask", "HANSEN", "SRfrac"]
	dsnams3 = ["Risk"]
	scale   = ({"GFED":1, "MODIS":10, "esacci":20, "COPERN_BA":15, "HANSEN_AFmask":20, "HANSEN":20, "Risk":20, "SRfrac":20})
	dsts    = [dsnams3, dsnams2, dsnams1]
	proj    = "polar"
	maskver = "Boreal"	
	for var in ["FRI"]:#, "AnBF"]:
		for dsnames, vmax in zip(dsts, [10000, 10000, 10000]):
			formats = [".png"]#, ".pdf"] # None 
			# mask    = True
			if TCF == 0:
				tcfs = ""
			else:
				tcfs = "_%dperTC" % np.round(TCF)


			# ========== Setup the plot dir ==========
			plotdir = "./plots/ShortPaper/PF01_FRI/"
			cf.pymkdir(plotdir)
			compath, backpath = syspath()
			# compath = "/media/ubuntu/Seagate Backup Plus Drive"

			for mwb in mwbox:
				dsinfo  = dsinfomaker(compath, backpath, mwb, tcfs)
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
					cf.pymkdir(ppath)
					datasets[dsnm] = ppath+fname #xr.open_dataset(ppath+fname)
					# ipdb.set_trace()
				
				mask = True
				bounds = [10.0, 170.0, 70.0, 49.0]
				plotmaker(dsinfo, datasets, var, mwb, plotdir, formats, mask, compath, vmax, backpath, proj, scale, bounds, maskver)
				# for mask, bounds in zip([True, False], [[10.0, 170.0, 70.0, 49.0], [-10.0, 180.0, 70.0, 40.0]]):
				# 	# testplotmaker(datasets, var, mwb, plotdir, formats, mask, compath, vmax, backpath, proj, scale)
				# 	breakpoint() 

				# ipdb.set_trace()

#==============================================================================

def plotmaker(dsinfo, datasets, var, mwb, plotdir, formats, mask, compath, vmax, backpath, proj, scale, bounds, maskver):
	"""Function builds a basic stack of maps """

	# ========== make the plot name ==========
	plotfname = plotdir + "PF01_%s_MW_%02dDegBox_V2_%s_%s" % (var, mwb, proj, "_".join(datasets.keys()))
	if mask:
		plotfname += "_ForestMask_V2"

	# ========== Setup the font ==========
	# ========== set the mpl rc params ==========
	font = ({
		'weight' : 'bold',
		'size'   : 11, 
		})
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# mpl.rc('font', **font)
	# plt.rcParams.update({'axes.titleweight':"bold", }) #'axes.titlesize':mapdet.latsize
		
	# ========== setup the figure ==========
	if proj == "polar":
		latiMid=np.mean([bounds[2], bounds[3]])
		longMid=np.mean([bounds[0], bounds[1]])
		# if len(datasets) == 4:
		# 	yv = 2
		# 	xv = 2
		# 	shrink=0.80
		# else:
		yv = len(datasets)
		xv = 1
		shrink=0.95

		fig, axs = plt.subplots(
			yv, xv, figsize=(12,5*len(datasets)), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})
	else:
		latiMid=np.mean([bounds[2], bounds[3]])
		longMid=np.mean([bounds[0], bounds[1]])
		fig, axs = plt.subplots(
			len(datasets), 1, sharex=True, 
			figsize=(16,9), subplot_kw={'projection': ccrs.PlateCarree()})
		shrink = None
	# bounds = [-10.0, 180.0, 70.0, 40.0]

	# ========== Loop over the figure ==========
	if len(datasets) == 1:
		enax = [axs]
	else:
		enax = axs.flat
	
	for num, (ax, dsn) in enumerate(zip(enax, datasets)):
		# make the figure
		im = _subplotmaker(dsinfo, num, ax, var, dsn, datasets, mask, compath, backpath, proj, scale, bounds, latiMid, longMid, maskver, vmax = vmax, shrink=shrink)

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
	elif len (datasets) == 1:
		plt.subplots_adjust(top=0.971,bottom=0.013,left=0.008,right=0.98,hspace=0.063,wspace=0.0)
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
def _SRfracBuilder(dsinfo, num, ax, var, dsn, datasets, mask,compath, backpath, 
	proj,scale, bounds, latiMid, longMid, maskver, region = "SIBERIA", 
	vmax = 80.0,shrink=0.8, xbounds = [-10.0, 180.0, 70.0, 40.0]): 

	# ========== Riskbuilder ==========
	ds_dsn = xr.open_dataset(dsinfo["esacci"]["fname"])
	# breakpoint()

	# ========== Get the data for the frame ==========
	frame = ds_dsn["AnBF"].sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
	
	# frame = None
	ds_dsn2 = xr.open_dataset(dsinfo["HANSEN_AFmask"]["fname"])
	frame2 = ds_dsn2["AnBF"].sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
	
	with ProgressBar():
		frm = (frame2/frame).compute()
	frm = frm.where(~(frm >1), 1)

	ds_SRF = xr.Dataset({"StandReplacingFireFraction":frm})
	GlobalAttributes(ds_SRF, fnameout=datasets[dsn])
	ds_SRF.attrs["title"] = "StandReplacingFireFraction"
	ds_SRF.attrs["summary"] = "StandReplacingFireFraction esacci and HANSEN_AFmask"
	ds_SRF.to_netcdf(datasets[dsn], format = 'NETCDF4', unlimited_dims = ["time"])
	print("FRI SR frac Dataset Built")






def _RiskBuilder(dsinfo, num, ax, var, dsn, datasets, mask,compath, backpath, 
	proj,scale, bounds, latiMid, longMid, maskver, region = "SIBERIA", 
	vmax = 80.0,shrink=0.8, xbounds = [-10.0, 180.0, 70.0, 40.0]): 

	# ========== Riskbuilder ==========
	ds_dsn = xr.open_dataset(dsinfo["esacci"]["fname"])
	# ========== Get the data for the frame ==========
	frame = ds_dsn["FRI"].sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
	
	FRI15 = (frame <=15).astype("int16")
	FRI30 = (frame <=30).astype("int16")
	frame = None


	# ========== Fetch the  FRIsr ==========
	ds_SRI = xr.open_dataset(dsinfo["HANSEN_AFmask"]["fname"])
	SR_da  = ds_SRI["FRI"].sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
	SRI60  = (SR_da <= 60).astype("int16")
	SRI120 = (SR_da <=120).astype("int16")
	SR_da  = None

	# ========= Workout my fire risk catogeries ==========
	HR = (SRI120 * FRI30) # High Risk Fire
	CR = np.logical_or(FRI15, SRI60).astype("int16").where(HR == 1, 0)
	MR = np.logical_or(FRI30, SRI120).astype("int16")
	
	# +++++ Cleanup +++++
	SRI60  = None
	SRI120 = None
	ds_SRI = None
	# ========= Workout my Dist risk catogeries ==========

	ds_DRI = xr.open_dataset(dsinfo["HANSEN"]["fname"])
	DR_da  = ds_DRI["FRI"].sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
	DRI60  = (DR_da <= 60).astype("int16")
	DRI120 = (DR_da <=120).astype("int16")
	DR_da  = None

	HRD = np.logical_or(HR, (DRI120 * FRI30)) # High Risk all
	CRD = np.logical_or(FRI15, DRI60).astype("int16").where(HRD == 1, 0)
	# CRD = np.logical_or(FRI15, SRI60).astype("int16").where(HR == 1, 0)
	MRD = np.logical_or(MR,(np.logical_or(FRI30, DRI120))).astype("int16")
	FRI15  = None
	DRI60  = None
	DRI120 = None

	def _quickplot (da, scale, dsn):
		
		with ProgressBar():
			dac = da.coarsen(
				{"latitude":scale[dsn]*2, "longitude":scale[dsn]*2
				}, boundary ="pad").max().compute()
		dac.plot(vmin=1)
		plt.show()
	Risk = CRD+MRD+HRD+MR+HR+CR
	# _quickplot((Risk==1).astype("int16"), scale, dsn)
	ds_risk = xr.Dataset({"ForestLossRisk":Risk})
	GlobalAttributes(ds_risk, fnameout=datasets[dsn])
	ds_risk.to_netcdf(datasets[dsn], format = 'NETCDF4', unlimited_dims = ["time"])
	print("Risk Dataset Built")


	
def dsinfomaker(compath, backpath, mwb, tcfs, SR="SR"):
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
	dsinfo["Risk"]          = ({"alias":"Forest Loss Risk"})
	dsinfo["SRfrac"]        = ({"alias":"Stand Replacing Fire Percentage", "long_name":f'FRI$_{{{"SR"}}}$ %'})

	for dsnm in dsinfo:
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
		dsinfo[dsnm]["fname"] = ppath+fname
	return dsinfo


def _subplotmaker(dsinfo, num, ax, var, dsn, datasets, mask,compath, backpath, proj,scale, bounds, latiMid, longMid, maskver, region = "SIBERIA", vmax = 80.0,shrink=0.85):
	"""
	Funstion to build subplots
	"""
	# ========== open the dataset ==========
	if dsn == "Risk":
		vmax = 6.5
		if not os.path.isfile(datasets[dsn]):
			_RiskBuilder(dsinfo, num, ax, var, dsn, datasets, mask, compath, backpath, proj, scale, bounds, latiMid, longMid, maskver, shrink=shrink)

		frame = _fileopen(dsinfo, datasets, dsn, "ForestLossRisk", scale, proj, mask, compath, region, bounds, maskver, func = "mean")

		# ========== Set the colors ==========
		cmap, norm, vmin, vmax, levels = _colours( "ForestLossRisk", vmax, dsn)

		# ========== Create the Title ==========
		title = ""
		extend = "neither"
	
	elif dsn == 'SRfrac':
		vmax = 1
		if not os.path.isfile(datasets[dsn]):
			_SRfracBuilder(dsinfo, num, ax, var, dsn, datasets, mask, compath, backpath, proj, scale, bounds, latiMid, longMid, maskver, shrink=shrink)
		
		frame = _fileopen(dsinfo, datasets, dsn, "StandReplacingFireFraction", scale, proj, mask, compath, region, bounds, maskver, func = "mean")
		frame *= 100
		# breakpoint()
		# frame = _fileopen(dsinfo, datasets, dsn, "ForestLossRisk", scale, proj, mask, compath, region, bounds, maskver, func = "mean")

		# ========== Set the colors ==========
		cmap, norm, vmin, vmax, levels = _colours( "StandReplacingFireFraction", vmax, dsn)

		# ========== Create the Title ==========
		title = ""
		extend = "neither"
		# breakpoint()

	else:
		if not os.path.isfile(datasets[dsn]):

			# The file is not in the folder
			warn.warn(f"File {datasets[dsn]} could not be found")
			breakpoint()
		else:
			frame = _fileopen(dsinfo, datasets, dsn, var, scale, proj, mask, compath, region, bounds, maskver)

	
		# ========== Set the colors ==========
		cmap, norm, vmin, vmax, levels = _colours(var, vmax, dsn)

		# ========== Create the Title ==========
		title = ""
		extend = "max"


	# ========== Grab the data ==========
	if proj == "polar":
		# .imshow
		# breakpoint()

		im = frame.compute().plot(
			ax=ax, vmin=vmin, vmax=vmax, 
			cmap=cmap, norm=norm, 
			transform=ccrs.PlateCarree(),
			# add_colorbar=False,
			cbar_kwargs={"pad": 0.02, "extend":extend, "shrink":shrink, "ticks":levels, "spacing":"uniform"}
			) #
			# subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)}
		if dsn == "Risk":
			cbar = im.colorbar
			keys =  pd.DataFrame(_riskkys()).T
			# cbar.set_ticklabels( keys.Code.values)  # horizontal colorbar
			cbar.set_ticklabels( keys.FullName.values)
			# 
		# breakpoint()
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

#==============================================================================
def _fileopen(dsinfo, datasets, dsn, var, scale, proj, mask, compath, region, bounds, maskver, func = "mean"):
	ds_dsn = xr.open_dataset(datasets[dsn])
	# xbounds [-10.0, 180.0, 70.0, 40.0]
	xbounds = [-10.0, 180.0, 70.0, 40.0]
	# ========== Get the data for the frame ==========
	frame = ds_dsn[var].isel(time=0).sortby("latitude", ascending=False).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1]))).drop("time")
	
	if proj == "polar" and not dsn == "GFED":
		# ========== Coarsen to make plotting easier =========
		if func == "mean":
			frame = frame.coarsen(
				{"latitude":scale[dsn], "longitude":scale[dsn]
				}, boundary ="pad").mean().compute()
		elif func == "max":
			frame = frame.coarsen(
				{"latitude":scale[dsn], "longitude":scale[dsn]
				}, boundary ="pad").max().compute()
		else:
			print("Unknown Function")
			breakpoint()
	
	frame.attrs = dsinfo[dsn]#{'long_name':"FRI", "units":"years"}

	# ========== mask ==========
	if mask:
		# +++++ Setup the paths +++++
		# stpath = compath +"/Data51/ForestExtent/%s/" % dsn
		stpath = compath + "/masks/broad/"

		if dsn.startswith("H") or (dsn in ["Risk", "SRfrac"]):
			fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedToesacci.nc" % (region)
			fnBmask = f"./data/LandCover/Regridded_forestzone_esacci.nc"
		else:
			fnmask = stpath + "Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsn)
			fnBmask = f"./data/LandCover/Regridded_forestzone_{dsn}.nc"

		# +++++ Check if the mask exists yet +++++
		if os.path.isfile(fnmask):
			with xr.open_dataset(fnmask).drop("treecover2000").rename({"datamask":"mask"}) as dsmask, xr.open_dataset(fnBmask).drop(["DinersteinRegions", "GlobalEcologicalZones", "LandCover"]) as Bmask:
				# breakpoint()
				if maskver == "Boreal":
					msk    = (dsmask.mask.isel(time=0)*((Bmask.BorealMask.isel(time=0)>0).astype("float32")))#.sel(dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
				else:
					msk    = (dsmask.mask.isel(time=0)).astype("float32")
				
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
			# breakpoint()
			# cmapHex = palettable.matplotlib.Viridis_9_r.hex_colors
			if dsn.startswith("H"):
				cmapHex = palettable.colorbrewer.diverging.Spectral_9.hex_colors
				del cmapHex[3] #remove some middle colours
				del cmapHex[1]
				levels = [0, 60, 120, 500, 1000, 3000, 10000, 10001]
			else:
				cmapHex = palettable.colorbrewer.diverging.Spectral_9.hex_colors
				levels = [0, 15, 30, 60, 120, 500, 1000, 3000, 10000, 10001]
		else:
			cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors

		cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
		
		if vmax == 10000:
			if dsn.startswith("H"):
				norm   = mpl.colors.BoundaryNorm([0, 60, 120, 500, 1000, 3000, 10000], cmap.N)
			else:
				norm   = mpl.colors.BoundaryNorm([0, 15, 30, 60, 120, 500, 1000, 3000, 10000], cmap.N)

		cmap.set_over(cmapHex[-1] )
		cmap.set_bad('dimgrey',1.)

	elif var ==  "ForestLossRisk":
		vmin = -0.5
		cmapHex = palettable.cartocolors.qualitative.Prism_9.hex_colors[2:]
		levels = [0, 1, 2, 3, 4, 5, 6]
		cmap    = mpl.colors.ListedColormap(cmapHex)
		cmap.set_bad('dimgrey',1.)

	elif var == "StandReplacingFireFraction":
		vmin = 0.01
		vmax = 100
		cmapHex = palettable.cmocean.diverging.Curl_20.hex_colors#[2:] #Matter_11
		# levels  = np.arange(vmin, vmax+0.05, 0.10)
		cmap    = mpl.colors.ListedColormap(cmapHex)#[1:-1])
		cmap.set_bad('dimgrey',1.)
		cmap.set_over(cmapHex[-1])
		cmap.set_under(cmapHex[0])
		norm=LogNorm(vmin = vmin, vmax = vmax)
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

def _riskkys():
	keys = OrderedDict()
	keys[0] = {"Code":"LR",   "FullName":"Low Risk"}
	keys[1] = {"Code":"MRd",  "FullName":"Mod. Risk (dist)"}
	keys[2] = {"Code":"MRf",  "FullName":"Mod. Risk (fire)"}
	keys[3] = {"Code":"HRd",  "FullName":"High Risk (dist)"}
	keys[4] = {"Code":"HRf",  "FullName":"High Risk (fire)"}
	keys[5] = {"Code":"VHRd", "FullName":"Extreme Risk (dist)"}
	keys[6] = {"Code":"VHRf", "FullName":"Extreme Risk (fire)"}
	return keys
	

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
	elif sysname in ['LAPTOP-8C4IGM68', 'DESKTOP-N9QFN7K']:
		dpath     = "./data"
		backpath = "/mnt/d/fireflies"
	else:
		ipdb.set_trace()
	return dpath, backpath

def GlobalAttributes(ds, fnameout=""):
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
	if ds is None:
		attr = OrderedDict()
	else:
		attr = ds.attrs

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["FileName"]            = fnameout
	attr["title"]               = "RiskFramework"
	attr["summary"]             = "BorealForestLossRisk" 
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. " % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
	
	if not ds is None:
		attr["history"]            += ds.history

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "Woodwell"
	attr["date_created"]        = str(pd.Timestamp.now())
	ds.longitude.attrs['units'] = 'degrees_east'
	ds.latitude.attrs['units']  = 'degrees_north'

	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	# attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	# attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	return attr	
#==============================================================================

if __name__ == '__main__':
	main()