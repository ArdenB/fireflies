"""
This script was started 21/8/2020.  The hansen results i have at this time 
are very different from the other FRI estimates. The goal of this script is 
to locate the source of this error and rectify it.  This will be done by focusing 
on a single, or small subset of locations. 
"""
#==============================================================================

__title__ = "Hansen Forest Loss Fixer"
__author__ = "Arden Burrell"
__version__ = "v1.0(21.08.2020)"
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
import rasterio
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats
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
import itertools
# Import debugging packages 
import ipdb
# from rasterio.warp import transform
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio import features
from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 

#==============================================================================

def main():

	# ========== Setup the params ==========
	mwbox   = [1, 2]#, 5]
	# dsnames = ["HANSEN_AFmask", "HANSEN"]
	dsnames = ["HANSEN"]
	compath = syspath()

	# ========== Open the hanesen results file ==========
	# // This file is (probably) wrong.  I need it to get the spatial domain 
	# // and to pick a location to do my testing on
	latmax = 55.
	latmin = 50.
	lonmin = 100.
	lonmax = 115.
	FCT     = 10.
	for mwb in mwbox:
		# ========== Setup the dataset ==========
		for dsnm in dsnames:
			ppath = compath + "/BurntArea/HANSEN/FRI/"
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
			ds_dsn = xr.open_dataset(ppath+fname).sel(dict(
				latitude =slice(latmax, latmin), 
				longitude=slice(lonmin, lonmax)))

			# ========== Open the original hansen ==========
			fn_og = "Hansen_GFC-2018-v1.6_lossyear_SIBERIA.nc"
			ds_og = xr.open_dataset(ppath+fn_og).sel(dict(
				latitude =slice(latmax+mwb, latmin-mwb), 
				longitude=slice(lonmin-mwb, lonmax+mwb))).load()

			# ========== Calculate scale factors ==========
			rat = np.round(mwb / np.array(ds_og["lossyear"].attrs["res"]) )
			if np.unique(rat).shape[0] == 1:
				# the scale factor between datasets
				SF    = int(rat[0])
				# RollF = int(SF/4 - 0.5) # the minus 0.5 is correct for rolling windows
				RollF = 100
			
			# ========== Import the Fraction of forest data ==========
			fn_fc = compath+"/BurntArea/HANSEN/FC2000/Hansen_GFC-2018-v1.6_treecover2000_SIBERIA.nc"
			ds_fc = (xr.open_dataset(fn_fc).sel(dict(
							latitude =slice(latmax+mwb, latmin-mwb), 
							longitude=slice(lonmin-mwb, lonmax+mwb)))>FCT).compute()

			# DS_af =  xr.open_dataset(
			# 	'/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN/HansenMODIS_activefiremask.nc', 
			# 	chunks={'latitude': dscf, "longitude": dscf}).sel(dict(latitude=slice(la[0], la[1]),longitude=slice(la[2], la[3])))

			# ========== Make a dataset of the same shape as ds_dsn ==========
			FRI     = np.zeros(ds_dsn.FRI.shape)
			FRI[:]  = np.NaN
			AnBF    = np.zeros(ds_dsn.AnBF.shape)
			AnBF[:] = np.NaN
			for yloc, lat in enumerate(ds_dsn.latitude):
				t0 = pd.Timestamp.now()
				# ========== Pull out a lon box of tree cover loss ==========
				ds_lsub = ds_og.sel(dict(
					latitude =slice(lat+(mwb/2.), lat-(mwb/2.))))#.compute()
				ds_subcom = (ds_lsub>0).sum(dim="latitude") # Make a boolean loss

				# ========== Do the same for tree cover ==========
				ds_latfc = ds_fc.sel(dict(
					latitude =slice(lat+(mwb/2.), lat-(mwb/2.)))).sum(dim="latitude")
				t1 = pd.Timestamp.now()
				# ========== Roll the lon box and the extract the relevant pixels ==========
				cnts = ds_subcom.lossyear.rolling(longitude=SF, center=True).sum().reindex(
					{"longitude":ds_dsn.longitude}, method="nearest")
				numf = ds_latfc.treecover2000.rolling(longitude=SF, center=True).sum().reindex(
					{"longitude":ds_dsn.longitude}, method="nearest")
				# breakpoint()

				# ========== Store the results in an array ==========
				# float(SF * SF)
				ABF = ((cnts /numf)/18.).values.ravel()
				AnBF[0, yloc, :] = ABF
				FRI [0, yloc, :] = 1./ABF
				
				# for xloc, lon in enumerate(ds_dsn.longitude):
				# 	ds_fcs = ds_fc.sel(dict(
				# 		latitude =slice(lat+(mwb/2.), lat-(mwb/2.)),
				# 		longitude=slice(lon-(mwb/2.), lon+(mwb/2.)))).compute()
					
				# 	ABP = ds_sub.lossyear.sum() /18.
				# 	pix = float(SF * SF)
				# 	ABF = ABP/(ds_fcs.treecover2000>10.).sum()
				# 	breakpoint()
					# ds_sub = ds_lsub.sel(dict(latitude =slice(lat+(mwb/2.), lat-(mwb/2.)),longitude=slice(lon-(mwb/2.), lon+(mwb/2.))))
					# ds_sub = ds_lsub.sel(dict(longitude=slice(lon-(mwb/2.), lon+(mwb/2.))))
					
					# ========== Slow way ===========
					# ds_sub = ds_subcom.sel(dict(longitude=slice(lon-(mwb/2.), lon+(mwb/2.))))
					# ABF =  (ds_sub.lossyear.sum()/float(SF * SF))/18.
					# AnBF[0, yloc, xloc] = ABF
					# FRI [0, yloc, xloc] = 1./ABF


					# ds_sub2.lossyear.sum()
					# +++++ Annual Lost here +++++
					# // Idea, i can use numpy unique to work out the number of places per year 
					# // This can be implemented later 
					# yrs, counts = np.unique(ds_sub.lossyear.values, return_counts=True)
					# # warn.warn("TO DO: remove the fraction that is not forested for any calculation")
					# fracburnt   = counts[yrs>0] / float(ds_sub.lossyear.size) 
					# breakpoint()
					
					# AnBF[0, yloc, xloc] = np.mean(fracburnt)
					# FRI [0, yloc, xloc] = 1./np.mean(fracburnt)
					# # need to exclude non forest here
					# +++++ Implement fire loss here +++++
					# // This is the other thing that needs to be implemented 
					# ========== Work out the total % of forest loss ==========
					# years  = np.arange(1, 19)
					# nyears = years.size
					# print(
					# 	# ds_dsn.sel(dict(latitude = lat,longitude=lon)).AnBF.values[0], 
					# 	np.mean(fracburnt), 
					# 	# ds_dsn.sel(dict(latitude = lat,longitude=lon)).FRI.values[0], 
					# 	1/np.mean(fracburnt))
				# breakpoint()
				t2 = pd.Timestamp.now()
				print(yloc, " of ", ds_dsn.latitude.size, pd.Timestamp.now(), t1-t0, t2-t1, t2-t0)
			ds_new = ds_dsn.copy(data={"AnBF":AnBF,"FRI":FRI})
			breakpoint()

#==============================================================================

def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
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
	elif sysname == 'burrell-pre5820':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		chunksize = 500
	elif sysname == 'arden-Precision-5820-Tower-X-Series':
		# WHRC linux distro
		dpath = "./data"
		breakpoint()
		# dpath= "/media/arden/Harbinger/Data51/BurntArea"
	else:
		ipdb.set_trace()
	return dpath

#==============================================================================
if __name__ == '__main__':
	main()