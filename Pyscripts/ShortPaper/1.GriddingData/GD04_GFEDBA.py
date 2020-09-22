"""
Script goal, 

Open the new fire dataset and upscale it using a nereast neibour to reduce 
the spatial error cause by even number moving windows

"""
#==============================================================================

__title__ = "GFED fire builder"
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
# import pdb as ipdb
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	# ========== Build the paths and the file names 
	dpath = "./data/BurntArea/GFED/"
	cf.pymkdir(dpath+"processed/")
	# fnames = sorted(glob.glob(dpath + "raw/GFED4.1s_*[0-9].hdf5"))
	upscale = 5
	box = [-10.0, 180, 40, 70]
	# ========== Make containers to hold lat an lon ==========
	# /// Calculate this once and then use it for the rest of the time 
	lat  = None
	lon  = None
	anls = [] #container for the annual sumed xr da's 

	# ========== Loop over the file names ==========
	for yr in range(1997, 2017):
		print(yr, pd.Timestamp.now())
		dsin = xr.open_dataset(dpath+f"raw/GFED4.1s_{yr}.hdf5", 
			engine="pynio")
		
		# ========== Calculate the lat and lon ==========
		if lat is None:
			# +++++ pull out the og lat an lon values
			latog    = dsin.lat.values[:, 0]
			latattrs = dsin.lat.attrs
			lonog    = dsin.lon.values[0, :]
			lonattrs = dsin.lon.attrs

			# ========== sf and tiling ==========
			# calculate the scale factors 
			ltsc = (latog[1]-latog[0])/upscale
			lnsc = (lonog[1]-lonog[0])/upscale
			if upscale % 2 == 1:
				sf = np.arange(-(upscale - (upscale % 2))/2, ((upscale - (upscale % 2))/2 +1))
			else:
				sf = np.arange(upscale) - upscale/2
			latsf = np.tile(sf*ltsc, latog.size)
			lonsf = np.tile(sf*lnsc, lonog.size)

			# ========== calculate the lat and lon ==========
			lat = np.repeat(latog, upscale) + latsf
			lon = np.repeat(lonog, upscale) + lonsf

		# ========== Build a container to put the montly numpy arrays in ==========
		das = []
		# ========== Loop over the months ==========
		for mn in range(1, 13):
			vanm = f"burned_area/{mn:02}/burned_fraction"

			da = dsin[vanm].values

			# ========== use repeate to make a NN upscaling ==========
			da_scaled = np.repeat(np.repeat(da, upscale, axis =1), upscale, axis=0)
			
			# ========== Build a 1D version of the array ==========
			das.append(da_scaled)
		# ========== Calculat ehe annual ba ==========
		arr = np.expand_dims( np.sum(np.dstack(das), axis=2), axis=0)
		da_an = xr.DataArray(arr, coords={
			"time":[pd.Timestamp(f"{yr}-12-31")],
			"latitude":lat, "longitude":lon}, dims=["time", "latitude", "longitude"])
		anls.append(da_an.sel(dict(latitude=slice(box[3], box[2]), longitude=slice(box[0], box[1]))))

	# ========== Build a data Array and add the attrivutes to it ==========
	fnout =  dpath+"processed/GFED_annual_burendfraction.nc"
	ds_out = xr.Dataset({"BA":xr.concat(anls, dim="time")})
	# /// add a mask for greater than 1 values ///
	ds_out = ds_out.where(ds_out.BA <=1, 1)

	# DAin_sub = .sel(dict(latitude=slice(box[3], box[2]), longitude=slice(box[0], box[1])))
	attrs = GlobalAttributes(ds_out, upscale, dsn="GFED")
	ds_out.attrs = attrs
	ds_out.latitude.attrs  = latattrs
	ds_out.longitude.attrs = lonattrs

	encoding =  ({"BA":{'shuffle':True,'zlib':True,'complevel':5}})
	delayed_obj = ds_out.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		encoding       = encoding,
		unlimited_dims = ["time"],
		compute=False)

	print("Starting write of upscaled GFED data at", pd.Timestamp.now())
	with ProgressBar():
		results = delayed_obj.compute()
	breakpoint()

def GlobalAttributes(ds, upscale, dsn="GFED"):
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
	attr["FileName"]            = ""
	attr["title"]               = "BurnedArea"
	attr["summary"]             = "Annual Burned area for GFED data"
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. %s data upsampled by %d by NN" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, dsn, upscale)

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "WHRC"
	attr["date_created"]        = str(pd.Timestamp.now())
	
	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	# attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	# attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	return attr

#==============================================================================
if __name__ == '__main__':
	main()