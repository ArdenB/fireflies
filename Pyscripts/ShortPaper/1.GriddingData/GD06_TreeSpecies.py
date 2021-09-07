"""
Script goal, 

GEt tree species data and convert it to a netcdf

"""
#==============================================================================

__title__ = "Tree species data"
__author__ = "Arden Burrell"
__version__ = "v1.0(07.09.2021)"
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
import rasterio

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
# import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	# ========== Setup the broad infomation
	region = "SIBERIA"
	xbounds = [-10.0, 180.0, 70.0, 40.0]

	# ========== Load in the different data from glc ==========
	path = "./data/LandCover/Bartalev"
	dtfn = f"{path}/Land_cover_map_Bartalev/land_cover_from_bartalev_tif_geo.tif"
	lkfn = f"{path}/Land_cover_map_Bartalev/BartalevLookup.csv"

	da = xr.open_rasterio(dtfn).rename({"x":"longitude", "y":"latitude", "band":"time"}).sel(
		dict(latitude=slice(xbounds[2], xbounds[3]), longitude=slice(xbounds[0], xbounds[1])))
	da = da.where(~(da == 255))
	da = da.where(~(da == 0))
	da["time"] = [pd.Timestamp("2018-12-31")]

	# ========== build the lookup table ==========
	df = pd.read_csv(lkfn) 
	tab = ""
	for vl, nm in zip(df.Value, df.Class_Name):
		if not vl ==0:
			tab += f"({vl},{nm}), "

	# ========== convert to a netcdf file ==========
	ds = xr.Dataset({"TreeSpecies":da})
	# ds["lookup"] = tab
	# breakpoint()
	fnout = f"{path}/Bartalev_TreeSpecies.nc"
	GlobalAttributes(ds, "Bartalev", fnout)
	ds.attrs["Lookup"] = tab
	ds.to_netcdf(fnout, format = 'NETCDF4', unlimited_dims = ["time"])
	ds = None


	# ========== load in esa data and save it
	for dsn  in ["esacci", "TerraClimate"]:
		mskfn  = "./data/masks/broad/Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsn)
		ds_msk = xr.open_dataset(mskfn)#.sel(dict(latitude=slice(box[3], box[2]), longitude=slice(box[0], box[1])))
		mask   = ds_msk.datamask
		da     = da.reindex_like(mask, method="nearest")
		# ========== convert to a netcdf file ==========
		ds = xr.Dataset({"TreeSpecies":da})
		# breakpoint()
		fnout = f"{path}/Bartalev_TreeSpecies_ProcessedTo{dsn}.nc"
		GlobalAttributes(ds, dsn, fnout)
		ds.attrs["Lookup"] = tab
		ds.to_netcdf(fnout, format = 'NETCDF4', unlimited_dims = ["time"])

	breakpoint()



def GlobalAttributes(ds, dsn, fnameout=""):
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
	attr["title"]               = "Tree species"
	attr["summary"]             = "BartalevTree_%sData" % (dsn)
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. Gridded to resolution using %s data" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, dsn)
	
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


# ==============================================================================

if __name__ == '__main__':
	main()