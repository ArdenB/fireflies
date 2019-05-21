
"""
Prelim script for looking at netcdf files and producing some trends
Broken into three parts
	Part 1 pull out the NDVI from the relevant sites
"""
#==============================================================================

__title__ = "Boreal Forest Mask"
__author__ = "Arden Burrell"
__version__ = "v1.0(03.05.2019)"
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
import xarray as xr
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
import regionmask as rm
# Import debugging packages 
import ipdb
# from rasterio.warp import transform
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio import features
from affine import Affine
# +++++ Import my packages +++++
# import MyModules.CoreFunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================
def main():

	# ========== Load in the data ==========
	# Boreal forest extent
	br     = gpd.read_file("./data/other/ForestExtent/borealbiome/boreal.shp")
	shapes = [(shape, n) for n, shape in enumerate(br.geometry)]

	# ========== Get the infomation about the vegetation datasets ==========
	data   = datasets()
	
	# ========== Create the mask dates ==========
	dates  = datefixer(2005, 1, 1)
	
	# ========== Loop over the included datasets ==========
	for dsn in data:
		if dsn in ["MODISaqua", "MODISterra"]:
			ds  = xr.open_mfdataset(data[dsn]["fname"])
		else:
			ds  = xr.open_dataset(data[dsn]["fname"])
		# ========== Set up the filename and global attributes
		fpath = "./data/other/ForestExtent/"
		fnout = fpath+ "BorealForestMask_%s.nc" % dsn
		global_attrs = GlobalAttributes(ds, fnout, dsn)	
		
		# ========== Pull out the old lat and lons ==========
		try:
			lon = ds.longitude
			lat = ds.latitude
		except:
			lon = ds.lon.values
			lat = ds.lat.values #.rename("latitude").v
		
		# ========== create an empty dataset for the mask ==========
		dsm  = xr.Dataset(
			coords={'time': dates["CFTime"], 'longitude': lon,'latitude': lat},
			attrs= global_attrs)
		dsm['BorealForest'] = rasterize(shapes, dsm.coords, dates)
		
		# ========== Save the file out ==========
		print("Starting write of data")
		dsm.to_netcdf(fnout, 
			format         = 'NETCDF4', 
			# encoding       = encoding,
			unlimited_dims = ["time"])
		print(".nc mask file for %s data created" % dsn)
		
	ipdb.set_trace()
	# this shapefile is from natural earth data
	# http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/
	# states = geopandas.read_file('/Users/shoyer/Downloads/ne_10m_admin_1_states_provinces_lakes')
	# us_states = states.query("admin == 'United States of America'").reset_index(drop=True)
	# state_ids = {k: i for i, k in enumerate(us_states.woe_name)}
	# shapes = [(shape, n) for n, shape in enumerate(us_states.geometry)]

	# ds = xray.Dataset(coords={'longitude': np.linspace(-125, -65, num=5000),
	#                           'latitude': np.linspace(50, 25, num=3000)})

	# # example of applying a mask
	# ds.states.where(ds.states == state_ids['California']).plot()

#==============================================================================

def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, dates, fill=np.nan, maskvals=True, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['latitude'], coords['longitude'])
    out_shape = (len(coords['latitude']), len(coords['longitude']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    
    # ========== change the values to mask ==========
    if maskvals:
    	raster[~np.isnan(raster)] = 1.0

    # ========== Make the DA ==========
    DA = xr.DataArray(
    	raster.reshape([1, raster.shape[0], raster.shape[1]]), 
    	coords=coords, 
    	dims=('time', 'latitude', 'longitude'),
    	attrs = ({
    		'_FillValue':-1, #9.96921e+36
    		'units'     :"1",
    		'standard_name':"BFmask",
    		'long_name':"BorealForestMask",
    		'valid_range': [0.0, 2.0]
    		}))
    DA.longitude.attrs['units'] = 'degrees_east'
    DA.latitude.attrs['units']  = 'degrees_north'
    DA.time.attrs["calendar"]   = dates["calendar"]
    DA.time.attrs["units"]      = dates["units"]
    
    return DA

#==============================================================================

def datefixer(year, month, day):
	"""
	Opens a netcdf file and fixes the data, then save a new file and returns
	the save file name
	args:
		ds: xarray dataset
			dataset of the xarray values
	return
		time: array
			array of new datetime objects
	"""


	# ========== create the new dates ==========
	# +++++ set up the list of dates +++++
	dates = OrderedDict()
	tm = [dt.datetime(int(year) , int(month), int(day))]
	dates["time"] = pd.to_datetime(tm)

	dates["calendar"] = 'standard'
	dates["units"]    = 'days since 1900-01-01 00:00'
	
	dates["CFTime"]   = date2num(
		tm, calendar=dates["calendar"], units=dates["units"])

	return dates

#==============================================================================

def GlobalAttributes(ds, fnout, dsn):
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

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["FileName"]           = fnout
	attr["title"]               = "BorealForestMask"
	attr["summary"]             = "BorealForestMaskFor%sData" % (dsn)
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. Maks built using data from %s" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, "https://glad.umd.edu/projects/gfm/boreal/data.html")
	attr["history"]            += ds.history

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "University of Leicester"
	attr["date_created"]        = str(pd.Timestamp.now())
	
	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	# attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	# attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	return attr

def datasets():
	# ========== set the filnames ==========
	data= OrderedDict()
	data["GIMMS"] = ({
		"fname":"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1981to2017_mergetime_compressed.nc",
		'var':"ndvi", "gridres":"8km", "region":"Global", "timestep":"16day", 
		"start":1981, "end":2017
		})
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
		'var':"NDVI", "gridres":"1km", "region":"Global", "timestep":"AnnualMax",
		"start":1999, "end":2018
		})
	data["MODISaqua"] = ({
		"fname":"./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc",
		'var':"ndvi", "gridres":"250m", "region":"SIBERIA", "timestep":"16day", 
		"start":2002, "end":2019
		})
	data["MODIS_CMG"] = ({
		"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/NDVI/5.MODIS/terra/processed/MODIS_terra_MOD13C1_5kmCMG_anmax.nc",
		'var':"ndvi", "gridres":"5km", "region":"Global", "timestep":"AnnualMax", 
		"start":2000, "end":2018
		})
	# data["MYD13C1"] = ({
	# 	"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/NDVI/5.MODIS/aqua/5km/processed/MODIS_aqua_MYD13C1_5kmCMG_anmax.nc",
	# 	'var':"ndvi", "gridres":"5km", "region":"Global", "timestep":"AnnualMax", 
	# 	"start":2000, "end":2018
	# 	})
	return data
#==============================================================================

if __name__ == '__main__':
	main()