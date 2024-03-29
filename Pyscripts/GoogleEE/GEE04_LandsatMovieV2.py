"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

"""
#==============================================================================

__title__ = "GEE Movie Maker"
__author__ = "Arden Burrell"
__version__ = "v1.0(04.04.2019)"
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
import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob

from collections import OrderedDict
from scipy import stats
from numba import jit

# Import the Earth Engine Python Package
import ee
import ee.mapclient
from ee import batch
from geetools import batch as gee_batch

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 

import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default
# import seaborn as sns
# import cartopy.crs as ccrs
# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================


def main():
	# ========== Initialize the Earth Engine object ==========
	ee.Initialize()

	# ========== create the geometery ==========
	# ========== Get the cordinates ==========
	# coords = geom_builder()
	# site="G10T1-50"
	site="G5T1-50"

	coords = geom_builder(site=site)

	# ========== Load the Site Data ==========
	# syear    = 2018
	# SiteInfo = Field_data()
	# geom   = ee.Geometry.Point([coords.lon.values[0], coords.lat.values[0]])

	geom = ee.Geometry.Polygon([
			[coords.lonr_min.values[0],coords.latr_min.values[0]],
			[coords.lonr_max.values[0],coords.latr_min.values[0]],
			[coords.lonr_max.values[0],coords.latr_max.values[0]],
			[coords.lonr_min.values[0],coords.latr_max.values[0]]])

	# ========== Rename the LS8 bands to match landsat archive ==========
	# def renamebands(image):
	# 	return image.rename(['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'])
	def renamebandsETM(image):
		# Landsat 4-7
		bands    = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa']
		new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']
		return image.select(bands).rename(new_bands)

	def renamebands(image):
		# Landsat 8
		bands     = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa']
		new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']
		return image.select(bands).rename(new_bands)

	def LS7fix(image):
		 filled1a = image.focal_mean(1, 'square', 'pixels', 2)
		 # ipdb.set_trace()
		 return filled1a.blend(image).set(
		 	'system:time_start', image.get('system:time_start')).set(
		 	"SATELLITE", image.get("SATELLITE"))

	# ========== Define the image collection ==========
	program  = "LANDSAT"
	# bandlist = ['B4','B3', 'B2', 'B1']
	# ========== setup and reverse the bandlists ==========
	bandlist = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']
	bandlist.reverse()

	# program   = "sentinal"
	if program == "LANDSAT":
		dschoice  = "SR"#
		dsinfom   = "LANDSAT_5_7_8"
		dsbands   = "NRGB"
		# dschoice = "TOA"
		ls8c = "LANDSAT/LC08/C01/T1_%s" % dschoice
		L5coll = ee.ImageCollection(
			"LANDSAT/LT05/C01/T1_%s" % dschoice).filter(
			ee.Filter.lt('CLOUD_COVER',15)).map(
			renamebandsETM).filterBounds(geom).select(bandlist)

		L7coll = ee.ImageCollection(
			'LANDSAT/LE07/C01/T1_%s' % dschoice).filter(
			ee.Filter.lt('CLOUD_COVER',15)).map(
			renamebandsETM).filterBounds(geom).map(LS7fix).select(bandlist)

		L8coll = ee.ImageCollection(
			ls8c).filter(
			ee.Filter.lt('CLOUD_COVER', 15)).map(
			renamebands).filterBounds(geom).select(bandlist)

		collection = ee.ImageCollection(L5coll.merge(L7coll.merge(L8coll))).sort('system:time_start', True)

	else:
		ipdb.set_trace()
		sys.exit()
	# ========== Fetch the dates ==========
	info = []
	for elem in collection.getInfo()["features"]:
		utime = elem["properties"]['system:time_start']
		sat   = elem["properties"]["SATELLITE"]
		try:
			if sat =='LANDSAT_7':
				uid   = elem["properties"]['system:index']
			else:
				uid   = elem["properties"]['LANDSAT_ID']
		except KeyError:
			ipdb.set_trace()
			
		info.append({"satellite":sat, "time":utime, "unid":uid })


	# ========== convert dates to pandas dataframe ==========
	df         = pd.DataFrame(info)
	df["date"] = pd.to_datetime(df["time"], unit='ms', origin='unix')  
	df.to_csv("./data/other/tmp/%s_%s_%s_timeinfo.csv" % (dsinfom, coords.name.values[0], dsbands))
	coords.to_csv("./data/other/tmp/%s_%s_%s_gridinfo.csv" % (dsinfom, coords.name.values[0], dsbands))


	# # ========== Create a geotif ==========
	# print("Starting to create GeoTIFF's for %s at:" % coords.name.values[0], pd.Timestamp.now())
	# gee_batch.imagecollection.toDrive(
	# 	collection, 
	# 	"FIREFLIES_geotifs_%s" % site,
	# 	namePattern='%s_%s_%s_%s_{system_date}_{id}' % (dsbands, dsinfom, coords.name.values[0], dsbands), 
	# 	region=geom, 
	# 	crs = "EPSG:4326", 
	# 	fileFormat='GeoTIFF'
	# 	)
	# 	# maxFrames=10000
	# ipdb.set_trace()
	## Make 8 bit data
	def convertBit(image):
	    return image.multiply(512).uint8()  

	def convertBitV2(image):
		return image.multiply(0.0001).multiply(512).uint8()  
	## Convert bands to output video  
	if dschoice == "TOA":
		outputVideo = collection.map(convertBit)
	else:
		outputVideo = collection.map(convertBitV2)


	# if len(bandlist)> 3:
	# 	outputVideo = outputVideo.select(['R', 'G', 'B'])


	# # Export video to Google Drive
	# print("Starting to create video for %s at:" % coords.name.values[0], pd.Timestamp.now())
	# out = batch.Export.video.toDrive(
	# 	outputVideo, description='%s_%s_%s' % (dsinfom, coords.name.values[0], dsbands), 
	# 	folder = "/GEE_VIDEO_EXPORTS",
	# 	framesPerSecond = 1, #dimensions = 1080, 
	# 	region=(
	# 		[coords.lonr_min.values[0],coords.latr_min.values[0]],
	# 		[coords.lonr_max.values[0],coords.latr_min.values[0]],
	# 		[coords.lonr_max.values[0],coords.latr_max.values[0]],
	# 		[coords.lonr_min.values[0],coords.latr_max.values[0]]), 
	# 	crs = "EPSG:4326",
	# 	maxFrames=10000)

	print("Starting to create single image %s at:" % coords.name.values[0], pd.Timestamp.now())

	img_list = collection.toList(collection.size())
	# ipdb.set_trace()
	nx = 10
	name     = '%s_%s_%s_%04d' % (dsinfom, coords.name.values[0], dsbands, nx)


	out = batch.Export.image.toDrive(
		ee.Image(img_list.get(1)).toFloat(), description=name, 
		folder = "/GEE_VIDEO_EXPORTS",
		region=(
			[coords.lonr_min.values[0],coords.latr_min.values[0]],
			[coords.lonr_max.values[0],coords.latr_min.values[0]],
			[coords.lonr_max.values[0],coords.latr_max.values[0]],
			[coords.lonr_min.values[0],coords.latr_max.values[0]]), 
		crs = "EPSG:4326", fileFormat='GeoTIFF')

	process = batch.Task.start(out)
	print("Process sent to cloud")


	ipdb.set_trace()

	# [112.40420250574721,51.22323236456422]


def geom_builder(site = "Burn2015 UP"):
	"""
	function to make the geometery 
	"""
	# ========== Create a container ==========
	coords = OrderedDict()
	
	# ========== Load the site data ==========
	pointfn = "./data/field/Points.kml"
	pointdt = gpd.read_file(pointfn, driver="kml")
	
	# ========== Pull out the location of a point ==========
	lon = pointdt[pointdt.Name == site].geometry.x.values
	lat = pointdt[pointdt.Name == site].geometry.y.values

	# ========== get the local data info ==========
	local_data = datasets()
	ldsi       = local_data["COPERN"]

	# ========== load in the grid data ==========
	ds_gr = xr.open_dataset(ldsi["fname"], chunks=ldsi["chunks"])["NDVI"].rename(ldsi["rename"]).isel(time=1)
	gr_bx = ds_gr.sel({"latitude":lat, "longitude":lon}, method="nearest")
	# ========== Work out the edges of the grid box ==========
	latstep = abs(np.unique(np.round(np.diff(ds_gr.latitude.values), decimals=9)))/2.0
	lonstep = abs(np.unique(np.round(np.diff(ds_gr.longitude.values), decimals=9)))/2.0

	# ========== Get values ready to export ==========
	if site == "Burn2015 UP":
		coords["name"] = "TestBurn"
	else:
		coords["name"] = site

	coords["lon"]      = lon
	coords["lat"]      = lat
	
	coords["lonb_max"] = gr_bx.longitude.values + lonstep
	coords["lonb_min"] = gr_bx.longitude.values - lonstep
	coords["latb_max"] = gr_bx.latitude.values  + latstep
	coords["latb_min"] = gr_bx.latitude.values  - latstep


	coords["lonr_max"] = (gr_bx.longitude.values + 2*(lonstep*2)) + lonstep
	coords["lonr_min"] = (gr_bx.longitude.values - 2*(lonstep*2)) - lonstep
	coords["latr_max"] = (gr_bx.latitude.values  + 2*(latstep*2)) + latstep
	coords["latr_min"] = (gr_bx.latitude.values  - 2*(latstep*2)) - latstep
	
	return pd.DataFrame(coords)

def Field_data(year = 2018):
	"""
	# Aim of this function is to look at the field data a bit

	To start it just opens the file and returns the lats and longs 
	i can then use these to look up netcdf fils
	"""
	# ========== Load in the relevant data ==========
	if year == 2018:
		fd18 = pd.read_csv("./data/field/2018data/siteDescriptions18.csv")
	else:
		fd18 = pd.read_csv("./data/field/2018data/siteDescriptions17.csv")

	fd18.sort_values(by=["site number"],inplace=True) 
	# ========== Create and Ordered Dict for important info ==========
	info = OrderedDict()
	info["sn"]  = fd18["site number"]
	try:
		info["lat"] = fd18.lat
		info["lon"] = fd18.lon
		info["RF"]  = fd18.rcrtmnt
	except AttributeError:
		info["lat"] = fd18.strtY
		info["lon"] = fd18.strtX
		info["RF"]  = fd18.recruitment
	
	# ========== function to return nan when a value is missing ==========
	def _missingvalfix(val):
		try:
			return float(val)
		except Exception as e:
			return np.NAN

	def _fireyear(val):
		try:
			year = float(val)
			if (year <= 2018):
				return year
			else:
				return np.NAN
		except ValueError: #not a simple values
			try:
				year = float(str(val[0]).split(" and ")[0])
				if year < 1980:
					warn.warn("wrong year is being returned")
					year = float(str(val).split(" ")[0])
					# ipdb.set_trace()

				return year
			except Exception as e:
				# ipdb.set_trace()
				# print(e)
				print(val)
				return np.NAN

	# info[den] = [_missingvalfix(
	# 	fcut[fcut.sn == sn][den].values) for sn in info['sn']]

	# info["RF17"] = [_missingvalfix(
	# 	fcut[fcut.sn == sn]["RF2017"].values) for sn in info['sn']]
	
		
	info["fireyear"] = [_fireyear(fyv) for fyv in fd18["estimated fire year"].values]
	# ========== Convert to dataframe and replace codes ==========
	RFinfo = pd.DataFrame(info).set_index("sn")
	return RFinfo

def datasets():
	# ========== set the filnames ==========
	data= OrderedDict()
	data["GIMMS"] = ({
		"fname":"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc",
		'var':"ndvi", "gridres":"8km", "region":"global", "timestep":"Annual", 
		"start":1982, "end":2017, "rasterio":False, "chunks":{'time': 36},
		"rename":None
		})
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
		'var':"NDVI", "gridres":"1km", "region":"Global", "timestep":"AnnualMax",
		"start":1999, "end":2018,"rasterio":False, "chunks":{'time':1}, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	data["COPERN_BA"] = ({
		'fname':"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/M0044633/c_gls_BA300_201812200000_GLOBE_PROBAV_V1.1.1.nc",
		'var':"BA_DEKAD", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":None, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	
	data["esacci"] = ({
		"fname":"./data/BurntArea/20010101-ESACCI-L3S_FIRE-BA-MODIS-AREA_4-fv5.1-JD.tif",
		'var':"BurntArea", "gridres":"250m", "region":"Asia", "timestep":"Monthly", 
		"start":2001, "end":2018, "rasterio":True, "chunks":{'latitude': 1000},
		"rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	# data["MODISaqua"] = ({
	# 	"fname":"./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc",
	# 	'var':"ndvi", "gridres":"250m", "region":"SIBERIA", "timestep":"16day", 
	# 	"start":2002, "end":2019
	# 	})
	# data["MODIS_CMG"] = ({
	# 	"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/NDVI/5.MODIS/terra/processed/MODIS_terra_MOD13C1_5kmCMG_anmax.nc",
	# 	'var':"ndvi", "gridres":"5km", "region":"Global", "timestep":"AnnualMax", 
	# 	"start":2000, "end":2018
		# })
	return data

#==============================================================================
if __name__ == '__main__':
	main()