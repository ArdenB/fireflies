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
import time

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
import geopy.distance as geodis
import myfunctions.corefunctions as cf 
# # Import debugging packages 
# import socket
# print(socket.gethostname())
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main(args):
	# ========== Initialize the Earth Engine object ==========
	ee.Initialize()

	# ========== Set an overwrite =========
	force = False
	cordf = True #force the creation of a new maskter coord list
	tsite = args.site
	cordg = True

	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-CSHARFM':
		# LAPTOP
		spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"

	elif sysname == "owner":
		spath = "/mnt/c/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
	elif sysname == "ubuntu":
		# Work PC
		spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"

	else:
		warn.warn("Paths not created for this computer")
		# spath =  "/media/ubuntu/Seagate Backup Plus Drive"
		ipdb.set_trace()
	cf.pymkdir(spath)

	# ========== create the geometery ==========
	cordname = "./data/other/GEE_sitelist.csv"
	if not os.path.isfile(cordname) or cordf:
		print("Generating and saving a new master coord table")
		site_coords = geom_builder()
		for col in site_coords.columns[1:]:
			site_coords = site_coords.astype({col:float})
		site_coords.to_csv(cordname)
	else:
		print("Loading master coord table")
		site_coords = pd.read_csv(cordname, index_col=0)#, parse_dates=True
		# warn.warn("THere is some form of bug here, going interactive. Look at the dataframe")
		# ipdb.set_trace()


	program = "LANDSAT"
	cordf = True
	# ========== Loop over each site ==========
	for index, coords in site_coords.iterrows():

		# ========== Check if the pathe and file exists ==========
		checkfile = "%s%s/%s_%s_gridinfo.csv" % (spath, coords["name"], program, coords["name"])

		if not args.site is None:
			# check is the site is correct 
			if tsite == coords["name"]:
				# ========== Get the start time ==========
				t0 = pd.Timestamp.now()

				if os.path.isfile("%s%s/raw/failed_geotifs.npy" % (spath, tsite)):

					fails = np.load("%s%s/raw/failed_geotifs.npy" % (spath, tsite))

					# Export the geotifs
					GEE_geotifexp(coords, spath, program, fails=fails)

					td = pd.Timestamp.now() - t0
					print("\n Data for %s sent sent for cloud processing. it took " % coords["name"], td)
				else:
					# ========== Get the start time ==========
					t0 = pd.Timestamp.now()

					# Export the geotifs
					GEE_geotifexp(coords, spath, program)

					td = pd.Timestamp.now() - t0
					print("\n Data for %s sent sent for cloud processing. it took " % coords["name"], td)
		elif cordg:
			print("Building a new cords file for %s" % coords["name"])
			ipdb.set_trace()
			coords.to_csv("%s%s/%s_%s_gridinfo.csv" % (spath, coords["name"], program, coords["name"]), header=True)
		elif os.path.isfile(checkfile) and not force:
			print("Data has already been exported for %s" % coords["name"])
			if cordf:
				coords.to_csv("%s%s/%s_%s_gridinfo.csv" % (spath, coords["name"], program, coords["name"]), header=True)
		else:
			# ipdb.set_trace()
			# sys.exit()
			# ========== Get the start time ==========
			t0 = pd.Timestamp.now()
			# make the dir
			cf.pymkdir(spath+coords["name"])

			# Export the geotifs
			GEE_geotifexp(coords, spath, program)

			td = pd.Timestamp.now() - t0
			print("\n Data for %s sent sent for cloud processing. it took " % coords["name"], td)

	if tsite is None:
		ipdb.set_trace()
		sys.exit()

#==============================================================================

def GEE_geotifexp(coords, spath, program, fails = None):
	""" function takes in coordinate infomation and begins the save out processs """
	try:
		geom = ee.Geometry.Polygon([
			[coords.lonr_min[0],coords.latr_min[0]],
			[coords.lonr_max[0],coords.latr_min[0]],
			[coords.lonr_max[0],coords.latr_max[0]],
			[coords.lonr_min[0],coords.latr_max[0]]])
	except:
		geom = ee.Geometry.Polygon([
			[coords.lonr_min,coords.latr_min],
			[coords.lonr_max,coords.latr_min],
			[coords.lonr_max,coords.latr_max],
			[coords.lonr_min,coords.latr_max]])

	# ========== Rename the LS8 bands to match landsat archive ==========
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
	
	# bandlist = ['B4','B3', 'B2', 'B1']
	# ========== setup and reverse the bandlists ==========
	bandlist = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']
	bandlist.reverse()

	# program   = "sentinal"
	if program == "LANDSAT":
		dschoice  = "SR"#
		dsinfom   = "LANDSAT_5_7_8"
		dsbands   = "SNRGB"
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
	# df.to_csv("%s%s/%s_%s_%s_timeinfo.csv" % (spath, coords["name"], dsinfom, coords["name"], dsbands))
	# coords.to_csv("%s%s/%s_%s_gridinfo.csv" % (spath, coords["name"], program, coords["name"]))
	# sys.exit()


	# gee_batch.imagecollection.toDrive(
	# 	collection, 
	# 	"FIREFLIES_geotifs" ,
	# 	namePattern='%s_%s_%s_%s_{system_date}_{id}' % (dsbands, dsinfom, coords["name"], dsbands), 
	# 	region=geom, 
	# 	crs = "EPSG:4326", 
	# 	fileFormat='GeoTIFF'
	# 	)

	print("Starting to create GeoTIFF's for %s at:" % coords["name"], pd.Timestamp.now())
	print("Attempting manual creation")

	# ========== Convert the collection into a selection of images
	img_list = collection.toList(collection.size())

	for nx, info in df.iterrows():
		# ========== Built to allow for scripts to be redone ==========
		if not fails is None:
			if not nx in fails:
				continue


		# ========== convert the datatype ==========
		img = ee.Image(img_list.get(nx)).toFloat()
		
		# ========== Create the name and path ==========
		name     = '%s_%s_%s_%04d' % (dsinfom, coords["name"], dsbands, nx)
		folder   = "FIREFLIES_geotifs"

		string = "\r Sending image %d of %d to the cloud for processing" % (nx, df.index.max())
		sys.stdout.write(string)
		sys.stdout.flush()
		# ========== Send the task to the cloud ==========
		try:

			task = ee.batch.Export.image.toDrive(
				image=img,
				description=name,
				folder=folder,
				crs = "EPSG:4326",
				region=(
					[coords.lonr_min[0],coords.latr_min[0]],
					[coords.lonr_max[0],coords.latr_min[0]],
					[coords.lonr_max[0],coords.latr_max[0]],
					[coords.lonr_min[0],coords.latr_max[0]]),
				scale=30, 
				fileFormat='GeoTIFF')
		except:
			task = ee.batch.Export.image.toDrive(
				image=img,
				description=name,
				folder=folder,
				crs = "EPSG:4326",
				region=(
					[coords.lonr_min,coords.latr_min],
					[coords.lonr_max,coords.latr_min],
					[coords.lonr_max,coords.latr_max],
					[coords.lonr_min,coords.latr_max]),
				scale=30, 
				fileFormat='GeoTIFF')
		try:
			process = batch.Task.start(task)
		except Exception as er:
			sle = 0
			print(str(er))
			warn.warn("Hit a task limit, sleeping for an hour to let tasks complete")
			while sle < 61:
				sle += 1
				string = "\r Starting sleep number %d at %s" % (sle, str(pd.Timestamp.now()))
				sys.stdout.write(string)
				sys.stdout.flush()
				time.sleep(60)

			process = batch.Task.start(task)
			# sys.exit()

	# ========== Code for old video export ==========
	oldvideo = False
	if oldvideo:
		# This is the way to use the google earth engine to make videos, i've
		# left the code here in case i need it again in the future


		## Convert bands to output video  
		if dschoice == "TOA":
			outputVideo = collection.map(convertBit)
		else:
			outputVideo = collection.map(convertBitV2)


		if len(bandlist)> 3:
			outputVideo = outputVideo.select(['B3', 'B2', 'B1'])


		testfirst = False
		if testfirst:
			task_config = {
				# 'description': 'imageToDriveTestExample',
				'scale': 30,  
				'region': geom,
				"crs" : "EPSG:4326", 
				"fileFormat":'GeoTIFF'
				}

			task = batch.Export.image.toDrive(outputVideo.first(), "testimage", task_config)
			task.start()
		# Export video to Google Drive
		print("Starting to create video for %s at:" % coords["name"], pd.Timestamp.now())
		out = batch.Export.video.toDrive(
			outputVideo, description='%s_%s_%s' % (dsinfom, coords["name"], dsbands), 
			folder = "/GEE_VIDEO_EXPORTS",
			framesPerSecond = 1, #dimensions = 1080, 
			region=(
				[coords.lonr_min[0],coords.latr_min[0]],
				[coords.lonr_max[0],coords.latr_min[0]],
				[coords.lonr_max[0],coords.latr_max[0]],
				[coords.lonr_min[0],coords.latr_max[0]]), 
			crs = "EPSG:4326",
			maxFrames=10000)
		process = batch.Task.start(out)
		print("Process sent to cloud")
	
	if fails is None:
		# ========== Save out the relevant infomation ==========
		df.to_csv("%s%s/%s_%s_%s_timeinfo.csv" % (spath, coords["name"], dsinfom, coords["name"], dsbands))
		coords.to_csv("%s%s/%s_%s_gridinfo.csv" % (spath, coords["name"], program, coords["name"]))

		# ========== Going to sleep to give GEE a rest before i slam it with new requests  ==========
		print("\n Starting 20 minutes of sleep at", pd.Timestamp.now(), "\n")
		sle = 0
		while sle < 20:
			sle += 1
			string = "\r Starting sleep number %d at %s" % (sle, str(pd.Timestamp.now()))
			sys.stdout.write(string)
			sys.stdout.flush()
			time.sleep(60)


#==============================================================================
#==============================================================================
#==============================================================================

## Make 8 bit data
def convertBit(image):
	return image.multiply(512).uint8()  

def convertBitV2(image):
	return image.multiply(0.0001).multiply(512).uint8()  


def geom_builder(site = "Burn2015 UP"):
	"""
	function to make the geometery 
	"""
	
	# ========== Load the site data ==========
	pointfn = "./data/field/Points.kml"
	pointdt = gpd.read_file(pointfn, driver="kml")


	sitenm = []
	latit  = []
	longi  = []
	year   = []    
	# ========== Loop over the names 2019 ==========
	for nm in pointdt.Name:
		if nm in ["Burn2015 UP", "GROUP BOX2 TRANS1-6"]:
			sitenm.append(nm)
			latit.append(pointdt[pointdt.Name == nm].geometry.y.values[0])
			longi.append(pointdt[pointdt.Name == nm].geometry.x.values[0])
			year.append(2019)
		elif "GROUP BOX" in nm:
			pass
		elif nm[-2:] == '-0':
			sitenm.append(nm)
			latit.append(pointdt[pointdt.Name == nm].geometry.y.values[0])
			longi.append(pointdt[pointdt.Name == nm].geometry.x.values[0])
			year.append(2019)

	# ========== add 2018 ==========
	fd18 = pd.read_csv("./data/field/2018data/siteDescriptions18.csv")
	fd18.sort_values(by=["site number"],inplace=True) 
	for nx, row in fd18.iterrows():
		sitenm.append("Site%02d" % row["site number"])
		latit.append(row.lat)
		longi.append(row.lon)
		year.append(2018)	
	
	# ========== add 2017 ==========
	fd17 = pd.read_csv("./data/field/2017data/siteDescriptions17.csv")
	fd17.sort_values(by=["site number"],inplace=True) 
	for nx, row in fd17.iterrows():
		stnm = "Site%02d" % row["site number"]
		if not stnm in sitenm:
			sitenm.append(stnm)
			latit.append(row.strtY)
			longi.append(row.strtX)
			year.append(2017)	
	
	# ========== Check the distance ==========
	# def distance_check(p1, p2):
		# pass
	STdf = pd.DataFrame({"siteds":sitenm, "lat":latit , "lon":longi , "year":year})
	STdf["Skip"] = 0
	STdf["SkipSite"] = ""
	for nx, row in STdf.iterrows():
		if STdf["Skip"][nx] > 0:
			# THis location has laready been skipped
			continue
		else:
			dist = np.array([geodis.distance((row.lat, row.lon), (lat, lon)).km for lat, lon in zip(STdf.lat[nx+1:].values, STdf.lon[nx+1:].values)])
			STdf["Skip"][nx+1:] += (dist<1).astype(int)
			def easy(inp, sitenm):
				if inp:
					return sitenm
				else:
					return ""

			close = [easy(inp, row.siteds) for inp in (dist<1)]
			STdf["SkipSite"][nx+1:] = STdf["SkipSite"][nx+1:].values + close
	# ipdb.set_trace()
	df = STdf[STdf.Skip == 0].reset_index(drop=True)

	def _sitemaker(site, sampleset, ds, dsn, sitinfoLS, lat, lon):
		
		""" wrapper to pull out site info as needed """
		
		# ========== Pull out the location of a point ==========
		# lon = pointdt[pointdt.Name == site].geometry.x.values
		# lat = pointdt[pointdt.Name == site].geometry.y.values
		
		# ========== Check if the site has already been built ==========
		if dsn == "COPERN": # The site has not been built yet
			# ========== set the key params ==========
			boxs  = 5   # number of grid cells considered 
			ident = "r" # The indertifing code of the dataset
			# ========== Create a container ==========
			coords = OrderedDict()


			# ========== Get values ready to export ==========
			if site == "Burn2015 UP":
				coords["name"] = "TestBurn"
			elif site == "GROUP BOX2 TRANS1-6":
				coords["name"] = "G2T1-6"
			else:
				coords["name"] = site
			coords["set"]      = sampleset
			coords["lon"]      = lon
			coords["lat"]      = lat

			# ========== Build the empty parts of the Ordered dic ==========
			for va_nm in ["r", "b_COP", "b_MOD"]:
				for ll in ["lon", "lat"]:
					for mm in ["max", "min"]:
						coords["%s%s_%s" % (ll, va_nm, mm)] = 0
		else:
			if dsn == "MODIS":
				boxs  = 3   # number of grid cells considered 
				ident = "b_MOD" # The indertifing code of the dataset
			elif dsn == "COPERN_BA":
				boxs  = 5   # number of grid cells considered 
				ident = "b_COP" # The indertifing code of the dataset
			
			coords = sitinfoLS[site]


		gr_bx = ds_gr.sel({"latitude":lat, "longitude":lon}, method="nearest")
		
		# ========== Work out the edges of the grid box ==========
		latstep = abs(np.unique(np.round(np.diff(ds_gr.latitude.values), decimals=9)))/2.0
		lonstep = abs(np.unique(np.round(np.diff(ds_gr.longitude.values), decimals=9)))/2.0
		if boxs == 3:
			coords["lon%s_max" % ident] = float((gr_bx.longitude.values + (lonstep*2)) + lonstep)
			coords["lon%s_min" % ident] = float((gr_bx.longitude.values - (lonstep*2)) - lonstep)
			coords["lat%s_max" % ident] = float((gr_bx.latitude.values  + (latstep*2)) + latstep)
			coords["lat%s_min" % ident] = float((gr_bx.latitude.values  - (latstep*2)) - latstep)
			# ipdb.set_trace()

		elif boxs == 5:
			coords["lon%s_max" % ident] = float((gr_bx.longitude.values + 2*(lonstep*2)) + lonstep)
			coords["lon%s_min" % ident] = float((gr_bx.longitude.values - 2*(lonstep*2)) - lonstep)
			coords["lat%s_max" % ident] = float((gr_bx.latitude.values  + 2*(latstep*2)) + latstep)
			coords["lat%s_min" % ident] = float((gr_bx.latitude.values  - 2*(latstep*2)) - latstep)


		sitinfoLS[site] = coords
		return sitinfoLS #coords
	
	# ========== setup an ordered dict of the names ==========
	sitinfoLS  = OrderedDict()
	local_data = datasets()
	for dsn in ["COPERN", "COPERN_BA", "MODIS"]:
		print(dsn)
		ldsi       = local_data[dsn]
		
		# ========== load in the grid data ==========
		if os.path.isfile(ldsi["fname"]):
			ds_gr = xr.open_dataset(
				ldsi["fname"], 
				chunks=ldsi["chunks"])[ldsi["var"]].rename(ldsi["rename"]).isel(time=0)
		else:
			ipdb.set_trace()

		# for nm in sitenm:
		for nx, row in df.iterrows():
			sitinfoLS = _sitemaker(row.siteds, row.year, ds_gr, dsn, sitinfoLS, row.lat, row.lon)
		
		# ========== Close the dataset ==========
		ds_gr = None
		# ipdb.set_trace()

	return pd.DataFrame(sitinfoLS).transpose()[sitinfoLS["Burn2015 UP"].keys()]

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

	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-CSHARFM':
		# LAPTOP
		dpath = "/mnt/e"
	elif sysname == "ubuntu":
		# Work PC
		dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51"
	elif sysname == "owner":
		# spath = "/mnt/c/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/mnt/d/Data51"
	else:
		warn.warn("Paths not created for this computer. System" + sysname)
		# dpath =  "/media/ubuntu/Seagate Backup Plus Drive"
		ipdb.set_trace()
	# ========== set the filnames ==========
	data= OrderedDict()
	data["MODIS"] = ({
		"fname":"%s/BurntArea/MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc" % dpath,
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
		})
	data["COPERN_BA"] = ({
		'fname':"%s/BurntArea/COPERN_BA/processed/COPERN_BA_gls_2014_burntarea_SensorGapFix.nc" % dpath,
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None
		})
	data["esacci"] = ({
		"fname":"%s/BurntArea/esacci/processed/esacci_FireCCI_2001_burntarea.nc" % dpath,
		'var':"BA", "gridres":"250m", "region":"Asia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
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

	return data

def string_format(string, replacement):
	""" Format a string using variables (as str.format) """

	s = ee.String(string)
	repl = ee.Dictionary(replacement)
	keys = repl.keys()
	values = repl.values().map(lambda v: ee.Algorithms.String(v))
	z = keys.zip(values)

	def wrap(kv, ini):
		keyval = ee.List(kv)
		ini = ee.String(ini)

		key = ee.String(keyval.get(0))
		value = ee.String(keyval.get(1))

		pattern = ee.String('{').cat(key).cat(ee.String('}'))

		return ini.replace(pattern, value)

	newstr = z.iterate(wrap, s)
	return ee.String(newstr)

def convertDataType(newtype):
    """ Convert an image to the specified data type
    :param newtype: the data type. One of 'float', 'int', 'byte', 'double',
        'Uint8','int8','Uint16', 'int16', 'Uint32','int32'
    :type newtype: str
    :return: a function to map over a collection
    :rtype: function
    """
    def wrap(image):
        TYPES = {'float': image.toFloat,
                 'int': image.toInt,
                 'byte': image.toByte,
                 'double': image.toDouble,
                 'Uint8': image.toUint8,
                 'int8': image.toInt8,
                 'Uint16': image.toUint16,
                 'int16': image.toInt16,
                 'Uint32': image.toUint32,
                 'int32': image.toInt32}
        return TYPES[newtype]()
    return wrap

#==============================================================================

if __name__ == '__main__':
	# ========== Set the args Description ==========
	description='Script to make movies'
	parser = argparse.ArgumentParser(description=description)
	
	# ========== Add additional arguments ==========
	parser.add_argument(
		"-s", "--site", type=str, default=None, help="Site to work with ")
	# parser.add_argument(
	# 	"--gparts", type=int, default=None,   
	# 	help="the max partnumber that has not been redone")
	parser.add_argument(
		"-f", "--force", action="store_true",
		help="the max partnumber that has not been redone")
	args = parser.parse_args() 
	
	# ========== Call the main function ==========
	main(args)
else:
	warn.warn("called from another script")
	# ========== Set the args Description ==========
	description='Script to make movies'
	parser = argparse.ArgumentParser(description=description)
	
	# ========== Add additional arguments ==========
	parser.add_argument(
		"-s", "--site", type=str, default=None, help="Site to work with ")

	# parser.add_argument(
	# 	"-x", "--cordforce", action="store_true",
	# 	help="just produce the cordinates without sending things to google earth engine")

	parser.add_argument(
		"-f", "--force", action="store_true",
		help="the max partnumber that has not been redone")
	
	args = parser.parse_args() 
	
	# ========== Call the main function ==========
	main(args)
