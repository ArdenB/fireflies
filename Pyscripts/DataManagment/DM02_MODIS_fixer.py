"""
Data Fixing script

Stiches together modis files into a single netcdf  

"""
#==============================================================================

__title__ = "Modis Fixer"
__author__ = "Arden Burrell"
__version__ = "v1.0(25.02.2019)"
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
import glob
import pandas as pd
import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
from netCDF4 import Dataset, num2date, date2num 
from scipy import stats
import subprocess as subp
import xarray as xr
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats
import subprocess as subp
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Import debugging packages 
import ipdb

# Import my own custom modules 
import myfunctions.corefunctions as cf

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================
def main():
	# ========== loop over each sensor ==========
	sensor = "aqua"
	cf.pymkdir("./data/veg/MODIS/%s/processed/" % sensor)
	
	SiteInfo = Field_data(year = 2017)
	# SiteInfo = None
	print("\n Starting nc creation \n")

	# ========== loop over each year ==========
	for year in range(2000, 2020):
		time0 = pd.Timestamp.now()
		# ========== loop over each day ==========
		for daynum in range(1, 367):
			if sensor == 'aqua':
				product = "MYD13Q1"
			else:
				ipdb.set_trace()
			sceneMoasic(sensor, product, year, daynum, SiteInfo=SiteInfo)
		time1 = pd.Timestamp.now()
		print("\n\n %d is complete. it took %s \n\n" % (year, str(time1-time0)))
		# ipdb.set_trace()
	ipdb.set_trace()

	pass

#==============================================================================
def sceneMoasic(sensor, product, year, daynum, force=False, SiteInfo=None):
	"""
	Uses the year and day number to look for files
	args:
		sensor: str
			string of the modis sensor name
		product: str
			string of the modis sensor product name
		year: int
			the year
		daynum: int
			the day number of the year
	"""

	# ========== get the filenames ==========
	fnames = glob.glob("./data/veg/MODIS/%s/M*_A%d%03d_*.nc" % (sensor, year, daynum))
	
	# ========== check if the file names are empty ==========
	if fnames==[]:
		return
	else:
		# ========== Create the file names ==========
		fO1 = "./data/veg/MODIS/%s/tmp/%stmp.nc" % (sensor, fnames[0].split('/')[-1][:-23])
		fO2 = "./data/veg/MODIS/%s/tmp/%stmp.nc" % (sensor, fnames[1].split('/')[-1][:-23])
		fout = "./data/veg/MODIS/%s/processed/%s_A%d%03d_merged.nc" % (sensor, product, year, daynum)
		
		# ========== Look for existing files ==========
		if os.path.isfile(fout) and not force:
			# Valid file already exists
			return
		else:
			print("\n Starting %d %03d at %s" % (year, daynum, str(pd.Timestamp.now())))
			t0 = pd.Timestamp.now()
	

	# ========== enlarge the grids ==========
	subp.call(
		" cdo -b F32 -enlargegrid,./data/veg/MODIS/%s/tmp/grid  %s %s" % (sensor, fnames[0], fO1),
		shell=True
	)
	subp.call(
		" cdo -b F32 -enlargegrid,./data/veg/MODIS/%s/tmp/grid  %s %s" % (sensor, fnames[1], fO2),
		shell=True
	)

	# ========== Open the files, rescale and merge ==========

	var = "_250m_16_days_NDVI"
	dst1   = xr.open_dataset(fO1)
	dst2   = xr.open_dataset(fO2)
	dstack = np.dstack([dst1[var].values*1e-8, dst2[var].values*1e-8]) #stacks and scales the file
	dmean  = np.expand_dims(bn.nanmean(dstack, axis=2), 0) #adds a time dim

	# ========== Set up the data Array properties ==========
	ky = 'ndvi' #short_name
	dates = datefixer(year, daynum) #dates
	lon   = dst1.longitude.values 
	lat   = dst1.latitude.values
	long_name = "normalized_difference_vegetation_index"
	fill_val  = -3

	# ========== Create the data array ==========
	da, encoding = DAbuilder(dmean, dates, lat, lon, ky, long_name, fill_val)

	# ========== Create the global attributes ==========
	global_attrs = GlobalAttributes(dst1, fO1, fO2, fout)

	# ========== Build the dataset ==========
	ds_out = xr.Dataset({ky:da}, attrs= global_attrs)

	# ========== Write the data out ==========
	print("Starting write of data at %s" % str(pd.Timestamp.now()))
	ds_out.to_netcdf(fout, 
		format         = 'NETCDF4', 
		encoding       = {ky:encoding},
		unlimited_dims = ["time"])
	t1 = pd.Timestamp.now()
	
	# ========== check the created dataset ==========
	try:
		ds = xr.open_dataset(fout)
		# ds.close()
	except Exception as e:
		warn.warn(str(e))
		warn.warn("Failed to create valid file")
		ipdb.set_trace()
	

	# ========== Create a list of temp file for deletion ==========
	tmp = [fO1, fO2]
	
	# ========== look for site matches ==========
	if not SiteInfo is None:
		sitematch(fnames[0], fnames[1], dst1, dst2, ds, var,SiteInfo)
	
	# ========== delete the temp file ==========
	for fn in tmp:
		os.remove(fn)
	print("temp files removed")
	print(".nc file created for %d %03d. File took %s" % (year, daynum, str(t1-t0)))


def sitematch(fn1, fn2, dst1, dst2, ds, var,SiteInfo):
	ds1 = xr.open_dataset(fn1)
	ds2 = xr.open_dataset(fn2)

	for index, row in SiteInfo.iterrows():
		values = ([
			ds1[var].sel({"latitude":row.lat, "longitude":row.lon}, method="nearest"),
			dst1[var].sel({"latitude":row.lat, "longitude":row.lon}, method="nearest"),
			ds2[var].sel({"latitude":row.lat, "longitude":row.lon}, method="nearest"),
			dst2[var].sel({"latitude":row.lat, "longitude":row.lon}, method="nearest"),
			ds['ndvi'].sel({"latitude":row.lat, "longitude":row.lon}, method="nearest")
			])
		for num in range(0, 4, 2):
			# print(num)
			if values[num].values == values[num+1].values:
				# print("sucess: both values the same")
				pass
			elif all(np.isnan([values[num].values, values[num+1].values])):
				# print("sucess: both values are NAN")
				pass
			else:
				warn.warn("The values do not match for sn: %d" % row.sn)
				# print(row.sn)
				# print(values[num], values[num+1])
				# print(values[num].longitude, values[num+1].longitude)

		if all([~np.isnan(val.values) for val in values]):
			warn.warn("Four valid values exist")
			print(row.sn)
			print(values[num], values[num+1])
			ipdb.set_trace()


def datefixer(year, daynum):
	"""
	Opens a netcdf file and fixes the data, then save a new file and returns
	the save file name
	args:
		year: int
			the year being tested
	return
		daynum: int
			the day of the year
	"""


	# ========== create the new dates ==========

	# +++++ set up the list of dates +++++
	dates = OrderedDict()
	# compute the date then turn it into a datetime
	tm = [dt.datetime.combine((dt.date(year,1,1) + dt.timedelta(daynum)), dt.datetime.min.time())] 
	dates["time"] = pd.to_datetime(tm)

	dates["calendar"] = 'standard'
	dates["units"]    = 'days since 1900-01-01 00:00'
	
	dates["CFTime"]   = date2num(
		tm, calendar=dates["calendar"], units=dates["units"])

	return dates


def GlobalAttributes(ds, fO1, fO2, fnout):
	"""
	Creates the global attributes for the netcdf file that is being written
	these attributes come from :
	https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/metadata/DataDiscoveryAttConvention.html
	args
		ds: xarray ds
			Dataset containing the infomation im intepereting
		fO1: str
			filename 1
		fO2: str
			filename 2
		fnout: str
			filename out 
	returns:
		attributes 	Ordered Dictionary cantaining the attribute infomation
	"""
	# ========== Create the ordered dictionary ==========
	attr = ds.attrs

	# ========== Delete the old metadata ==========
	for ky in ['StructMetadata_0', 'OldStructMetadata_0', 'OldCoreMetadata_0', 'OldArchiveMetadata_0', 'coremetadata']:
		del attr[ky]

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["FileName"]           = fnout
	attr["title"]               = "NDVI"
	attr["summary"]             = "Reprocessed GIMMS31v1.1 NDVI" 
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created by combining %s and %s using %s (%s):%s by %s" % (
		str(pd.Timestamp.now()), fO1, fO2, __title__, __file__, __version__, __author__)
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

def DAbuilder(Val, dates, lat, lon, ky, long_name, fill_val):
	"""
	Args: 
		Val: array
			np array for shape time, lat, lon with values
		ds: xarray ds
			ds being processsed
		dates: dict
			dictionary of the dataset infomation
		ky: str
			short name
		long_name: str
			the CF longname
		fill_value:
			fill value of the array

	"""
	# ========== Create the xr DA ==========
	try:
		# build xarray dataset
		DA=xr.DataArray(Val,
			dims = ['time', 'latitude', 'longitude'], 
			coords = {'time': dates["CFTime"],'latitude': lat, 'longitude': lon},
			attrs = ({
				'_FillValue':fill_val, #9.96921e+36
				'units'     :"1",
				'standard_name':ky,
				'long_name':long_name,
				'valid_range': [bn.nanmin(Val), np.ceil(np.nanmax(Val))],
				'origname':'/MODIS_Grid_16DAY_250m_500m_VI_7/250m 16 days NDVI'
				}),
		)

		DA.longitude.attrs['units'] = 'degrees_east'
		DA.latitude.attrs['units']  = 'degrees_north'
		DA.time.attrs["calendar"]   = dates["calendar"]
		DA.time.attrs["units"]      = dates["units"]
		
		encoding = ({'shuffle':True, 
			'zlib':True,
			'complevel':6})
	
		return DA, encoding
	except Exception as e:
		warn.warn("Code failed with: \n %s \n Going Interactive" % e)
		ipdb.set_trace()
		raise e


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
	RFinfo = pd.DataFrame(info)
	# ipdb.set_trace()
	# RFinfo["RF17"].replace(0.0, "AR", inplace=True)
	# RFinfo["RF17"].replace(1.0, "RF", inplace=True)
	# RFinfo["RF17"].replace(2.0, "IR", inplace=True)
	# RFinfo["YearsPostFire"] = 2017.0 - RFinfo.fireyear
	return RFinfo



if __name__ == '__main__':
	main()