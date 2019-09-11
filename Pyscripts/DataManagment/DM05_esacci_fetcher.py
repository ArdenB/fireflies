"""
Script goal, 

Build evaluation maps of GEE data

"""
#==============================================================================

__title__ = "data fetcher"
__author__ = "Arden Burrell"
__version__ = "v1.0(11.09.2019)"
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

import ipdb
import wget
import glob
import tarfile  
import xarray as xr
import numpy as np
import pandas as pd
import warnings as warn
import datetime as dt
from dask.diagnostics import ProgressBar
from collections import OrderedDict
from netCDF4 import Dataset, num2date, date2num 
# import gzip

# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 

#==============================================================================

def main():
	# ========== setup the path ========== 
	path = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/"
	ppath = path+"processed/"
	cf.pymkdir(path)
	cf.pymkdir(path+"tmp/")
	cf.pymkdir(ppath)
	force = False

	for yr in range(2001, 2020):
			# ========== loop over layers ==========		
		for layer in ["JD"]:#, "CL", "LC"]:
			print("Starting %d %s at:" % (yr, layer), pd.Timestamp.now())

			# ========== Check if a file already exists ==========
			ANfn    = ppath + "esacci_FireCCI_%d_burntarea.nc" % yr
			if os.path.isfile(ANfn) and not force:
				ds = xr.open_dataset(ANfn)
				continue
			else:
				ds = _monthlyfile(yr, path, ppath, force, layer, ANfn)
			print("%d BA calculation completed at:" % (yr), pd.Timestamp.now(), " Starting Cleanup")
			

				
			# da_bl[4,10301, 4865].values

		_cleanup(yr, path)
		# ipdb.set_trace()
	ipdb.set_trace()
		

#==============================================================================	

def _cleanup(yr, path):
	"""
	Function to remove alll the unnesssary files
	"""
	fntz = glob.glob(path+"%d*.gz" % yr)
	fntm = glob.glob(path+"tmp/%d*" % yr)
	for fnc in (fntz+fntm):
		os.remove(fnc)


def _monthlyfile(yr, path, ppath, force, layer, ANfn):
	"""
	Function to proccess the monthl;y data into an annual file
	args:
		yr: int
			year
		path:	str
			dir to do the work in
		ppath:	str
			processed path
	"""

	# ========== get the web address ==========
	address =  "ftp://anon-ftp.ceda.ac.uk/neodc/esacci/fire/data/burned_area/MODIS/pixel/v5.1/compressed/%d/" % yr
	
	# ========== list to holf the file names ==========
	ptnames = []

	# ========== loop over the month ==========
	for mn in range(1, 13):
		# ========== Create the file name and check if they need to get downloaded ==========
		fnA  = "%d%02d01-ESACCI-L3S_FIRE-BA-MODIS-AREA_4-fv5.1.tar.gz" % (yr, mn)
		fnE  = "%d%02d01-ESACCI-L3S_FIRE-BA-MODIS-AREA_3-fv5.1.tar.gz" % (yr, mn)
		for fn in [fnA, fnE]:
			filefetcher(fn, address, path)

		# ========== Make the file name and see if it already exists ==========
		ftout  = path+"tmp/%d%02d01_tmp_%s.nc" %(yr, mn, layer)
		if os.path.isfile(ftout):
			ds_testing = xr.open_dataset(ftout)
			ptnames.append(ftout)
			print(ds_testing[layer].shape)
			continue

		# ========== open the components ==========
		fn_XR = glob.glob(path+"tmp/%d%02d*-%s.tif" % (yr, mn, layer))
		renm  = {"band":"time","x":"longitude", "y":"latitude"}
		da_ls = [xr.open_rasterio(fnr).rename(renm).sel(
			dict(
				latitude=slice(70.0, 45.0),
				longitude=slice(0.0, 150.0))) for fnr in fn_XR]

		# ========== Merge into a single dataset ==========
		ds_out = xr.Dataset({layer:xr.concat(da_ls, dim="longitude").chunk({"longitude":1000})})#.sortby("latitude", ascending=False)#.transpose("latitude")
		date   = pd.Timestamp("%d-%02d-01" % (yr, mn))
		ds_out["time"] = [date]

		# ========== Save a tempoary netcdf fiel ==========
		ds_out = tempNCmaker(ds_out, ftout, layer, chunks={'longitude': 1000}, skip=False)

		# ========== append the save name ==========
		ptnames.append(ftout)

	# ========== Build annual dataset ==========
	da = xr.open_mfdataset(
		ptnames, concat_dim="time",
		chunks={"time":1, "latitude":1000, 'longitude': 1000})[layer]
	da = da.reindex(latitude=list(reversed(da.latitude)))

	# ========== mask it away ==========
	da_bl = da.where(   da > 0)

	# ========== Aggragate and finalise the da ==========
	dates  = datefixer(yr, 12, 31)
	da_out = da_bl.sum("time")
	da_out = da_out.where(da_out<= 0, 1).rename()
	da_out = attrs_fixer(da_out, dates)

	# ========== Setup the metadata  ==========
	global_attrs = GlobalAttributes(ANfn)
	layers       = OrderedDict()
	layers["BA"] = da_out

	# ========== create the dataset ==========
	ds = xr.Dataset(layers, attrs= global_attrs)
	ds = tempNCmaker(ds, ANfn, "BA", chunks={"latitude":1000, 'longitude': 1000}, pro = "%d Burnt Area"% yr)

	# ========== return the dataset ==========
	return ds

#==============================================================================	

def attrs_fixer(da, dates):
	"""
	Function to fix the metadata of the dataarrays
	args:
		origda:	XR datarray, 
			the original meta data
		da: xa dataarray
			the dataarray to be fixed
		vname: str
			name of the variable
	retunrs
		da: xr dataarray with fixed metadata
	"""
	# ========== Make the DA ==========
	# da.attrs = da.attrs.copy()
	da.attrs['_FillValue']	=-1, #9.96921e+36
	da.attrs['units'     ]	="1",
	da.attrs['standard_name']	="BA"
	da.attrs['long_name'	]	="BurntAreA"
	da.attrs['valid_range']	= [0.0, 1.0]
	da.longitude.attrs['units'] = 'degrees_east'
	da.latitude.attrs['units']  = 'degrees_north'
	da = da.expand_dims("time")
	
	da["time"] = dates["CFTime"]

	da.time.attrs["calendar"]   = dates["calendar"]
	da.time.attrs["units"]      = dates["units"]

	return da

def GlobalAttributes(fname, ds=None):
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
	attr["FileName"]            = fname
	attr["title"]               = "Annual Burnt Area product"
	attr["summary"]             = "FireCCI_Data aggragated into an annual boolean burnt are product"
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s with FireCCI data" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
	
	if not ds is None:
		attr["history"]            += ds.history

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "University of Leicester"
	attr["date_created"]        = str(pd.Timestamp.now())
	
	# ++++++++++ Netcdf Summary infomation ++++++++++ 

	return attr

def tempNCmaker(ds, fntmp, vname, chunks={'longitude': 1000}, skip=False, pro = "tmp"):

	""" Function to save out a tempary netcdf """
	# cf.pymkdir(tmppath)
	
	encoding =  ({vname:{'shuffle':True,'zlib':True,'complevel':5}})
	if not all([skip, os.path.isfile(fntmp)]):
		delayed_obj = ds.to_netcdf(fntmp, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of %s data at" % pro, pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()
	dsout = xr.open_dataset(fntmp, chunks=chunks) 
	return dsout

def filefetcher(fn, address, path):
	"""
	downloads and uncompresses files 
	"""
	url = address+fn
	out = path + fn

	# ========== Download and unxompress the file ========== 
	try:
		# ========== only download new files ========== 
		if not os.path.isfile(out):
			# ========== Download the file ==========
			wget.download(url, out=out)
			# ========== decompress it ==========
			tf  = tarfile.open(out)
			tf.extractall(path=path+"tmp/")

	except Exception as e:
		warn.warn(str(e))
		ipdb.set_trace()
		sys.exit()

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
if __name__ == '__main__':
	main()