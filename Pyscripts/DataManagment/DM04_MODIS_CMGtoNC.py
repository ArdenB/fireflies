

"""
Script to process the HDF5 files and convert them to netcdf 

"""
#==============================================================================

__title__ = "Site Vegetation data"
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
import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
from scipy import stats
import xarray as xr
from numba import jit
import bottleneck as bn
import scipy as sp
import glob
from dask.diagnostics import ProgressBar
import myfunctions.PlotFunctions as pf
import myfunctions.corefunctions as cf
from netCDF4 import Dataset, num2date, date2num 
import shutil

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

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

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================
def main():

	force = False
	# ========== loop over the sensors and start years
	for sen, yrst, dsn in zip(["terra", "aqua"], [2000, 2002], ["MOD13C1", "MYD13C1"]):
		# ========== Make the path and tmp folder ==========
		path = "/media/ubuntu/Seagate Backup Plus Drive/Data51/NDVI/5.MODIS/%s/" % sen

		if sen == "aqua":
			path += "5km/"
		tmp  = path + "tmp/"
		cf.pymkdir(tmp)
		prop = path+"processed/"
		cf.pymkdir(prop)
		

		# ========== Set out the file types ==========
		nclist = ["complete", "monmax", "anmax"]
		fncom = [prop + 'MODIS_%s_%s_5kmCMG_%s.nc' % (sen, dsn, ty) for ty in nclist]
		# ipdb.set_trace()
		if all([os.path.isfile(fn) for fn in fncom]) and not force:
			# ds   = xr.open_dataset(fncom)
			ftmp   = glob.glob("%s%s.A*.*.nc" % (path, dsn))
			fmmtmp = glob.glob("%s/monmax/*.nc" % (path))
			famtmp = glob.glob("%s/anmax/*.nc" % (path))
			# ipdb.set_trace()
		else:
			# ========== Make a list of the temp files ==========
			ftmp   = []
			fmmtmp = []
			famtmp = []

			# ========== Loop over each year ==========
			for year in range(yrst, 2020):
				# ========== get the list of files in a given year ==========
				files = glob.glob("%s%s.A%d*.*.hdf" % (path, dsn, year))
				fctmp = [filefix(sen, fname, tmp, year, force) for fname in files]
				ftmp += fctmp	

				dsin = xr.open_mfdataset(fctmp)

				# ========== Fix the datte issues ==========
				dsin = dsin.reindex(time=sorted(dsin.time.values))

				mm, am = dsresample(dsin, sen, dsn, tmp, year, force)
				fmmtmp.append(mm)
				famtmp.append(am)

			# ========== Loop over the configs ========== 
			for ty, fnl, fn in zip(nclist, [ftmp, fmmtmp, famtmp], fncom):
				if not os.path.isfile(fn) or force:
					print("stacking the %s MODIS %s data: " % (ty, sen), pd.Timestamp.now())
					# ========== Create a new multifile dataset ========== 
					ds = xr.open_mfdataset(fnl)

					# ========== Fix the datte issues ==========
					ds = ds.reindex(time=sorted(ds.time.values))

					# # ========== Slice off the unnessary data ========== 
					if ty == "anmax":
						ds = ds.sel(time=slice(None,"2018-12-31")) 
					
					dts = datefixer(pd.to_datetime(ds.time.values).to_pydatetime())
					ds["time"] = dts["CFTime"]
					ds.time.attrs["calendar"] = dts["calendar"]
					ds.time.attrs["units"]    = dts["units"]

					# ========== Create the encoding ==========
					encoding = OrderedDict()
					encoding["ndvi"] = ({
						'shuffle':True, 
						'zlib':True,
						'complevel':5})

					# ========== Save the file using dask delayed ==========
					delayed_obj = ds.to_netcdf(
						prop + 'MODIS_%s_%s_5kmCMG_%s.nc' % (sen, dsn, ty), 
						format         = 'NETCDF4', 
						unlimited_dims = ["time"],
						encoding=encoding,
						compute=False)

					with ProgressBar():
						results = delayed_obj.compute()

					# =========== close the dataset ==========
					ds.close()
					ds = None
		
		print("Removing temp files")			
		for flist, fn in zip([ftmp, fmmtmp, famtmp], fncom):

			# ========== make sure the data can be loaded ==========
			ds = xr.open_dataset(fn)
			# ipdb.set_trace()
			# =========== cleanup the files ==========
			for file in flist: 
				os.remove(file)

		# ============  remove the parts ===========
		shutil.rmtree(tmp)
	# # ========== work out the date ==========
	# pd.to_datetime("2000049", format='%Y%j') 


#==============================================================================

def dsresample(ds, sen, dsn, tmppath, year, force):
	fnamesout = []
	# ========== Calculate the monthly and Annual max ========== 
	for period, tname in zip(["monmax", "anmax"], ["1MS", "1Y"]):
		tmpp = tmppath + "%s/" % period
		cf.pymkdir(tmpp)
		ftout = tmpp + 'MODIS_%s_%s_5kmCMG_%s_%d.nc' % (sen, dsn, period, year)
		if not os.path.isfile(ftout) or force:
			print("Resampling the %d MODIS %s data to %s. " % (year, sen, period), pd.Timestamp.now())
			# ========== Resample the data ========== 
			tmax = ds.resample(time=tname).max()
			# ========== Add attributs ========== 
			tmax.attrs = ds.attrs
			tmax.ndvi.attrs = ds.ndvi.attrs 
			dates = datefixer(pd.to_datetime(tmax.time.values).to_pydatetime())
			tmax["time"] = dates["CFTime"]
			tmax.time.attrs["calendar"]   = dates["calendar"]
			tmax.time.attrs["units"]      = dates["units"]
			# tmax.longitude.attrs = ds.longitude.attrs
			# tmax.latitude.attrs = ds.latitude.attrs

			# ========== Create the encoding ==========
			encoding = OrderedDict()
			encoding["ndvi"] = ({
				'shuffle':True, 
				'zlib':True,
				'complevel':5})

			# ========== Save the file using dask delayed ==========
			delayed_obj = tmax.to_netcdf(
				ftout, 
				format         = 'NETCDF4', 
				unlimited_dims = ["time"],
				encoding=encoding,
				compute=False)
			
			with ProgressBar():
				results = delayed_obj.compute()	
		fnamesout.append(ftout)
	return fnamesout

def filefix(sen, fname, tmp, year, force):
	"""
	Function to make a temp file for a given location
	args:
		sen:		str
			sname of the mODIS sensor
		fname:		str
			name of the file to be opened 
		tmp:		str
			path of the tempoary folder
		year:		int
			the year

	returns:
		ftmp:		str
			name of the tempary file 

	"""
	def _filename(fname):
		"""
		Function that takes the filename and processes key infomation
		argsL
			fname:		str
				name of the file to be opened 
		returns:
			tmpfn:		str
				tempary file name to write first netcdf
			fdate: 		pd time
				the date of the hdf file
		"""
		# ========== get just the uniwue file name ==========
		fln = fname.split("/")[-1]

		# ========== Make a tmp file name ==========
		tmpfn = fln[:-3]+"tmp.nc"
		
		# ========== Make the date ==========
		tm    = [dt.datetime.strptime(fln.split(".")[1][1:], '%Y%j')]
		fdate = datefixer(tm)

		return fln, tmpfn, fdate
	
	fln, tmpfn, fdate = _filename(fname)
	fnout = tmp+tmpfn

	if os.path.isfile(fnout) and not force:
		return fnout
		# print("A valid file already exists for")

	# ========== open the file ==========
	ds = xr.open_dataset(fname, engine="pynio") 

	# ========== fix the lon and lats ==========
	ds = ds.rename({"XDim_MODIS_Grid_16Day_VI_CMG":"longitude", "YDim_MODIS_Grid_16Day_VI_CMG":"latitude"}) 
	xv = np.arange(-179.975, 180.025, 0.05) 
	yv = np.arange(89.975, -90.025, -0.05)
	ds["longitude"] = xv
	ds["latitude"]  = yv
	ds = ds.expand_dims({'time':fdate["CFTime"]})
	ds["time"] = fdate["CFTime"]


	# ========== capture the global attributes ==========
	global_attrs = GlobalAttributes(ds, fnout, sen)

	# ========== Pull out and scale the NDVI ==========
	try:
		DA = ds["CMG_0_05_Deg_16_days_NDVI"].rename("ndvi").copy()
		DA.values *= 1e-8
	except Exception as e:
		warn.warn("somthing is broken here")
		warn.warn(str(e))
		ipdb.set_trace()
	
	# ========== Set up the relevant attrs ==========
	DA.attrs['valid_range']   = [(DA.min().values), np.ceil(DA.max().values)]    
	DA.attrs['units']         ="1"
	DA.attrs['standard_name'] ="ndvi"

	DA.longitude.attrs['units'] = 'degrees_east'
	DA.latitude.attrs['units']  = 'degrees_north'
	DA.time.attrs["calendar"]   = fdate["calendar"]
	DA.time.attrs["units"]      = fdate["units"]

	# ========== Create the encoding ==========
	encoding = OrderedDict()
	encoding["ndvi"] = ({
		'shuffle':True, 
		# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
		'zlib':True,
		'complevel':5})
	
	# ========== Pull out and scale the NDVI ==========
	ds_out = xr.Dataset({"ndvi":DA}, attrs= global_attrs)
	print("Starting write of data")
	ds_out.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		encoding       = encoding,
		unlimited_dims = ["time"])
	print(".nc file created for MODIS %s %d" % (sen, year))
	# fixed.append(fnout)

	return fnout
	# sys.exit()

#==============================================================================
def GlobalAttributes(ds, fnout, sen):
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
	attr["FileName"]            = fnout
	attr["title"]               = "NDVI"
	attr["summary"]             = "Reprocessed MODIS %s CMG NDVI" % sen
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
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

def datefixer(tm):
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
	# year = ds.Year

	# +++++ set up the list of dates +++++
	dates = OrderedDict()
	# tm = [dt.datetime(int(year) , int(np.floor(tm)), int(tm%1*30+1)) for tm in ds.time]
	dates["time"] = pd.to_datetime(tm)

	dates["calendar"] = 'standard'
	dates["units"]    = 'days since 1900-01-01 00:00'

	
	dates["CFTime"]   = date2num(
		tm, calendar=dates["calendar"], units=dates["units"])


	return dates
#==============================================================================
if __name__ == '__main__':
	main()