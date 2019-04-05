"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

"""
#==============================================================================

__title__ = "GEE NDVI extraction"
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
# import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
# from scipy import stats
# import xarray as xr
# from numba import jit
# import bottleneck as bn
import scipy as sp
import glob

# Import the Earth Engine Python Package
import ee
import ee.mapclient

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
# import matplotlib.colors as mpc
# import matplotlib as mpl
# import palettable 
# import seaborn as sns
# import cartopy.crs as ccrs
# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# # Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
# print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	# ========== Initialize the Earth Engine object ==========
	ee.Initialize()

	# ========== Define the image collection ==========
	# landsat 8 Surface reflectance 
	# collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
	data = OrderedDict()
	# landsat 8 NDVI
	data["LANDSAT8"] = ({
		"NDVI":ee.ImageCollection("LANDSAT/LC08/C01/T1_8DAY_NDVI").select("NDVI"),
		"start":2013, "end":2019, "gridres":"30m", "region":"global", "timestep":"8day",
		"resolution":30, "scalefactor":1.0
		})
	data["LANDSAT7"] = ({
		"NDVI":ee.ImageCollection("LANDSAT/LE07/C01/T1_8DAY_NDVI").select("NDVI"),
		"start":1999, "end":2018, "gridres":"30m", "region":"global", "timestep":"8day",
		"resolution":30, "scalefactor":1.0
		})
	data["LANDSAT5"] = ({
		"NDVI":ee.ImageCollection('LANDSAT/LT05/C01/T1_8DAY_NDVI').select("NDVI"),
		"start":1994, "end":2012, "gridres":"60m", "region":"global", "timestep":"8day",
		"resolution":60, "scalefactor":1.0,
		"link":"https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C01_T1_8DAY_NDVI"
		})
	data["LANDSAT4"] = ({
		"NDVI":ee.ImageCollection("LANDSAT/LT04/C01/T1_8DAY_NDVI").select("NDVI"),
		"start":1982, "end":1993, "gridres":"60m", "region":"global", "timestep":"8day",
		"resolution":60, "scalefactor":1.0
		})
	data["MODISterra"] = ({
		"NDVI":ee.ImageCollection("MODIS/006/MOD13Q1").select("NDVI"),
		"start":2000, "end":2018, "gridres":"250m", "region":"global", "timestep":"16day",
		"resolution":250, "scalefactor":0.0001, 
		"link":"https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD13A1"
		})
	data["VIIRS"] = ({
		"NDVI":ee.ImageCollection("NOAA/VIIRS/001/VNP13A1").select("NDVI"),
		"start":2012, "end":2018, "gridres":"5000m", "region":"global", "timestep":"16day",
		"resolution":500, "scalefactor":0.0001, 
		"link":"https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP13A1"
		})

	# ========== Load the Site Data ==========
	syear = 2018
	SiteInfo = Field_data(year=syear)

	# ========== Loop over the datasets ==========
	for dsn in data:
		# ========== Create the data lists ==========
		VIdata = [] # All values
		MNdata = [] # Monthly Max data
		ANdata = [] # Annual Max values

		t0     = pd.Timestamp.now()
		print("starting %s at:" % dsn, t0)
		# ========== Loop over the sites ==========
		for index, row in SiteInfo.iterrows():
			string = ("\rFetching %s from the google earth engine: site: %d of %d" % 
										(dsn, index, SiteInfo.shape[0]))
			sys.stdout.write(string)
			sys.stdout.flush()

			# ========== Create a geometery point at the site ==========
			geom = ee.Geometry.Point([row.lon, row.lat])
			result = data[dsn]["NDVI"].getRegion(geom, data[dsn]["resolution"]).getInfo()
			
			# ========== build a pandas dataframe ==========
			# ipdb.set_trace()
			df = pd.DataFrame(result[1:], columns=result[0])
			
			# ========== Change the index to the dates ==========
			df.index = [pd.Timestamp(t*1000000) for t in df.time]
			df.NDVI *= data[dsn]["scalefactor"]
			
			# ========== Calculate the Grouped data ==========
			VIdata.append(df.NDVI)
			MNdata.append(df.NDVI.resample("1M").max())
			ANdata.append(df.NDVI.resample("1Y").max())
		
		# ========== Save the data out ==========
		outfile = ("./data/field/exportedNDVI/NDVI_%dsites_%s_%dto%d_%s_"
			% (syear, dsn, data[dsn]["start"], data[dsn]["end"], data[dsn]["gridres"])) 

		dfc = pd.DataFrame(VIdata, index=SiteInfo.index)
		dfm = pd.DataFrame(MNdata, index=SiteInfo.index)
		dfa = pd.DataFrame(ANdata, index=SiteInfo.index)

		dfc.to_csv(outfile+"complete.csv", header=True)	
		dfm.to_csv(outfile+"MonthlyMax.csv", header=True)
		dfa.to_csv(outfile+"AnnualMax.csv", header=True)	
		print("\n Total time taken to fetch %s values: %s" % (dsn, str(pd.Timestamp.now() - t0)))
	# pass

#==============================================================================

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

#==============================================================================
if __name__ == '__main__':
	main()