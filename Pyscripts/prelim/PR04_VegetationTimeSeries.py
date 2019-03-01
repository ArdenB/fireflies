
"""
Prelim script for looking at netcdf files and producing some trends
Broken into three parts
	Part 1 pull out the NDVI from the relevant sites
"""
#==============================================================================

__title__ = "Vegetation time series"
__author__ = "Arden Burrell"
__version__ = "v1.0(27.02.2019)"
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
from netCDF4 import Dataset, num2date 
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
# Import debugging packages 
import ipdb
# +++++ Import my packages +++++
# import MyModules.CoreFunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================
def main(args):
	# ========== Get the key infomation from the args ==========
	# fdpath = args.fdpath 
	warn.warn(
		'''
		This is currently only in alpha testing form
		I will replace all the variables and infomation 
		for experiments in a dataframe so i can look at
		different aproaches
		''')

	# ========== set the filnames ==========
	data= OrderedDict()
	data["GIMMS"] = ({
		"fname":"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1981to2017_mergetime_compressed.nc",
		'var':"ndvi", "gridres":"8km", "region":"Global", "timestep":"16day", 
		"start":1981, "end":2017
		})
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_MonthlyMax_1999to2018_at_1kmRUSSIA.nc",
		'var':"NDVI", "gridres":"1km", "region":"RUSSIA", "timestep":"Monthly",
		"start":1999, "end":2018
		})


	for syear in [2017, 2018]:
		# ========== Pull out info needed from the field data ==========
		SiteInfo = Field_data(year = syear)

		# ========== Loop over each of the included vegetation datasets ==========
		for dt in data:
			# ========== Get the vegetation values ==========
			VIdata, MNdata, ANdata = NDVIpuller(
				data[dt]["fname"], data[dt]["var"], SiteInfo, data[dt]["timestep"])


			# ========== Save the data out ==========
			outfile = ("./data/field/exportedNDVI/NDVI_%dsites_%s_%dto%d_%s_"
				% (syear, dt,data[dt]["start"], data[dt]["end"], data[dt]["gridres"]))

			if not data[dt]["timestep"] == "Monthly":
				VIdata.to_csv(outfile+"complete.csv", header=True)	
			MNdata.to_csv(outfile+"MonthlyMax.csv", header=True)
			ANdata.to_csv(outfile+"AnnualMax.csv", header=True)			
	
	ipdb.set_trace()

#==============================================================================



def NDVIpuller(fname, var, SiteInfo, timestep):
	"""
	args:
		fname: str
			name of the netcdf file to be opened
		SiteInfo: df
			dataframe with site details
	"""
	# ========== Load the file ==========
	ds   = xr.open_dataset(fname)

	# ========== Get the vegetation data from the netcdf ==========
	VIdata = [] # All values
	MNdata = [] # Monthly Max data
	ANdata = [] # Annual Max values
	for index, row in SiteInfo.iterrows():
		try:
			array = ds[var].sel({"latitude":row.lat, "longitude":row.lon},	method="nearest")
		except ValueError:
			array = ds[var].sel({"lat":row.lat, "lon":row.lon},	method="nearest")
		# +++++ append the complete series +++++
		VIdata.append(pd.Series(array.values, index=pd.to_datetime(ds.time.values)))
		
		# +++++ append the monthly max +++++
		if timestep == "Monthly":
			MNdata.append(pd.Series(array.values, index=pd.to_datetime(ds.time.values)))	
		else:
			mon = array.resample(time="1M").max()
			MNdata.append(pd.Series(mon.values, index=pd.to_datetime(mon['time'].values)))
		
		# +++++ append the annual max +++++
		ann = array.groupby('time.year').max()
		tm = [dt.datetime(int(year) , 6, 30) for year in ann.year]
		ANdata.append(pd.Series(ann.values, index= pd.to_datetime(tm)))
	

	# ========== COnvert to DF ==========
	dfc = pd.DataFrame(VIdata, index=SiteInfo.sn)
	dfm = pd.DataFrame(MNdata, index=SiteInfo.sn)
	dfa = pd.DataFrame(ANdata, index=SiteInfo.sn)
	return dfc, dfm, dfa


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


#==============================================================================
if __name__ == '__main__':
	# ========== Set the args Description ==========
	description='Passed argumants'
	parser = argparse.ArgumentParser(description=description)
	
	# ========== Add additional arguments ==========
	parser.add_argument(
		'--fdpath', type=str, action="store", 
		default="./data/field/2018data/siteDescriptions18.csv", 
		help='The path to the field results')
	parser.add_argument(
		'--path2', type=str, default=None, 
		help='The path to the second runs results')
	# parser.add_argument(
	# 	"--gparts", type=int, default=None,   
	# 	help="the max original partnumber")
	args = parser.parse_args() 
	
	# ========== Call the main function ==========
	main(args)


