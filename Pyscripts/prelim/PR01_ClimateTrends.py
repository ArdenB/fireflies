
"""
Prelim script for looking at netcdf files and producing some trends
Broken into three parts
	Part 1 looks at vegetation trends
	Part 2 looks at climate trends
	Part 3 interigates the field data
"""
#==============================================================================

__title__ = "Vegetation and climate trends"
__author__ = "Arden Burrell"
__version__ = "v1.0(28.01.2019)"
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
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
# Import debugging packages 
import ipdb
# +++++ Import my packages +++++
# import MyModules.CoreFunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================
def main(args):
	# ========== Get the key infomation from the args ==========
	fdpath = args.fdpath 
	
	# ========== Pull out info needed from the field data ==========
	RFinfo = Field_data(fdpath)

	# ========== Compare the overall site infomation ==========
	VI_trend(RFinfo)

	ipdb.set_trace()

#==============================================================================
def VI_trend(RFinfo):
	"""
	This is a function for looking for any correspondense between 
	sites and observed vi trends

	Note:
		This is a niave approach. I'm ignoring key infomation about the
		timing of the fire, the intensity of the fire etc etc. This is more
		of a proof of concept 
	"""

	warn.warn(
		'''
		This is currently only in alpha testing form
		i'm going to using a simple trend test without
		any considderation of significance.
		''')

	# ========== Load in the trend data using xarray ==========
	ncin = "./data/veg/COPERN/NDVI_anmax_Russia_cdoregres.nc"
	ds   = xr.open_dataset(ncin)

	ipdb.set_trace()


def Field_data(fdpath):
	"""
	# Aim of this function is to look at the field data a bit

	To start it just opens the file and returns the lats and longs 
	i can then use these to look up netcdf fils
	"""
	fsum = pd.read_csv("./data/field/RF_catsum.csv")
	fsum.sort_values(by=["sn"],inplace=True) 
	fcut = fsum[fsum.sn<64]
	fd18 = pd.read_csv(fdpath)
	fd17 = pd.read_csv("./data/field/2017data/siteDescriptions.csv")
	# lat = fd.lat 
	# lon = fd.lon

	info = OrderedDict()
	info["sn"] = fd17["site number"]
	info["lat"] = fd17.strtY
	info["lon"] = fd17.strtX
	# for sn in info['sn']:
	# 	ipdb.set_trace()
	def test(val):
		try:
			return float(val)
		except Exception as e:
			print(e)
			# ipdb.set_trace()
			return np.NAN
		# if np.isnan(val):

	info["sden17"] = [test(fcut[fcut.sn == sn]["sDens2017Ls"].values) for sn in info['sn']]

	# info["sden17"] = fcut.sDens2017Ls
	info["RF17"] = [test(fcut[fcut.sn == sn]["RF2017"].values) for sn in info['sn']]
	RFinfo = pd.DataFrame(info)
	return RFinfo
#==============================================================================

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


