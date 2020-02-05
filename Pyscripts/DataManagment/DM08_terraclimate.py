"""
TERRACLIMATE data management


"""
#==============================================================================

__title__ = "TERRACLIMATE fixer"
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
# import wget
import glob
# import tarfile  
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
	clpath = "/srv/ccrc/data51/z3466821/Input_data/TerraClimate"
	ppt    = xr.open_mfdataset(
		clpath+"/TerraClimate_ppt_*.nc").drop(["crs", "station_influence"]).rename(
		{"lat":"latitude", "lon":"longitude"})

	ppd_sel = ppt.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
	ipdb.set_trace()


if __name__ == '__main__':
	main()