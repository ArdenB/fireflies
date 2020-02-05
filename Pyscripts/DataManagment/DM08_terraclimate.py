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
	for var in ["tmaan", "ppt"]:
		if var == "ppt":
			ppt    = xr.open_mfdataset(
				clpath+"/TerraClimate_%s_*.nc" % var).drop(["crs", "station_influence"]).rename(
				{"lat":"latitude", "lon":"longitude"})

			ppt_sel = ppt.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
		else:
			ppt    = xr.open_mfdataset(
				clpath+"/TerraClimate_SIBERIA_%s_*.nc" % var).drop(["crs", "station_influence"]).rename(
				{"lat":"latitude", "lon":"longitude"})

		ppt_sel.attrs["history"]  = "%s: Netcdf file created using %s (%s):%s by %s from terraclimate data" % (
			str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
		


		ppt_sel.attrs["creator_name"]        = __author__
		ppt_sel.attrs["creator_url"]         = "ardenburrell.com"
		ppt_sel.attrs["creator_email"]       = __email__
		ppt_sel.attrs["Institution"]         = "University of Leicester"
		ppt_sel.attrs["date_created"]        = str(pd.Timestamp.now())

		fnout = clpath + "/TerraClimate_SIBERIA_%s_1958to2018.nc" % var

		encoding =  ({var:{'shuffle':True,'zlib':True,'complevel':5}})
		delayed_obj = ppt_sel.to_netcdf(fnout, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of %s data at:" % var, pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()
		ipdb.set_trace()


if __name__ == '__main__':
	main()