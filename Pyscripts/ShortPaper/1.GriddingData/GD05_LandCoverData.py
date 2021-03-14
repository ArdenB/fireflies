"""
Script goal, 

Open land cover data and build a simple cover map

"""
#==============================================================================

__title__ = "LandCover"
__author__ = "Arden Burrell"
__version__ = "v1.0(12.03.2021)"
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
# import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import shutil
import time
from dask.diagnostics import ProgressBar
import rasterio

from collections import OrderedDict
# from scipy import stats
# from numba import jit


# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl

import palettable 

# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
# import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	# ========== Setup the broad infomation
	region = "SIBERIA"
	box = [-10.0, 180, 40, 70]

	# ========== Load in the different data from glc ==========
	path   = "./data/LandCover/"
	# years     = [2000, 2010]
	legendfn  = [f"{path}glc2000_v1_1/Tiff/Global_Legend.csv", f"{path}gez2010/Lookup.csv", f"{path}Dinerstein_Aggregated/Lookup.csv", f"{path}Dinerstein_Aggregated/Lookup.csv"]
	# geotiffn  = [f"{path}glc2000_v1_1/Tiff/glc2000_v1_1.tif", f"{path}gez2010/OUTPUT.tif", f"{path}gez2010/IsBorealV3.tif"]

	Down = ["MODIS", "esacci", "COPERN_BA"]
	res     = ["MODIS", ]#"TerraClimate"]#,  "GFED","COPERN_BA",  "esacci", ]#
	force = True
	for dsres in res:
		fnout = f"{path}Regridded_forestzone_{dsres}.nc"
		if os.path.isfile(fnout) and not force:
			print(f"{dsres} has an existing file")
			continue
		else:
			print(dsres)
		dataname  = ["LandCover", "GlobalEcologicalZones", "DinersteinRegions", "BorealMask"]
		if dsres in Down:
			datares = "MODIS"
		else:
			datares = dsres
		geotiffn  = [f"{path}glc2000_v1_1/Tiff/glc2000_v1_1.tif", f"{path}Dinerstein_Aggregated/Masks/Boreal_climatic_{datares}.tif", f"{path}Dinerstein_Aggregated/Masks/BorealEco_2017_{datares}.tif", f"{path}Dinerstein_Aggregated/Masks/Boreal_buf_{datares}.tif"]
		mskfn = "./data/masks/broad/Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsres)
		ds_msk = xr.open_dataset(mskfn).sel(dict(latitude=slice(box[3], box[2]), longitude=slice(box[0], box[1]))).chunk()
		mask   = ds_msk.datamask

		# out_dic = OrderedDict()
		outlist = []
		key_dic = OrderedDict()
		for dsnx, legfn, tiffn in zip(dataname, legendfn, geotiffn):
			print(dsnx)
			# +++++ open the dataarray +++++
			key_dic[dsnx] = pd.read_csv(legfn)
			# da           = xr.open_rasterio(tiffn, chunks=10).transpose("y", "x", "band").rename({"x":"longitude", "y":"latitude", "band":"time"}).sel(dict(latitude=slice(box[3], box[2]), longitude=slice(box[0], box[1])))
			# da["time"]   = [pd.Timestamp("2018-12-31")]
			# if da.longitude.shape > ds_msk.longitude.shape:
			# 	print(da.latitude.shape[0], ds_msk.latitude.shape[0])
			# 	print ("Coarsnening data started at: ", pd.Timestamp.now())
			# 	# breakpoint()
			# 	# Coarsen/ downscale 
			# 	latscale = int(da.latitude.shape[0] / ds_msk.latitude.shape[0])
			# 	lonscale = int(da.longitude.shape[0] / ds_msk.longitude.shape[0])

			# 	da = da.coarsen(latitude=latscale, longitude=lonscale, boundary ="pad").median()
			# 	da = da.round()

			# da = da.reindex_like(mask, method="nearest")
			# delay =  xr.Dataset({dsnx:da}).to_netcdf(f"/tmp/{dsres}_{dsnx}.nc", format = 'NETCDF4', unlimited_dims = ["time"], compute=False)
			# print(f"Creating temp netcdf for {dsres} {dsnx} at: {pd.Timestamp.now()}")
			# with ProgressBar():
			# 	delay.compute()
			# out_dic[dsnx] 
			outlist.append(f"/tmp/{dsres}_{dsnx}.nc")
			da = None
		# breakpoint()
		# ========== get the FAO climate zones ==========
		# ds     = xr.Dataset(out_dic)
		ds     = xr.open_mfdataset(outlist).transpose('time', 'latitude', 'longitude')

		GlobalAttributes(ds, dsres, fnameout=fnout)

		delayed_obj = ds.to_netcdf(fnout, format = 'NETCDF4', unlimited_dims = ["time"], compute=False)
		print("Starting write of %s data at" % name, pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()

		print(f"{dsres} completed at: {pd.Timestamp.now()}")


		# breakpoint()

	breakpoint()



	for dsn in ["TerraClimate","GFED", "MODIS", "esacci", "COPERN_BA"]:
		print(dsn)
		mskfn = "./data/masks/broad/Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (region, dsn)
		ds_msk = xr.open_dataset(mskfn).sel(dict(latitude=slice(box[3], box[2]), longitude=slice(box[0], box[1])))
		# ds_mod =  ds.reindex_like(ds_msk, method="nearest")
		# mask   = ds_msk.datamask
		# # mask   = ds_msk.datamask.reindex_like(ds, method="nearest")

		# # boreal mask
		# title = "FAO Boreal Zone"
		# plotmaker(ds_mod.Boreal, title, mask)

		# # Tree cover mask
		# title = "Needle Leaf Tree species"
		# plotmaker(((ds_mod.LandCover == 4)+(ds_mod.LandCover == 5)), title, mask)

		# title = "Needle Leaf and mixed fores"
		# plotmaker(((ds_mod.LandCover == 6)+(ds_mod.LandCover == 4)+(ds_mod.LandCover == 5)), title, mask)

		# title = "Broadleaf forest"
		# plotmaker(((ds_mod.LandCover == 1)+(ds_mod.LandCover == 2)+(ds_mod.LandCover == 3)), title, mask)
		breakpoint()




	breakpoint()


#==============================================================================
# def _lookupkeys():
# 	dataname  = ["LandCover", "GlobalEcologicalZones", "DinersteinRegions", "BorealMask"]
# 	legendfn  = ([f"{path}glc2000_v1_1/Tiff/Global_Legend.csv", f"{path}gez2010/Lookup.csv", f"{path}Dinerstein_Aggregated/Lookup.csv", f"{path}Dinerstein_Aggregated/Lookup.csv"])
# 	for  nm, lfn in zip(dataname, legendfn)

def GlobalAttributes(ds, dsn, fnameout=""):
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
	attr["FileName"]            = fnameout
	attr["title"]               = "Datamasks"
	attr["summary"]             = "BorealForestCovermaks_%sData" % (dsn)
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. FRI caluculated using %s data" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, dsn)
	
	if not ds is None:
		attr["history"]            += ds.history

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "Woodwell"
	attr["date_created"]        = str(pd.Timestamp.now())
	
	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	# attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	# attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	return attr	

def _mode(da):
	vals = sp.stats.mode(da, axis=None, nan_policy="omit")

	return vals[0][0]




def plotmaker(ds_in, title, mask):

	# breakpoint()

	latiMid=np.mean([70.0, 40.0])
	longMid=np.mean([-10.0, 180.0])

	fig, ax = plt.subplots(1, 1, figsize=(20,12), subplot_kw={'projection': ccrs.Orthographic(longMid, latiMid)})
	ds_in.where(mask==1).plot(transform=ccrs.PlateCarree(), ax=ax)
	coast = cpf.GSHHSFeature(scale="intermediate")
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
	ax.add_feature(coast, zorder=101, alpha=0.5)
	# coast_50m = cpf.GSHHSFeature(scale="high")
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.set_title(f"{title}")
	plt.show()





#==============================================================================

if __name__ == '__main__':
	main()