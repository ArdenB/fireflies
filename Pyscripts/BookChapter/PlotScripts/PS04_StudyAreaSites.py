"""
Script goal, to produce trends in netcdf files
This script can also be used in P03 if required

"""
#==============================================================================

__title__ = "Study Site map"
__author__ = "Arden Burrell"
__version__ = "v1.0(03.06.2019)"
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
from netCDF4 import Dataset, num2date, date2num 
from scipy import stats
import xarray as xr
# from dask.diagnostics import ProgressBar
# from numba import jit
import bottleneck as bn
import scipy as sp
import glob
from scipy import stats
import statsmodels.stats.multitest as smsM
import myfunctions.PlotFunctions as pf
import myfunctions.corefunctions as cf

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
import cartopy.io.img_tiles as cimgt
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from cartopy.io import PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM3Source, SRTM1Source
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)
#==============================================================================

def main():
	# ========== Get the recuitment status and GPS ==========
	
	syear = 2018
	SiteInfo = Field_data(year=syear)
	formats=[".png", ".pdf"]
	dpi=500
	fpath = "./plots/bookchapter/firstdraft/"
	fname = fpath+ "Zab_%dsites" % syear


	# ========== Load the global basemaps ==========
	# da = xr.open_rasterio("./data/other/BlueMarble/BlueMarbleNG_2004-12-01_rgb_3600x1800.TIFF")  
	
	# ========== Set up the map ==========
	bounds = [101.0, 116.0, 53.0, 50.0]

	stamen_terrain = cimgt.Stamen('terrain-background')
	# shaded_srtm = PostprocessedRasterSource(SRTM3Source(), shade)
	# ipdb.set_trace()
	# da    = da.rename({"y":"latitude", "x":"longitude"})
	# da = da.loc[dict(
	# 	longitude=slice(bounds[0], bounds[1]),
	# 	latitude=slice(bounds[2], bounds[3]))]

	# ipdb.set_trace()
	fig, ax = plt.subplots(1, 1, figsize=(18,6),
		subplot_kw={'projection': ccrs.PlateCarree()}, 
		num=("Map of Sites" ), dpi=dpi)
	# ax.add_raster(shaded_srtm, cmap='Greys')
	ax.set_extent(bounds)
	# ipdb.set_trace()
	ax.add_image(stamen_terrain, 8, cmap='Greys')

	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(cpf.COASTLINE, zorder=101)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	# ax.stock_img()
	gl = ax.gridlines(
		crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, 
		linestyle='--', zorder=105)
	gl.xlabels_top = False
	gl.ylabels_right = False

	# da.plot.imshow(ax=ax, transform=ccrs.PlateCarree())
	

	for vas, cl, colour in zip(["RF", "IR", "AR"], ["r*", 'yx', "k."], ["#d7191c","#e66101","#5e3c99"]):
		ax.plot(
			SiteInfo[SiteInfo.RF == vas].lon.values, 
			SiteInfo[SiteInfo.RF == vas].lat.values, cl,
			color=colour, 
			markersize=8, transform=ccrs.PlateCarree(), label=vas)
	ax.legend()

	# ========== Save the plot ==========
	if not (formats is None): 
		print("starting plot save at:", pd.Timestamp.now())
		# ========== loop over the formats ==========
		for fmt in formats:
			plt.savefig(fname+fmt, dpi=dpi)

	plt.show()
	if not (fname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, fname, gitinfo]
		cf.writemetadata(fname, infomation)

	ipdb.set_trace()


	

#==============================================================================
def shade(located_elevations):
    """
    Given an array of elevations in a LocatedImage, add a relief (shadows) to
    give a realistic 3d appearance.

    """
    new_img = srtm.add_shading(located_elevations.image,
                               azimuth=135, altitude=15)
    return LocatedImage(new_img, located_elevations.extent)

def Field_data(year = 2017):
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

	# ipdb.set_trace()
	RFinfo.RF[    RFinfo["RF"].str.contains("poor")] = "RF"  #"no regeneration"
	RFinfo.RF[    RFinfo["RF"].str.contains("no regeneration")] = "RF" 
	RFinfo.RF[RFinfo["RF"].str.contains("singular")] = "IR"  
	for repstring in ["abundunt", "sufficient", "abundant", "sufficent", "sifficient"]:
		RFinfo.RF[RFinfo["RF"].str.contains(repstring)] = "AR"  
	
	return RFinfo
if __name__ == '__main__':
	main()