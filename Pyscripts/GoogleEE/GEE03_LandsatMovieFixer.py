"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

"""
#==============================================================================

__title__ = "GEE Movie Fixer"
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
import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob

from collections import OrderedDict
from scipy import stats
from numba import jit

# Import the Earth Engine Python Package
import ee
import ee.mapclient
from ee import batch

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 

import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default


import moviepy.editor as mpe
import skvideo.io     as skv
from moviepy.video.io.bindings import mplfig_to_npimage


# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():

	# ========== setup the filename ==========
	fn     = "/home/ubuntu/Downloads/LANDSAT_5_7_8_TestBurn_RGB.mp4" 
	# fn = "/home/ubuntu/Downloads/Site4_video_region_L8_time_v8_SR.mp4"    

	# ========== load the additional indomation ==========
	dft = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_TestBurn_RGB_timeinfo.csv", index_col=0, parse_dates=True)
	dfg = pd.read_csv("./data/other/tmp/LANDSAT_5_7_8_TestBurn_RGB_gridinfo.csv", index_col=0, parse_dates=True)      
	
	# ========== Set up the universal infomation ==========	
	bounds = [dfg.lonr_min[0], dfg.lonr_max[0], dfg.latr_max[0], dfg.latr_min[0]]

	# ========== Open the video a single frame at a time ==========
	videoin = skv.vread(fn)
	# videoin = skv.vreader(fn)

	# =========== Setup the annimation ===========
	fig, ax = plt.subplots(1, figsize=(10,10), dpi=400, subplot_kw={'projection': ccrs.PlateCarree()})
	ax.set_extent(bounds, crs=ccrs.PlateCarree())
	
	# ========== Loop over each frame of the video ==========
	nx = []

	def frame_maker(index):
		# ========== Pull the infomation from the pandas part of the loop ==========
		# index = rowinfo[0]
		# ipdb.set_trace()
		info  = dft.iloc[int(index)] #rowinfo[1]
		frame = videoin[int(index), :, :, :]
		# nx += 1
		ax.clear()
		ax.set_title("%s %s" % (info.satellite, info.date.split(" ")[0]))
		ax.imshow(frame, extent=bounds, transform=ccrs.PlateCarree())
		ax.scatter(dfg.lon[0], dfg.lat[0], 5, c='r', marker='+', transform=ccrs.PlateCarree())
		# nx.append(mplfig_to_npimage(fig))
		return mplfig_to_npimage(fig)

	# for frame, rowinfo in zip(videoin, dft.iterrows()):
	mov = mpe.VideoClip(frame_maker, duration=int(videoin.shape[0]))
	
	fnout = "/home/ubuntu/Downloads/LANDSAT_5_7_8_TestBurn_RGB_updated.mp4" 

	print("Starting Write of the data at:", pd.Timestamp.now())
	mov.write_videofile(fnout, fps=1)

	# plt.show()
	ipdb.set_trace()

#==============================================================================

if __name__ == '__main__':
	main()