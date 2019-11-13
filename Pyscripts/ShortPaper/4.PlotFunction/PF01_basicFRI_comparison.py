"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

"""
#==============================================================================

__title__ = "FRI Comparison"
__author__ = "Arden Burrell"
__version__ = "v1.0(08.11.2019)"
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
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():

	# ========== Setup the params ==========

	mwbox   = [1, 2, 5]
	dsnames = ["COPERN_BA", "MODIS", "esacci"]#
	formats = [".png", ".pdf"] # None
	# mask    = True

	# ========== Setup the plot dir ==========
	plotdir = "./plots/ShortPaper/"
	cf.pymkdir(plotdir)

	for mwb in mwbox:
		# ========== Setup the dataset ==========
		datasets = OrderedDict()
		for dsnm in dsnames:
			# +++++ make a path +++++
			ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/%s/FRI/" %  dsnm
			fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsnm, mwb)
			
			# +++++ open the datasets +++++
			datasets[dsnm] = xr.open_dataset(ppath+fname)
			# ipdb.set_trace()
		
		for var in ["FRI", "AnBF"]:
			for mask in [True, False]:
				plotmaker(datasets, var, mwb, plotdir, formats, mask)

			ipdb.set_trace()

#==============================================================================
def plotmaker(datasets, var, mwb, plotdir, formats, mask):
	"""Function builds a basic stack of maps """

	# ========== make the plot name ==========
	plotfname = plotdir + "PF01_%s_MW_%02dDegBox" % (var, mwb)
	if mask:
		plotfname += "_ForestMask"
		
	# ========== setup the figure ==========
	fig, axs = plt.subplots(
		len(datasets), 1, sharex=True, 
		figsize=(16,9), subplot_kw={'projection': ccrs.PlateCarree()})

	# ========== Loop over the figure ==========
	for ax, dsn, in zip(axs, datasets):
		# make the figure
		im = _subplotmaker(ax, var, dsn, datasets, mask)
		ax.set_aspect('equal')

	# ========== Make the final figure adjusments ==========
	# +++++ Get rid of the excess lats +++++
	for ax in axs.flat:
		ax.label_outer()

	# +++++ Add a single colorbar +++++
	fig.colorbar(im, ax=axs.ravel().tolist(), extend="max")
	
	# ========== Change parms for the entire plot =========
	plt.axis('scaled')

	if not (formats is None): 
		print("starting plot save at:", pd.Timestamp.now())
		# ========== loop over the formats ==========
		for fmt in formats:
			plt.savefig(plotfname+fmt)#, dpi=dpi)

	plt.show()
	if not (plotfname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotfname, gitinfo]
		cf.writemetadata(plotfname, infomation)

#==============================================================================
def _subplotmaker(ax, var, dsn, datasets, mask, region = "SIBERIA"):
	
	# ========== Get the data for the frame ==========
	frame = datasets[dsn][var].isel(time=0)
	bounds = [-10.0, 180.0, 70.0, 40.0]

	# ========== mask ==========
	if mask:
		# +++++ Setup the paths +++++
		stpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/ForestExtent/%s/" % dsn
		fnmask = stpath + "Hansen_GFC-2018-v1.6_regrid_%s_%s_BorealMaskV2.nc" % (dsn, region)

		# +++++ Check if the mask exists yet +++++
		if os.path.isfile(fnmask):
			with xr.open_dataset(fnmask).drop("ForestFraction") as dsmask:
				
				msk    = dsmask.mask.isel(time=0).astype("float32").values

				# +++++ Change the boolean mask to NaNs +++++
				msk[msk == 0] = np.NAN
				
				print("Masking %s frame at:" % dsn, pd.Timestamp.now())
				# +++++ mask the frame +++++
				frame *= msk

				# +++++ close the mask +++++
				msk = None


		else:
			print("No mask exists for ", dsn)
	
	# ========== Set the colors ==========
	if var == "FRI":
		# +++++ set the min and max values +++++
		vmin = 0.0
		vmax = 80.0

		# +++++ create hte colormap +++++
		# cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors
		cmapHex = palettable.matplotlib.Viridis_9_r.hex_colors
		

		cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
		cmap.set_over(cmapHex[-1] )
		cmap.set_bad('dimgrey',1.)

	else:
		# ========== Set the colors ==========
		vmin = 0.0
		vmax = 0.20

		# +++++ create the colormap +++++
		# cmapHex = palettable.matplotlib.Inferno_10.hex_colors
		# cmapHex = palettable.matplotlib.Viridis_11_r.hex_colors
		cmapHex = palettable.colorbrewer.sequential.OrRd_9.hex_colors
		

		cmap    = mpl.colors.ListedColormap(cmapHex[:-1])
		cmap.set_over(cmapHex[-1] )
		cmap.set_bad('dimgrey',1.)

	# ========== Grab the data ==========
	im = frame.plot.imshow(
		ax=ax, extent=bounds, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False,
		transform=ccrs.PlateCarree())

	ax.set_extent(bounds, crs=ccrs.PlateCarree())

	# ========== Add features to the map ==========
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(cpf.COASTLINE, zorder=101)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	ax.outline_patch.set_visible(False)
	# ax.gridlines()

	# =========== Set up the gridlines ==========
	gl = ax.gridlines(
		crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, 
		linestyle='--', zorder=105)

	# +++++ get rid of the excess lables +++++
	gl.xlabels_top = False
	gl.ylabels_right = False
	if not dsn == [dss for dss in datasets][-1]:
		# Get rid of lables in the middle of the subplot
		gl.ylabels_bottom = False

	gl.xlocator = mticker.FixedLocator(np.arange(bounds[0], bounds[1]+10.0, 20.0)+10)
	gl.ylocator = mticker.FixedLocator(np.arange(bounds[2], bounds[3]-10.0, -10.0))
	
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER

	# ax.coastlines(resolution='110m')
	# ax.gridlines()
	## =========== Setup the annimation ===========
	ax.set_title(dsn)

	return im

#==============================================================================
if __name__ == '__main__':
	main()