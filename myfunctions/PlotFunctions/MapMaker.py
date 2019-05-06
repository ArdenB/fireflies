# -*- coding: utf-8 -*-
"""
Function to make maps

This is a fork of my mapmaker script built with xarray in mind
rather than plt imsho and numpy arrays


"""
#==============================================================================

__title__ = "Xarray Map Maker"
__author__ = "Arden Burrell"
__version__ = "2.0 (04.05.2019)"
__email__ = "arden.burrell@gmail.com"

#==============================================================================

# Import packages
import numpy as np
import pandas as pd
import sys	
import ipdb
import xarray as xr
import bottleneck as bn
# import datetime as dt

# Mapping packages
import cartopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib.pyplot as plt
import warnings as warn

# if sys.version.startswith("2.7"):
# 	from  .. import CoreFunctions as cf 	
# elif sys.version.startswith("3.7"):
# 	from  .. import CoreFunctions as cf 	

#==============================================================================
def mapmaker(ds, mapdet):
	"""
	Function to Build some maps using xarray 
	args:
		ds:		xr DS
			dataset containing results
		mapdet:	
			object of class pf.mapclass
	"""
	# ========== Check the class of the input data ==========
	if type(mapdet).__name__ != 'mapclass':
		raise TypeError("mapdet must be of class mapclass")

	# =========== Crop the relevant variabile ===========
	DA = ds[mapdet.var]#.isel(time=0)

	# =========== Check for a datamask mask  ===========
	if not (mapdet.mask is None):
		DA = DA.where(mapdet.mask[mapdet.masknm].values == 1.0)

	# =========== Check for a FDR significance mask  ===========
	if (not mapdet.sigmask is None):
		# ========== Check the method of significance masking ==========
		if (not mapdet.sighatch):
			# The approach zeros the non significant values
			DA *= ds[mapdet.sigmask] 
		else:
			# Use a hatching to denote significance
			dfhatch = _hatchmaker(ds, mapdet)

	plt.rcParams.update({'figure.subplot.right' : 0.85 })
	plt.rcParams.update({'figure.subplot.left' : 0.05 })
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : mapdet.latsize}

	matplotlib.rc('font', **font)

	aus_names = ["AUS", "Australia"]

	if mapdet.region in aus_names:
		fig, ax = plt.subplots(1, 1, figsize=(12,9),
			subplot_kw={'projection': ccrs.PlateCarree()}, 
			num=("Map of %s" % mapdet.var), dpi=mapdet.dpi)
	if mapdet.region in "MONG":
		fig, ax = plt.subplots(1, 1, figsize=(18,6.5),
			subplot_kw={'projection': ccrs.PlateCarree()}, 
			num=("Map of %s" % mapdet.var), dpi=mapdet.dpi)
	elif mapdet.region == "boreal":
		fig, ax = plt.subplots(1, 1, figsize=(18,4),
			subplot_kw={'projection': mapdet.projection}, 
			num=("Map of %s" % mapdet.var), dpi=mapdet.dpi)
	else:
		fig, ax = plt.subplots(1, 1, figsize=(18,8),
			subplot_kw={'projection': ccrs.PlateCarree()}, 
			num=("Map of %s" % mapdet.var), dpi=mapdet.dpi)
	

	# ========== Add features to the map ==========
	ax.add_feature(cpf.LAND, facecolor=mapdet.maskcol, alpha=1, zorder=0)
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(cpf.COASTLINE, zorder=101)
	if mapdet.national:
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	ax.outline_patch.set_visible(False)
	# ax.gridlines()



	# =========== Set up the axis ==========
	if mapdet.projection == ccrs.PlateCarree():
		gl = ax.gridlines(
			crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, 
			linestyle='--', zorder=105)

		gl.xlabels_top = False
		gl.ylabels_right = False
		# gl.xlines = False
		if mapdet.region in aus_names:
			gl.xlocator = mticker.FixedLocator(range(110, 170, 10))
			gl.ylocator = mticker.FixedLocator(range(-10, -60, -10))
		elif mapdet.region == "MONG":
			gl.xlocator = mticker.FixedLocator(np.arange(80.0, 125.0, 5.0))
			gl.ylocator = mticker.FixedLocator(np.arange(56.0, 40.0, -2.0))
		elif mapdet.region == "boreal":
			# gl.xlocator = mticker.FixedLocator(np.arange(80.0, 125.0, 5.0))
			gl.ylocator = mticker.FixedLocator(
				np.arange(mapdet.bounds[2]+5, mapdet.bounds[3],  -10.0))
		# elif mapdet.region == "Cropped":
		# 	gl.xlocator = mticker.FixedLocator(range(-140, 161, 20))
			# gl.ylocator = mticker.FixedLocator(range(mapdet.crop[0], mapdet.crop[1], -10))
		if not mapdet.xlocator is None:
			gl.xlocator = mticker.FixedLocator(mapdet.xlocator)
		else:pass

		if not mapdet.ylocator is None:
			gl.ylocator = mticker.FixedLocator(mapdet.ylocator)
		else:pass

		gl.xformatter = LONGITUDE_FORMATTER
		gl.yformatter = LATITUDE_FORMATTER


	# ========== Subset the dataarray using the bounds ==========
	if not mapdet.bounds is None:
		DA = DA.loc[dict(
			longitude=slice(mapdet.bounds[0], mapdet.bounds[1]),
			latitude=slice(mapdet.bounds[2], mapdet.bounds[3]))]
		# ax.set_extent(mapdet.bounds, crs=ccrs.PlateCarree())
	else:
		# ax.set_global()
		pass
	

	# ========== Create the plot ==========
	im = DA.plot(
		ax=ax, transform=ccrs.PlateCarree(),
		cmap=mapdet.cmap, 
		add_labels=False,
		# cbar_kwargs={"extend":mapdet.extend},
		add_colorbar=False) #cmap=cmap, vmin=vmin, vmax=vmax, 
		# vmin=vmin, vmax=vmax, 

	# ========== Add any hatching ==========
	if (not mapdet.sigmask is None):
		# ========== Check the method of significance masking ==========
		if mapdet.sighatch:
			ax.scatter(dfhatch["xlons"], dfhatch["ylats"] ,s=4, c='k', marker='X', 
				facecolors='none', edgecolors="none",  
				alpha=0.35, transform=ccrs.PlateCarree())
	

	# ========== Find the posistion of the ax ==========
	# re-calculated at each figure resize. 
	posn = ax.get_position()
	cbar_ax = fig.add_axes([posn.x0 + posn.width + 0.005, posn.y0, 0.025, posn.height])
	
	# ========== Add an autoresizing colorbar ========== 
	def resize_colobar(event):
		plt.draw()

		posn = ax.get_position()
		cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.024, posn.height])

	fig.canvas.mpl_connect('resize_event', resize_colobar)

	# set the limits on the colorbar
	im.set_clim(mapdet.cmin, mapdet.cmax)
	# set the position of the colorbar
	posn = ax.get_position()
	cbar_ax.set_position([posn.x0 + posn.width + 0.010, posn.y0*1.1, 0.024, posn.height*0.97]) 


	cb = plt.colorbar(
		im, 
		cax        = cbar_ax, 
		extend     = mapdet.extend, 
		norm       = mapdet.norm,
		ticks      = mapdet.ticks, 
		spacing    = mapdet.spacing,
		boundaries = mapdet.cbounds
		)

	# ========== Fix all the tick labels ==========
	if not (mapdet.ticknm is None):
		cb.ax.set_yticklabels(mapdet.ticknm) 
		for t in cb.ax.get_yticklabels():
			t.set_fontsize(mapdet.fontsize)
			t.set_x(mapdet.set_x)
			if isinstance(mapdet.ticknm[0], float):
				t.set_horizontalalignment('right')

	else:
		# Change the horrixontal allignment of cb ticks to right
		for t in cb.ax.get_yticklabels():
			t.set_horizontalalignment('right')   
			t.set_x(mapdet.set_x)
			t.set_fontsize(mapdet.fontsize)
			# t.set_fontsize(16)

	# ========== Add a label to the colorbar ==========
	if not (mapdet.cblabel is None):
		cbar_ax.set_ylabel(mapdet.cblabel, rotation=90, weight='bold')

	if mapdet.borders:
		#Adding the borders
		for bord in ['top', 'right', 'bottom', 'left']:

			ax.spines[bord].set_visible(True)
			ax.spines[bord].set_zorder(20)
			ax.spines[bord].set_color('k')
			ax.spines[bord].set_linewidth(2.0)

	# ========== Save the plot ==========
	if mapdet.save:
		# Make a pdf version
		print("\n Starting the figure save process at" , pd.Timestamp.now())
		print("At high DPI or on linux this can be very slow \n")
		for ext in mapdet.format:
			plt.savefig(mapdet.fname+ext, dpi=fig.dpi)
		# plt.savefig(mapdet.fname+".eps", dpi=fig.dpi)
		# plt.savefig(mapdet.fname+".png", dpi=fig.dpi)

		plotinfo = "PLOT INFO: Plot of %s made using %s:v.%s" % (
			mapdet.var, __title__, __version__)
		if mapdet.pshow:
			plt.show()
		fname =  mapdet.fname+".pdf"
	else:
		if mapdet.pshow:
			plt.show()
		warn.warn("mapdet.save = False has put Maker in dev mode to allow mods")
		ipdb.set_trace()
		# Reset the the plt paramters to the defualts
		fname    = None
		plotinfo = None
	plt.close()
	plt.rcParams.update(plt.rcParamsDefault)
	return plotinfo, fname


#==============================================================================

def _hatchmaker(ds, mapdet):
	"""
	Function takes the mapdet and dataset and build a significance hatching 
	pandas dataframe with to columns. 
	args:
		ds:		xr DS
			dataset containing results
		mapdet:	
			object of class pf.mapclass
	returns
		df:		pd.df
			contains the lats and lons of the hatching
	"""
	# ========== Containers to hold the xvals ==========
	lat_vals = []
	lon_vals = []

	# ========== Loop over each subset ==========
	DA_SM = ds[mapdet.sigmask]
	
	# =========== Check for a datamask mask  ===========
	if not (mapdet.mask is None):
		DA_SM = DA_SM.where(mapdet.mask[mapdet.masknm].values == 1.0)

	print("Starting hatching location calculation at:", pd.Timestamp.now())
	# =========== loop over the lats and the lons  ===========
	for yv in range(0, ds.latitude.shape[0],  mapdet.sigbox):

		for xv in range(0, ds.longitude.shape[0],  mapdet.sigbox):
			# ========== Pull out the Region to check ==========
			box = DA_SM[dict(
				latitude=slice(yv, yv+mapdet.sigbox), 
				longitude=slice(xv, xv+mapdet.sigbox))]
			nmb = bn.nansum(box)/float(mapdet.sigbox**2) # The number of valid values in the box
			
			# ========== add the lat and lon if significant ==========
			if nmb >= mapdet.sigfrac:
				lat_vals.append(np.median(box.latitude))
				lon_vals.append(np.median(box.longitude))
		# ========== Stop the looping once outside the bounds ==========
		if not (mapdet.bounds is None):
			if box.latitude.min() > mapdet.bounds[2]:
				continue
			elif box.latitude.max() < mapdet.bounds[3]:
				break
	return pd.DataFrame({"ylats":np.array(lat_vals), "xlons":np.array(lon_vals)})