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
import xarray as xr

import matplotlib.pyplot as plt
import warnings as warn

# if sys.version.startswith("2.7"):
# 	from  .. import CoreFunctions as cf 	
# elif sys.version.startswith("3.7"):
# 	from  .. import CoreFunctions as cf 	

#==============================================================================

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
			warn.warn("\n\n Hatching has not yet been implemented \n\n")
			ipdb.set_trace()


	# if QRB:
	# 	image = cf.immkr(array, mapdet.column, ret=True, plot=False)
	# else:
	# 	image = array
		# try:
		# 	image *= mapdet.mask
		
		# except TypeError:
		# 	warn.warn("\n\nThe masking seems to have failed, Trying to change the dtype\n\n")
		# 	image = image.astype(float) * mapdet.mask.astype(float)
		# except:
		# 	warn.warn("\n\nThe masking seems to have failed, Going interactive\n\n")



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
	else:
		# Change the horrixontal allignment of cb ticks to right
		for t in cb.ax.get_yticklabels():
			t.set_horizontalalignment('right')   
			# sx = 3.1
			t.set_x(mapdet.set_x)
			# t.set_x(3.0)
			t.set_fontsize(16)

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
		print("Starting the figure save process at" , pd.Timestamp.now())
		print("At high DPI or on my work desktop this can be very slow")
		plt.savefig(mapdet.fname+".pdf", dpi=fig.dpi)
		plt.savefig(mapdet.fname+".eps", dpi=fig.dpi)
		plt.savefig(mapdet.fname+".png", dpi=fig.dpi)

		plotinfo = "PLOT INFO: Plot of %s made using %s:v.%s" % (
			mapdet.var, __title__, __version__)
		plt.show()
		return mapdet.fname+".pdf", plotinfo
	else:
		plt.show()
		ipdb.set_trace()
		return None, None

	sys.exit()

	# if not (mapdet.crop is None):
	# 	y0 = np.min(np.where(mapdet.lats<=mapdet.crop[0]))
	# 	x0 = np.min(np.where(mapdet.lons>=mapdet.crop[2]))
	# 	# Doing a crop in only one axis
	# 	try:
	# 		y1 = np.min(np.where(mapdet.lats< mapdet.crop[1]))  
	# 	except ValueError:
	# 		y1 = -1
	# 	try:
	# 		x1 = np.min(np.where(mapdet.lons> mapdet.crop[3]))  
	# 	except ValueError:
	# 		x1 = -1
		
	# 	# Crop the image
	# 	image = image[y0:y1, x0:x1]
	# 	mapdet.bounds = [mapdet.crop[2], mapdet.crop[3], mapdet.crop[0], mapdet.crop[1]]
	# 	mapdet.region = "Cropped"
	# 	ax.set_extent(mapdet.bounds, crs=ccrs.PlateCarree())

	# if mapdet.region == "GLOBAL":
	# 	ax.set_global()
	# # elif mapdet.region == "MONG":
	# else:
	# 	ax.set_extent(mapdet.bounds, crs=ccrs.PlateCarree())
	
	ipdb.set_trace()
	# ========== plot the image ==========
	# if mapdet.region == "MONG":
	# 	origin="upper"
	# else:
	# origin="lower"
	# im = ax.imshow(image, 
	# 	extent=mapdet.bounds, 
	# 	cmap=mapdet.cmap, 
	# 	norm=mapdet.norm, 
	# 	origin=origin
	# 	) # added after australia looked lame

	

	if not mapdet.sigmask is None:
		ipdb.set_trace()
		# Calculate the lat and lon values
		slats = mapdet.lats[mapdet.sigmask["yv"]]
		slons = mapdet.lons[mapdet.sigmask["xv"]]

		ax.scatter(
			slons, slats,s=4, c='k', marker='X', 
			facecolors='none', edgecolors="none",  
			alpha=0.35, transform=ccrs.PlateCarree())

		# ipdb.set_trace()
		# plt.scatter(xvals, yvals, s=1, c='k', marker='.', alpha=0.5)


	# divider = make_axes_locatable(ax)
	# cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

	# im = ax.pcolormesh(image,  cmap=cmap, norm=mapdet.norm)
	# old feature commands
	# ax.add_feature(cpf.LAND, col="dimgrey")
	# ax.add_feature(states_provinces, edgecolor='gray')
	# ax.ocean(col='w')
	# ax.coastlines(resolution='110m')

	
	# gl.xlabel_style = {'size': 15, 'color': 'gray'}
	# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
	
	
	# test = True
	# if not test:
	# 	cbar_ax.set_position(
	# 		[posn.x0 + posn.width + 0.005, posn.y0, 0.025, posn.height]
	# 		)
	# else:
	# 	# cbar_ax.set_position([posn.x0 + posn.width + 0.005, 0, 0.05, 1])
	# 	ipdb.set_trace()



	if mapdet.dpi is None:
		dpi = fig.dpi
	else:
		dpi = mapdet.dpi
	# ========== save the plot ==========
	plt.draw()
	if not (mapdet.plotpath is None):
		if mapdet.ensemble:
			# catch the different types of ensembles
			if "diff" in mapdet.var:
				# difference between two runs:
				# Make a pdf version
				fnm_pdf = "%s%d.%d.map_%s.pdf" % (
					mapdet.plotpath, (mapdet.column-2), mapdet.paper,  mapdet.var)
				plt.savefig(fnm_pdf, dpi=dpi)
				
				# make png version 
				fname = "%s%d.%d.map_%s.png" % (
						mapdet.plotpath, (mapdet.column-2), mapdet.paper,  mapdet.var)
				plt.savefig(fname, dpi=dpi)
				
				plotinfo = "PLOT INFO: rundif: %dsub%d plot of %s made using %s:v.%s" % (
					mapdet.run[0], mapdet.run[1], mapdet.var, __title__, __version__)
			else:
				# ===== Run Ensembles ===== 
				# Make a pdf version
				fnm_pdf = "%s%d.%s_ensemble_map_%s.pdf" % ( 
					mapdet.plotpath, mapdet.paper,mapdet.desc, mapdet.var)

				plt.savefig(fnm_pdf, dpi=dpi)
				
				# make png version 
				fname = "%s%d.%s_ensemble_map_%s.png" % ( 
					mapdet.plotpath, mapdet.paper,mapdet.desc, mapdet.var)
				plt.savefig(fname, dpi=dpi)
				
				plotinfo = "PLOT INFO: Ensmble plot of %s made using %s:v.%s" % (
					mapdet.var,  __title__, __version__)
		else:
			# Make a pdf version
			fnm_pdf = "%s%d.%d_BasicFigs_%s.pdf" % (
				mapdet.plotpath, (mapdet.column-2), mapdet.run, mapdet.var)
			plt.savefig(fnm_pdf, dpi=fig.dpi)
				
			# make png version 
			fname = "%s%d.%d_BasicFigs_%s.png" % (
				mapdet.plotpath, (mapdet.column-2),mapdet.run, mapdet.var)
			plt.savefig(fname, dpi=fig.dpi)
			plotinfo = "PLOT INFO: Run:%d plot of %s made using %s:v.%s" % (
				mapdet.run, mapdet.var, __title__, __version__)
	else:
		fname = None
	if mapdet.pshow:
		plt.show()
	# ipdb.set_trace()
	plt.close()
	# Reset the the plt paramters to the defualts
	plt.rcParams.update(plt.rcParamsDefault)
	# return the infomation
	return plotinfo, fname

