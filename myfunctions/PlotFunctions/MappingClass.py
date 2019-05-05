""" 
A class that can be used to store the key paramaters about the tested run 
"""
#==============================================================================

__title__ = "Mapping Class"
__author__ = "Arden Burrell"
__version__ = "1.0 (13.02.2018)"
__email__ = "arden.burrell@gmail.com"
import cartopy.crs as ccrs

#==============================================================================
class mapclass(object):
	"""Class of object that contains infomation about the run
	Takes:
		The opened runs_info csv file 
		The run number
	Returns:
		
	"""
	def __init__(self, region, pshow=True):
		# ========== the variables defined by the rundet ========== 

		# Useful infomation
		aus_names = ["AUS", "Australia"]
		# ===== Using the run infomation, determine the bounds for a map =====
		if region == "GLOBAL":
			self.bounds = [-180, 180, 90, -90]
			self.set_x  = 3.1   # The spacing of the colorbar lables
		elif region in aus_names:
			self.bounds = [112.0, 156.25, -44.5, -10]
		elif region == "MONG":
			self.bounds = [85.0, 120.0, 52.0, 40.0]
		elif region == "boreal":
			self.bounds = [-180.0, 180.0, 80.0, 40.0]
			self.set_x  = 1.75   # The spacing of the colorbar lables
		else:
			Warning("The region code is unknown, unable to set bounds")
			self.bounds = None
		self.region = region
		self.pshow  = pshow # show the plot after saving?
		
		# ========== Set the blank variables ========== 
		self.var        = None  # The variable being mapped
		self.mask       = None  # used for passing drylands masks around
		self.masknm     = None  # When the mask file is an xr dataset, the var name
		self.sigmask    = None  # used for passing the column of significance maskis around
		self.sighatch   = False # used for passing the column of significance maskis around
		self.projection = ccrs.PlateCarree()
		self.sigbox     = 10    # The size of the box used to create the hatching
		self.cmap     = None  # Colormap set later
		self.cmin     = None  # the min of the colormap
		self.cmax     = None  # the max of the colormap
		self.cZero    = None  # the zero point of the colormap
		self.column   = None  # the column to be mapped
		self.norm     = None
		self.ticks    = None # The ticks on the colorbar
		self.ticknm   = None # The tick names on the colorbar
		self.cbounds  = None
		self.save     = True # Save the plot 
		self.fname    = None
		# ========== Set the Histogram lables ==========
		self.hist_x   = None  # X lable for the Histogram
		self.hist_y   = None  # X lable for the Histogram
		self.hist_b   = True  # Include a boxplot 
		self.cblabel  = None  # add label to colorbar
		self.cbsci    = None  # The scientific notation power range
		self.dpi      = 500   # The DPI of the output figures
		self.national = True  # Inculde national boundaries
		self.borders  = False # Figure borders
		
		self.xlocator = None  # mticker.FixedLocator(np.arange(80.0, 125.0, 5.0))
		self.ylocator = None  # mticker.FixedLocator(np.arange(56.0, 40.0, -2.0))

		
		
		# ========== Set the plot defualts ==========
		self.maskcol  = 'dimgrey'
		self.Oceancol = 'w'
		# Extend the colorbar to cover any data outside the cmap range
		self.extend   = "both" 
		self.spacing  = 'uniform'
		self.fontsize = 11
		self.latsize  = 16

		


