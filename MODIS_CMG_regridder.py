import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt

ds = xr.open_dataset("MOD13C1.A2000049.006.2015147153445.hdf", engine="pynio") 

# ========== fix the lon and lats ==========
ds = ds.rename({"XDim_MODIS_Grid_16Day_VI_CMG":"longitude", "YDim_MODIS_Grid_16Day_VI_CMG":"latitude"}) 

xv = np.arange(-179.975, 180.025, 0.05) 
yv = np.arange(89.975, -90.025, -0.05)

ds["longitude"] = xv
ds["latitude"]  = yv

ds = ds["CMG_0_05_Deg_16_days_NDVI"].rename("ndvi") 



# ========== work out the date ==========
pd.to_datetime("2000049", format='%Y%j') 