## python
## Plot albedo from high-res scene files
## Date: 2019-01-30
## Author: Jeff Nivitanont

import sys
import numpy as np
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import shapely.geometry as sgeom
from netCDF4 import Dataset

alb_dir = '/home/jnivitanont/analysis/WRF/daily_albedo/'

alb_f= Dataset('WRF_alb_avg_20160801.nc','r')
albedo = alb_f['ALBEDO']
print(albedo)
plt.imshow(albedo)
plt.savefig('alb_plt', format='png')