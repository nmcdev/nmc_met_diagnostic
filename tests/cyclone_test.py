# _*_ coding: utf-8 _*_

"""
Cyclones identification test.
"""

import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
from nmc_met_graphics.draw_synoptic_analysis import draw_850_wind
from nmc_met_diagnostic.feature import cyclone_loc


data_dir = "H:/case_20160719/data/raw/tigge/ecmf/nc"
ana_time = dt.datetime(2016, 7, 20, 0)

# read pressure level data
filename = os.path.join(data_dir, "ecmf_fc_pl_" + ana_time.strftime('%Y%m%d%H') + ".nc")
fio = Dataset(filename, mode='r')
lon = fio.variables['longitude'][:]
lat = fio.variables['latitude'][:]
levs = fio.variables['level'][:]
id_lev = np.where(levs == 850)
u = np.squeeze((fio.variables['u'][:])[0, id_lev, :, :])
v = np.squeeze((fio.variables['v'][:])[0, id_lev, :, :])
fio.close()

# read mean sea level pressure
filename = os.path.join(data_dir, "ecmf_fc_sfc_" + ana_time.strftime('%Y%m%d%H') + ".nc")
fio = Dataset(filename, mode='r')
msl = np.squeeze((fio.variables['msl'][:])[0, :, :]) / 100.
fio.close()

# identify cyclone
low_loc = cyclone_loc(msl, lon, lat, edge_distance=600e3,
                      search_rad_max=300e3, search_rad_min=150e3,
                      search_rad_ndiv=3, slp_diff_test=0.5, limit=[110, 125, 28, 42])

# set figure
plotcrs = ccrs.PlateCarree(central_longitude=110.)
fig = plt.figure(figsize=(6, 6.8))
ax = plt.axes(projection=plotcrs)
right_title = "Analysis: {}".format(ana_time.strftime('%Y-%m-%d %H:00'))
cf, bb = draw_850_wind(ax, lon, lat, u, v, mslp=[lon, lat, msl],
                       map_extent=[102, 122, 23, 43], left_title="", right_title=right_title)
bb.length = 0.4

# add cyclone center
if low_loc is not None:
    ax.annotate("{:6.1f}".format(low_loc[0, 2]), xy=(low_loc[0, 0]-0.5, low_loc[0, 1]-0.5),
                xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow'))
    ax.scatter(low_loc[0, 0], low_loc[0, 1], edgecolors="k", facecolors="white",
               linewidth=2, s=100, transform=ccrs.PlateCarree())

fig.subplots_adjust(bottom=0.15)
cax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
cb = plt.colorbar(cf, cax=cax, orientation='horizontal', extendrect=True)
cb.set_label('850hPa wind speed [m/s]', size='large', fontsize=18)
cb.ax.tick_params(labelsize=16)

plt.show()

