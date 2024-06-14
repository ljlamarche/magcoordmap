import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from apexpy import Apex
import magcoordmap as mcm

#proj = ccrs.PlateCarree()
proj = ccrs.AzimuthalEquidistant(central_longitude=-147, central_latitude=64)
#proj = ccrs.NorthPolarStereo(central_longitude=0.)

fig = plt.figure()
ax = fig.add_subplot(111, projection=proj)

gl = ax.gridlines(draw_labels=True, zorder=1)
gl.right_labels = False
gl.top_labels = False
#ax.set_extent([-170., -35., -80., 80.], crs=ccrs.PlateCarree())
ax.set_extent([-170., -135., 52., 72.], crs=ccrs.PlateCarree())
#ax.set_extent([-45., 135., 40., 40.], crs=ccrs.PlateCarree())

ax.coastlines()
ax.gridlines()

A = Apex(2003)

mgl = mcm.maggridlines(ax, draw_labels=True)
#gl = mcm.maggridlines(ax, apex=A, draw_labels=True, xlabel_style={'color':'limegreen'}, collection_kwargs={'color':'magenta'})
mgl.bottom_labels = False
mgl.left_labels = False

plt.show()
