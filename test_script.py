import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#from apexpy import Apex
import magcoordmap as mcm

#proj = ccrs.Mercator()
proj = ccrs.AzimuthalEquidistant(central_longitude=-147, central_latitude=64)

fig = plt.figure()
ax = fig.add_subplot(111, projection=proj)

gl = ax.gridlines(draw_labels=True, zorder=1)
gl.right_labels = False
ax.set_extent([-170., -135., 52., 72.], crs=ccrs.PlateCarree())

ax.coastlines()
ax.gridlines()

#A = Apex(2023)
mcm.add_magnetic_gridlines(ax)

plt.show()
