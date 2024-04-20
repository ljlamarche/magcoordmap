import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import magcoordmap as mcm

#map_proj = ccrs.Mercator()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection=map_proj)

fig = plt.figure()
proj = ccrs.AzimuthalEquidistant(central_longitude=-147, central_latitude=64)
ax = fig.add_subplot(111, projection=proj)
gl = ax.gridlines(draw_labels=True, zorder=1)
gl.right_labels = False
ax.set_extent([-170., -135., 52., 72.], crs=ccrs.PlateCarree())

ax.coastlines()
ax.gridlines()
mcm.add_magnetic_gridlines(ax)

plt.show()
