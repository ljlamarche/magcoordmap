import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#from apexpy import Apex
#import magcoordmap as mcm
from magcoordmap import mag_gridliner

#proj = ccrs.Mollweide()
proj = ccrs.AzimuthalEquidistant(central_longitude=-147, central_latitude=64)
#proj = ccrs.NorthPolarStereo(central_longitude=0.)

fig = plt.figure()
ax = fig.add_subplot(111, projection=proj)

gl = ax.gridlines(draw_labels=True, zorder=1)
#gl.right_labels = False
#gl.top_labels = False
#ax.set_extent([-170., -35., -80., 80.], crs=ccrs.PlateCarree())
#ax.set_extent([-170., -135., 52., 72.], crs=ccrs.PlateCarree())
ax.set_extent([-170., -135., 52., 72.], crs=ccrs.PlateCarree())
#ax.set_extent([-45., 135., 40., 40.], crs=ccrs.PlateCarree())

ax.coastlines()
ax.gridlines()

#A = Apex(2023)
#mcm.add_magnetic_gridlines(ax)

#gl = mag_gridliner.maggridlines(ax, draw_labels=True)
gl = mag_gridliner.maggridlines(ax, draw_labels=True, xlabel_style={'color':'limegreen'}, collection_kwargs={'color':'magenta'})
#print(dir(gl))
#print(gl.xline_artists)
#ax.add_artist(gl)

#from cartopy.mpl.gridliner import Gridliner
#gl = Gridliner(ax, proj)
#print(ax)
#ax.add_artist(gl)

plt.show()
#print(gl.xline_artists)
