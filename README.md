# magcoordmap
Adds a grid of constant magnetic latitude and longitude to a cartopy map.  This currently only supports Apex coordinates as provided by [apexpy](https://apexpy.readthedocs.io/en/latest/).

![Example map of Alaska with both Geodetic and Apex magnetic gridlines on it.](https://github.com/ljlamarche/magcoordmap/blob/main/example_map.png)

# Installation
This package can be installed with pip after cloning.

```
git clone https://github.com/ljlamarche/magcoordmap.git
cd magcoordmap
pip install .
```

Note that some dependencies ([apexpy](https://apexpy.readthedocs.io/en/latest/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/)) are dependent on correct linking with compiled libraries on your system.  If installation is proving problematic, try installing these packages independently following their own installation instructions before installing magcoordmap with pip.

# Usage
To add a magnetic coordinate grid to a cartopy map with the axes `ax`, simpily import `magcoordmap` and use the `add_magnetic_gridlines` function.

```python
import magcoordmap as mcm
mcm.maggridlines(ax)
```

By default, this uses Apex coordinates for the present system date at 0 km altitude.  If you would like to customize this, initialize your own Apex object and pass it to the function.

```python
import magcoordmap as mcm
from apexpy import Apex
A = Apex(2015)
mcm.add_magnetic_gridlines(ax, apex=A)
```

The function also has an option to specify the altitude of the magnetic grid with the `apex_height` option.  Note that all apex latitudes are not defined at all altitudes, and an error will be raised if the plot includes a magnetic field line with an apex lower than the specified height.  For global maps, it is recommended to keep this value at 0.

Additional keyword arguments are avaiable that control the characteristics and positioning of gridlines.  Because `MagGridliner` inherets from cartopy's `Gridliner` class, all the parameters from [gridlines](https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.geoaxes.GeoAxes.html#cartopy.mpl.geoaxes.GeoAxes.gridlines) will also be accepted by the maggridlines function (with the exception of `crs`, which is not relevant).


The following produces the figure shown on this page.

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import magcoordmap as mcm

# Set up figure
fig = plt.figure()
proj = ccrs.AzimuthalEquidistant(central_longitude=-147, central_latitude=64)
ax = fig.add_subplot(111, projection=proj)

# Set up cartopy map
gl = ax.gridlines(draw_labels=True, zorder=1)
gl.right_labels = False
gl.top_labels = False
ax.set_extent([-170., -135., 52., 72.], crs=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()

# Add magnetic field lines
mgl = mcm.maggridlines(ax)
mgl.left_labels=False
mgl.bottom_labels=False

plt.show()
```
