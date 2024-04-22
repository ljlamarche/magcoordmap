# magcoordmap
Adds a grid of constant magnetic latitude and longitude to a cartopy map.  This currently only supports Apex coordinates as provided by [apexpy](https://apexpy.readthedocs.io/en/latest/).

![Example map of Alaska with both Geodetic and Apex magnetic gridlines on it.](https://github.com/ljlamarche/magcoordmap/blob/main/example_map.png)

# Installation
This package can be installed with pip after cloning.

```
git clone https://github.com/ljlamarche/magcoordmap.git
pip install magcoordmap
```

Note that some dependencies ([apexpy](https://apexpy.readthedocs.io/en/latest/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/)) are dependent on correct linking with compiled libraries on your system.  If installation is proving problematic, try installing these packages independently following their own installation instructions before installing magcoordmap with pip.

# Usage
To add a magnetic coordinate grid to a cartopy map with the axes `ax`, simpily import `magcoordmap` and use the `add_magnetic_gridlines` function.

```python
import magcoordmap as mcm
mcm.add_magnetic_gridlines(ax)
```

By default, this uses Apex coordinates for the present system date at 0 km altitude.  If you would like to customize this, initialize your own Apex object and pass it to the function.

```python
import magcoordmap as mcm
from apexpy import Apex
A = Apex(2015)
mcm.add_magnetic_gridlines(ax, apex=A)
```

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
mcm.add_magnetic_gridlines(ax)

plt.show()
```
