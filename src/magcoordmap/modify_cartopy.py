# modify_cartopy.py

import numpy as np
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from apexpy import Apex

def add_magnetic_gridlines(ax, draw_parallels=True, draw_meridians=True):

    proj = ax.projection
    A = Apex()

    x0, x1, y0, y1 = ax.get_extent()
    boundaries_x = np.array([x0, x1, x1, x0])
    boundaries_y = np.array([y0, y0, y1, y1])
    dataproj = ccrs.PlateCarree() # the data projection, PlateCarree for lat/lon data
    boundaries_geo = dataproj.transform_points(proj, boundaries_x, boundaries_y)
    boundaries_glon, boundaries_glat, _ = boundaries_geo.T
    
    boundaries_mlat, boundaries_mlon = A.geo2apex(boundaries_glat, boundaries_glon, height=0.)
    
    mlon0 = np.min(boundaries_mlon)
    mlon1 = np.max(boundaries_mlon)
    mlat0 = np.min(boundaries_mlat)
    mlat1 = np.max(boundaries_mlat)
    
    
    ticker = MaxNLocator()

    if draw_parallels:
        # Add Magnetic Parallels (lines of constant MLAT)
        magnetic_parallels = ticker.tick_values(mlat0, mlat1)
        mlon_arr = np.linspace(mlon0, mlon1, 100)
        
        for mlat in magnetic_parallels:
            mlat_arr = np.full(100, mlat)
        
            glat_arr, glon_arr, _ = A.apex2geo(mlat_arr, mlon_arr, height=0.)
            line = proj.transform_points(ccrs.Geodetic(), glon_arr, glat_arr)
            ax.plot(line[:,0], line[:,1], linewidth=0.5, color='orange', zorder=2)
        
            yl = np.interp(x1, line[:,0], line[:,1])
            if yl>y0 and yl<y1:
                ax.text(x1+3e4, yl, f'{mlat:.0f}°N', verticalalignment='center', color='orange')
    
    if draw_meridians:
        # Add Magnetic Meridians (lines of constant MLON)
        magnetic_meridians = ticker.tick_values(mlon0, mlon1)
        mlat_arr = np.linspace(mlat0, mlat1, 100)
        
        for mlon in magnetic_meridians:
            mlon_arr = np.full(100, mlon)
        
            glat_arr, glon_arr, _ = A.apex2geo(mlat_arr, mlon_arr, height=0.)
            line = proj.transform_points(ccrs.Geodetic(), glon_arr, glat_arr)
            ax.plot(line[:,0], line[:,1], linewidth=0.5, color='orange', zorder=2)
        
            xl = np.interp(y1, line[:,1], line[:,0])
            if xl>x0 and xl<x1:
                ax.text(xl, y1+3e4, f'{mlon:.0f}°E', rotation='vertical', verticalalignment='bottom', horizontalalignment='center', color='orange')



