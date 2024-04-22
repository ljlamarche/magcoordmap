# modify_cartopy.py

import numpy as np
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from apexpy import Apex

def add_magnetic_gridlines(ax, apex=None, apex_height=0., draw_parallels=True, draw_meridians=True):
    """
    Adds a magnetic coordinate system grid to an existing cartopy plot.

    Parameters
    ----------

    ax : :obj:`matplotlib.axes.Axes`
        Axes on which to add gridline.  Must have a cartopy projection.
    apex : :obj:`apexpy.Apex` (optional)
        Apex object that defines coordinate system and conversions.  An 
        Apex object that uses the system's current date will be initialized
        if not provided.
    apex_height : float
        Altitude to use for apex coordinate conversions.  Defaults to 0.
    draw_parallels : bool
        Whether or not to draw parallels (lines of constant MLAT).  Defauts
        to True.
    draw_meridians : bool
        Whether or not to draw meridians (lines of constant MLON).  Defaults
        to True
    """


    # Create and Apex object for coordinate conversion if one is not provided
    if not apex:
        apex = Apex()

    # Find map boundaries
    mapproj = ax.projection
    geoproj = ccrs.Geodetic()

    x0, x1, y0, y1 = ax.get_extent()
    boundaries_x = np.array([x0, x1, x1, x0])
    boundaries_y = np.array([y0, y0, y1, y1])

    boundaries_geo = geoproj.transform_points(mapproj, boundaries_x, boundaries_y)
    boundaries_glon, boundaries_glat, _ = boundaries_geo.T

    #ax.scatter(boundaries_glon, boundaries_glat, transform=geoproj)
    
    boundaries_mlat, boundaries_mlon = apex.geo2apex(boundaries_glat, boundaries_glon, height=apex_height)
    
    mlon0 = np.min(boundaries_mlon)
    mlon1 = np.max(boundaries_mlon)
    mlat0 = np.min(boundaries_mlat)
    mlat1 = np.max(boundaries_mlat)

    #print(mlon0, mlon1, mlat0, mlat1)
    
    
    ticker = MaxNLocator()

    if draw_parallels:
        # Add Magnetic Parallels (lines of constant MLAT)
        magnetic_parallels = ticker.tick_values(mlat0, mlat1)
        magnetic_parallels = magnetic_parallels[(magnetic_parallels>=-90.) & (magnetic_parallels<=90.)]
        mlon_arr = np.linspace(mlon0, mlon1, 100)
        
        for mlat in magnetic_parallels:
            mlat_arr = np.full(100, mlat)
        
            glat_arr, glon_arr, _ = apex.apex2geo(mlat_arr, mlon_arr, height=apex_height)
            glon_arr[glon_arr<0] = glon_arr[glon_arr<0] + 360.
            line = mapproj.transform_points(geoproj, glon_arr, glat_arr)
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
        
            glat_arr, glon_arr, _ = apex.apex2geo(mlat_arr, mlon_arr, height=apex_height)
            line = mapproj.transform_points(geoproj, glon_arr, glat_arr)
            ax.plot(line[:,0], line[:,1], linewidth=0.5, color='orange', zorder=2)
        
            xl = np.interp(y1, line[:,1], line[:,0])
            if xl>x0 and xl<x1:
                ax.text(xl, y1+3e4, f'{mlon:.0f}°E', rotation='vertical', verticalalignment='bottom', horizontalalignment='center', color='orange')



