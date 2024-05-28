# modify_cartopy.py

import numpy as np
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import cartopy.crs as ccrs
import cartopy.mpl.gridliner as cgl
from apexpy import Apex

def axes_domain(ax, apex, apex_height):
    """
    Find max and min mlat and mlon value contained within axes.  To be used to select appropriate grid lines.
    """

    nx = 30
    ny = 30

    # Get projections for later transforms
    mapproj = ax.projection
    geoproj = ccrs.Geodetic()

    # Find map extent
    x0, x1, y0, y1 = ax.get_extent()
    x, y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))

    # Get map extent in geographic coordinates
    extent_geo = geoproj.transform_points(mapproj, x, y)
    extent_glon, extent_glat, _ = extent_geo.T
    # Get map extent in Apex coordinates
    extent_mlat, extent_mlon = apex.geo2apex(extent_glat, extent_glon, height=apex_height)
    #ax.scatter(extent_glon, extent_glat, c=extent_mlat, transform=ccrs.Geodetic())
    # Find min/max magnetic latitude and longitude 
    mlon0 = np.nanmin(extent_mlon)
    mlon1 = np.nanmax(extent_mlon)
    mlat0 = np.nanmin(extent_mlat)
    mlat1 = np.nanmax(extent_mlat)
    #print('MLON', mlon0, mlon1)
    #print('MLAT', mlat0, mlat1)

    return mlon0, mlon1, mlat0, mlat1


def add_magnetic_gridlines(ax, apex=None, apex_height=0., draw_parallels=True, draw_meridians=True, xlocator=None, ylocator=None, **collection_kwargs):
    """
    Adds a magnetic coordinate system grid to an existing cartopy plot.

    Parameters
    ----------

    ax : :class:`matplotlib.axes.Axes`
        Axes on which to add gridline.  Must have a cartopy projection.
    apex : :class:`apexpy.Apex` (optional)
        Apex object that defines coordinate system and conversions.  An 
        Apex object that uses the system's current date will be initialized
        if not provided.
    apex_height : float (optional)
        Altitude to use for apex coordinate conversions.  Defaults to 0.
    draw_parallels : bool (optional)
        Whether or not to draw parallels (lines of constant MLAT).  Defauts
        to True.
    draw_meridians : bool (optional)
        Whether or not to draw meridians (lines of constant MLON).  Defaults
        to True
    xlocator : :class:`matplotlib.ticker.Locator` (optional)
        Locator object which will be used to determine the locations of the 
        MLON gridlines.
    ylocator : :class:`matplotlib.ticker.Locator` (optional)
        Locator object which will be used to determine the locations of the 
        MLAT gridlines.
    **collection_kwargs : (optional)
        :class:`matplotlib.collections.Collection` properties, used to 
        specify properties of the gridline such as linewidth and color.

    """


    # Create and Apex object for coordinate conversion if one is not provided
    if not apex:
        apex = Apex()

    # Use the cartopy default locator objects if not provided
    if not xlocator:
        xlocator = cgl.degree_locator
    if not ylocator:
        ylocator = cgl.degree_locator

    # Set gridline parameters based on defaults and provided keywords
    line_params = dict(color='orange', linewidth=0.5, zorder=2)
    for key, value in collection_kwargs.items():
        line_params[key] = value

    # Get projections for later transforms
    mapproj = ax.projection
    geoproj = ccrs.Geodetic()

    # Find map boundaries
    mlon0, mlon1, mlat0, mlat1 = axes_domain(ax, apex, apex_height)


    if draw_parallels:
        # Add Magnetic Parallels (lines of constant MLAT)

        # Use matplotlib ticker to find parallel line locations
        magnetic_parallels = ylocator.tick_values(mlat0, mlat1)
        magnetic_parallels = magnetic_parallels[(magnetic_parallels>=-90.) & (magnetic_parallels<=90.)]
        #print('MAGNETIC_PARALLELS', magnetic_parallels)

        # Define mlon array to use for all parallels
        mlon_arr = np.linspace(mlon0, mlon1, 100)
        
        for mlat in magnetic_parallels:
            # Define mlat array to use for this parallel
            mlat_arr = np.full(100, mlat)

            # Find grid line in geographic coordinates
            glat_arr, glon_arr, _ = apex.apex2geo(mlat_arr, mlon_arr, height=apex_height)
            # Transform grid line to matplotlib Data coordinates
            line = mapproj.transform_points(geoproj, glon_arr, glat_arr)

            # Plot grid line
            ax.plot(line[:,0], line[:,1], **line_params)


    if draw_meridians:
        # Add Magnetic Meridians (lines of constant MLON)

        # Use matplotlib ticker to find meridian line locations
        magnetic_meridians = xlocator.tick_values(mlon0, mlon1)

        # Define transform to be used with labels
        #   (data on x and vertical shift on y)
        #label_transform = mtrans.blended_transform_factory(x_transform=ax.transData, y_transform=tr_y)

        # Define mlat array to use for all meridians
        mlat_arr = np.linspace(mlat0, mlat1, 100)
        
        for mlon in magnetic_meridians:
            # Define mlon array to use for this parallel
            mlon_arr = np.full(100, mlon)
        
            # Find grid line in geographic coordinates
            glat_arr, glon_arr, _ = apex.apex2geo(mlat_arr, mlon_arr, height=apex_height)
            # Transform grid line to matplotlib Data coordinates
            line = mapproj.transform_points(geoproj, glon_arr, glat_arr)

            # Plot grid line
            ax.plot(line[:,0], line[:,1], **line_params)


