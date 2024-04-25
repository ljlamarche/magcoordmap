# modify_cartopy.py

import numpy as np
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import cartopy.crs as ccrs
import cartopy.mpl.gridliner as cgl
from apexpy import Apex

def add_magnetic_gridlines(ax, apex=None, apex_height=0., draw_parallels=True, draw_meridians=True, xlocator=None, ylocator=None, **collection_kwargs):
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

    # Create default locator objects if not provided
    if not xlocator:
        xlocator = mticker.MaxNLocator()
    if not ylocator:
        ylocator = mticker.MaxNLocator()

    # Set gridline parameters based on defaults and provided keywords
    line_params = dict(color='orange', linewidth=0.5, zorder=2)
    for key, value in collection_kwargs.items():
        line_params[key] = value

    # Get projections for later transforms
    mapproj = ax.projection
    geoproj = ccrs.Geodetic()

    # Find map boundaries
    x0, x1, y0, y1 = ax.get_extent()
    boundaries_x = np.array([x0, x1, x1, x0])
    boundaries_y = np.array([y0, y0, y1, y1])
    # Get boundaries in geographic coordinates
    boundaries_geo = geoproj.transform_points(mapproj, boundaries_x, boundaries_y)
    boundaries_glon, boundaries_glat, _ = boundaries_geo.T
    # Get boundaries in Apex coordinates
    boundaries_mlat, boundaries_mlon = apex.geo2apex(boundaries_glat, boundaries_glon, height=apex_height)
    # Find min/max magnetic latitude and longitude 
    mlon0 = np.min(boundaries_mlon)
    mlon1 = np.max(boundaries_mlon)
    mlat0 = np.min(boundaries_mlat)
    mlat1 = np.max(boundaries_mlat)

   

    # Label formatter functions from https://scitools.org.uk/cartopy/docs/v0.13/_modules/cartopy/mpl/gridliner.html
    #: A formatter which turns longitude values into nice longitudes such as 110W
    longitude_formatter = mticker.FuncFormatter(lambda v, pos:
                                                cgl._east_west_formatted(v))
    #: A formatter which turns latitude values into nice latitudes such as 45S
    latitude_formatter = mticker.FuncFormatter(lambda v, pos:
                                           cgl._north_south_formatted(v))

    # Label tranformation functions (to offsel labels from edge of plot)
    shift_dist_points = 5
    tr_y = ax.transAxes + mtrans.ScaledTranslation(0.0, shift_dist_points * (1.0 / 72), ax.figure.dpi_scale_trans)
    tr_x = ax.transAxes + mtrans.ScaledTranslation(shift_dist_points * (1.0 / 72), 0.0, ax.figure.dpi_scale_trans)


    if draw_parallels:
        # Add Magnetic Parallels (lines of constant MLAT)

        # Use matplotlib ticker to find parallel line locations
        magnetic_parallels = ylocator.tick_values(mlat0, mlat1)
        magnetic_parallels = magnetic_parallels[(magnetic_parallels>=-90.) & (magnetic_parallels<=90.)]

        # Define transform to be used with labels
        #   (horizontal shift on x and data on y)
        label_transform = mtrans.blended_transform_factory(x_transform=tr_x, y_transform=ax.transData)

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

            # Find intersection of gridline with edge of plot
            yl = np.interp(x1, line[:,0], line[:,1])

            # If intersection within the limits of plot, add label
            if yl>y0 and yl<y1:
                label_str = latitude_formatter(mlat)
                ax.text(1., yl, label_str, verticalalignment='center', color=line_params['color'], transform=label_transform)
    

    if draw_meridians:
        # Add Magnetic Meridians (lines of constant MLON)

        # Use matplotlib ticker to find meridian line locations
        magnetic_meridians = xlocator.tick_values(mlon0, mlon1)

        # Define transform to be used with labels
        #   (data on x and vertical shift on y)
        label_transform = mtrans.blended_transform_factory(x_transform=ax.transData, y_transform=tr_y)

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

            # Fine intersection of gridline with edge of plot
            xl = np.interp(y1, line[:,1], line[:,0])

            # If intersection within the limits of plot, add label
            if xl>x0 and xl<x1:
                label_str = longitude_formatter(mlon)
                ax.text(xl, 1., label_str, rotation='vertical', verticalalignment='bottom', horizontalalignment='center', color=line_params['color'], transform=label_transform)



