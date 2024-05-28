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


def find_roots(x,y):
    # from answer on https://stackoverflow.com/questions/46909373/how-to-find-the-exact-intersection-of-a-curve-as-np-array-with-y-0
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

def edge_intercept(ax, line):
    """
    Find points at which the line intersects the edges of the axes.
    """

    x0, x1, y0, y1 = ax.get_extent()

   # # find intersection with left side of plot
   # yl = np.interp(x0, line[:,0], line[:,1], left=np.nan, right=np.nan)
   # if yl>y0 and yl<y1:
   #     left = yl
   # else:
   #     left = None
    yl = find_roots(line[:,1], line[:,0]-x0)
    left = [y for y in yl if (y>y0 and y<y1)]

   # # find intersection with right side of plot
   # yl = np.interp(x1, line[:,0], line[:,1], left=np.nan, right=np.nan)
   # if yl>y0 and yl<y1:
   #     right = yl
   # else:
   #     right = None
    yl = find_roots(line[:,1], line[:,0]-x1)
    right = [y for y in yl if (y>y0 and y<y1)]

    # find intersection with bottom of plot
    #xl = np.interp(y0, line[:,1], line[:,0], left=np.nan, right=np.nan)
    #if xl>x0 and xl<x1:
    #    bottom = xl
    #else:
    #    bottom = None
    xl = find_roots(line[:,0], line[:,1]-y0)
    bottom = [x for x in xl if (x>x0 and x<x1)]

    # find intersection with top of plot
    xl = find_roots(line[:,0], line[:,1]-y1)
    #xl = np.interp(y1, line[:,1], line[:,0], left=np.nan, right=np.nan)
    #print('line', line)
    #print('y1', y1)
    top = [x for x in xl if (x>x0 and x<x1)]
    #if xl>x0 and xl<x1:
    #    top = xl
    #else:
    #    top = None

    return left, right, top, bottom

def add_magnetic_gridlines(ax, apex=None, apex_height=0., draw_parallels=True, draw_meridians=True, xlocator=None, ylocator=None, draw_labels=False, **collection_kwargs):
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

    ## Find map boundaries
    #x0, x1, y0, y1 = ax.get_extent()
    #boundaries_x = np.array([x0, x1, x1, x0])
    #boundaries_y = np.array([y0, y0, y1, y1])
    ## Get boundaries in geographic coordinates
    #boundaries_geo = geoproj.transform_points(mapproj, boundaries_x, boundaries_y)
    #boundaries_glon, boundaries_glat, _ = boundaries_geo.T
    ## Get boundaries in Apex coordinates
    #boundaries_mlat, boundaries_mlon = apex.geo2apex(boundaries_glat, boundaries_glon, height=apex_height)
    ## Find min/max magnetic latitude and longitude 
    #mlon0 = np.min(boundaries_mlon)
    #mlon1 = np.max(boundaries_mlon)
    #mlat0 = np.min(boundaries_mlat)
    #mlat1 = np.max(boundaries_mlat)

    mlon0, mlon1, mlat0, mlat1 = axes_domain(ax, apex, apex_height)

   

    # Label formatter functions from https://scitools.org.uk/cartopy/docs/v0.13/_modules/cartopy/mpl/gridliner.html
    #: A formatter which turns longitude values into nice longitudes such as 110W
    longitude_formatter = mticker.FuncFormatter(lambda v, pos:
                                                cgl._east_west_formatted(v))
    #: A formatter which turns latitude values into nice latitudes such as 45S
    latitude_formatter = mticker.FuncFormatter(lambda v, pos:
                                           cgl._north_south_formatted(v))

    # Label tranformation functions (to offsel labels from edge of plot)
    shift_dist_points = 5
    tr_t = ax.transAxes + mtrans.ScaledTranslation(0.0, 1 * shift_dist_points * (1.0 / 72), ax.figure.dpi_scale_trans)
    tr_b = ax.transAxes + mtrans.ScaledTranslation(0.0, -1 * shift_dist_points * (1.0 / 72), ax.figure.dpi_scale_trans)
    tr_l = ax.transAxes + mtrans.ScaledTranslation(-1 * shift_dist_points * (1.0 / 72), 0.0, ax.figure.dpi_scale_trans)
    tr_r = ax.transAxes + mtrans.ScaledTranslation(1 * shift_dist_points * (1.0 / 72), 0.0, ax.figure.dpi_scale_trans)


    # Define transform to be used with labels
    #   (horizontal shift on x and data on y)
    label_transform_left = mtrans.blended_transform_factory(x_transform=tr_l, y_transform=ax.transData)
    label_transform_right = mtrans.blended_transform_factory(x_transform=tr_r, y_transform=ax.transData)
    # Define transform to be used with labels
    #   (data on x and vertical shift on y)
    label_transform_top = mtrans.blended_transform_factory(x_transform=ax.transData, y_transform=tr_t)
    label_transform_bottom = mtrans.blended_transform_factory(x_transform=ax.transData, y_transform=tr_b)

    if draw_parallels:
        # Add Magnetic Parallels (lines of constant MLAT)

        # Use matplotlib ticker to find parallel line locations
        magnetic_parallels = ylocator.tick_values(mlat0, mlat1)
        magnetic_parallels = magnetic_parallels[(magnetic_parallels>=-90.) & (magnetic_parallels<=90.)]
        #print('MAGNETIC_PARALLELS', magnetic_parallels)

        # Define transform to be used with labels
        #   (horizontal shift on x and data on y)
        #label_transform = mtrans.blended_transform_factory(x_transform=tr_x, y_transform=ax.transData)

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

            # Add gridline labels
            if draw_labels:
                # Find intersection of gridline with edge of plot
                left, right, top, bottom = edge_intercept(ax, line)

                label_str = latitude_formatter(mlat)

                # transform is diferent for right and left side of plot
                # also need to sepecify different alignements
                for loc in left:
                    ax.text(0., loc, label_str, horizontalalignment='right', verticalalignment='center', color=line_params['color'], transform=label_transform_left)
                for loc in right:
                    ax.text(1., loc, label_str, horizontalalignment='left', verticalalignment='center', color=line_params['color'], transform=label_transform_right)
                for loc in bottom:
                    ax.text(loc, 0., label_str, horizontalalignment='center', verticalalignment='top', color=line_params['color'], transform=label_transform_bottom)
                for loc in top:
                    ax.text(loc, 1., label_str, horizontalalignment='center', verticalalignment='bottom', color=line_params['color'], transform=label_transform_top)


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

            # Add gridline labels
            if draw_labels:
                # Find intersection of gridline with edge of plot
                left, right, top, bottom = edge_intercept(ax, line)

                label_str = longitude_formatter(mlon)

                # transform is diferent for right and left side of plot
                # also need to sepecify different alignements
                for loc in left:
                    ax.text(0., loc, label_str, horizontalalignment='right', verticalalignment='center', color=line_params['color'], transform=label_transform_left)
                for loc in right:
                    ax.text(1., loc, label_str, horizontalalignment='left', verticalalignment='center', color=line_params['color'], transform=label_transform_right)
                for loc in bottom:
                    ax.text(loc, 0., label_str, horizontalalignment='center', verticalalignment='top', color=line_params['color'], transform=label_transform_bottom)
                for loc in top:
                    ax.text(loc, 1., label_str, horizontalalignment='center', verticalalignment='bottom', color=line_params['color'], transform=label_transform_top)


