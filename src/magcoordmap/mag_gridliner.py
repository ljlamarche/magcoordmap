# mag_gridliner.py
#import cartopy.mpl.gridliner as cart_gl
#import cartopy.crs as ccrs
from apexpy import Apex

import operator
import itertools

import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom

import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.gridliner import Gridliner
from cartopy.mpl.ticker import (
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeFormatter,
    LongitudeLocator,
)


#def maggridlines(ax, apex=None, apex_height=0., draw_labels=False,
#              xlocs=None, ylocs=None, dms=False,
#              x_inline=None, y_inline=None, auto_inline=True,
#              xformatter=None, yformatter=None, xlim=None, ylim=None,
#              rotate_labels=None, xlabel_style=None, ylabel_style=None,
#              labels_bbox_style=None, xpadding=5, ypadding=5,
#              offset_angle=25, auto_update=None, formatter_kwargs=None,
#              **kwargs):
def maggridlines(ax, apex=None, apex_height=0., **gl_kwargs):
    """
    Automatically add gridlines to the axes, in the given coordinate
    system, at draw time.

    Parameters
    ----------
    crs: optional
        The :class:`cartopy._crs.CRS` defining the coordinate system in
        which gridlines are drawn.
        Defaults to :class:`cartopy.crs.PlateCarree`.
    draw_labels: optional
        Toggle whether to draw labels. For finer control, attributes of
        :class:`Gridliner` may be modified individually. Defaults to False.

        - string: "x" or "y" to only draw labels of the respective
          coordinate in the CRS.
        - list: Can contain the side identifiers and/or coordinate
          types to select which ones to draw.
          For all labels one would use
          `["x", "y", "top", "bottom", "left", "right", "geo"]`.
        - dict: The keys are the side identifiers
          ("top", "bottom", "left", "right") and the values are the
          coordinates ("x", "y"); this way you can precisely
          decide what kind of label to draw and where.
          For x labels on the bottom and y labels on the right you
          could pass in `{"bottom": "x", "left": "y"}`.

        Note that, by default, x and y labels are not drawn on left/right
        and top/bottom edges respectively unless explicitly requested.

    xlocs: optional
        An iterable of gridline locations or a
        :class:`matplotlib.ticker.Locator` instance which will be
        used to determine the locations of the gridlines in the
        x-coordinate of the given CRS. Defaults to None, which
        implies automatic locating of the gridlines.
    ylocs: optional
        An iterable of gridline locations or a
        :class:`matplotlib.ticker.Locator` instance which will be
        used to determine the locations of the gridlines in the
        y-coordinate of the given CRS. Defaults to None, which
        implies automatic locating of the gridlines.
    dms: bool
        When default longitude and latitude locators and formatters are
        used, ticks are able to stop on minutes and seconds if minutes is
        set to True, and not fraction of degrees. This keyword is passed
        to :class:`~cartopy.mpl.gridliner.Gridliner` and has no effect
        if xlocs and ylocs are explicitly set.
    x_inline: optional
        Toggle whether the x labels drawn should be inline.
    y_inline: optional
        Toggle whether the y labels drawn should be inline.
    auto_inline: optional
        Set x_inline and y_inline automatically based on projection
    xformatter: optional
        A :class:`matplotlib.ticker.Formatter` instance to format labels
        for x-coordinate gridlines. It defaults to None, which implies the
        use of a :class:`cartopy.mpl.ticker.LongitudeFormatter` initiated
        with the ``dms`` argument, if the crs is of
        :class:`~cartopy.crs.PlateCarree` type.
    yformatter: optional
        A :class:`matplotlib.ticker.Formatter` instance to format labels
        for y-coordinate gridlines. It defaults to None, which implies the
        use of a :class:`cartopy.mpl.ticker.LatitudeFormatter` initiated
        with the ``dms`` argument, if the crs is of
        :class:`~cartopy.crs.PlateCarree` type.
    xlim: optional
        Set a limit for the gridlines so that they do not go all the
        way to the edge of the boundary. xlim can be a single number or
        a (min, max) tuple. If a single number, the limits will be
        (-xlim, +xlim).
    ylim: optional
        Set a limit for the gridlines so that they do not go all the
        way to the edge of the boundary. ylim can be a single number or
        a (min, max) tuple. If a single number, the limits will be
        (-ylim, +ylim).
    rotate_labels: optional, bool, str
        Allow the rotation of non-inline labels.

        - False: Do not rotate the labels.
        - True: Rotate the labels parallel to the gridlines.
        - None: no rotation except for some projections (default).
        - A float: Rotate labels by this value in degrees.

    xlabel_style: dict
        A dictionary passed through to ``ax.text`` on x label creation
        for styling of the text labels.
    ylabel_style: dict
        A dictionary passed through to ``ax.text`` on y label creation
        for styling of the text labels.
    labels_bbox_style: dict
        bbox style for all text labels.
    xpadding: float
        Padding for x labels. If negative, the labels are
        drawn inside the map.
    ypadding: float
        Padding for y labels. If negative, the labels are
        drawn inside the map.
    offset_angle: float
        Difference of angle in degrees from 90 to define when
        a label must be flipped to be more readable.
        For example, a value of 10 makes a vertical top label to be
        flipped only at 100 degrees.
    auto_update: bool, default=True
        Whether to update the gridlines and labels when the plot is
        refreshed.

        .. deprecated:: 0.23
           In future the gridlines and labels will always be updated.

    formatter_kwargs: dict, optional
        Options passed to the default formatters.
        See :class:`~cartopy.mpl.ticker.LongitudeFormatter` and
        :class:`~cartopy.mpl.ticker.LatitudeFormatter`

    Keyword Parameters
    ------------------
    **kwargs:
        All other keywords control line properties.  These are passed
        through to :class:`matplotlib.collections.Collection`.

    Returns
    -------
    gridliner
        A :class:`cartopy.mpl.gridliner.Gridliner` instance.

    Notes
    -----
    The "x" and "y" for locations and inline settings do not necessarily
    correspond to X and Y, but to the first and second coordinates of the
    specified CRS. For the common case of PlateCarree gridlines, these
    correspond to longitudes and latitudes. Depending on the projection
    used for the map, meridians and parallels can cross both the X axis and
    the Y axis.
    """

    #mgl = MagGridliner(
    #    ax, apex=apex, apex_height=apex_height, draw_labels=draw_labels, 
    #    xlocator=xlocs, ylocator=ylocs, collection_kwargs=kwargs, dms=dms,
    #    x_inline=x_inline, y_inline=y_inline, auto_inline=auto_inline,
    #    xformatter=xformatter, yformatter=yformatter,
    #    xlim=xlim, ylim=ylim, rotate_labels=rotate_labels,
    #    xlabel_style=xlabel_style, ylabel_style=ylabel_style,
    #    labels_bbox_style=labels_bbox_style,
    #    xpadding=xpadding, ypadding=ypadding, offset_angle=offset_angle,
    #    auto_update=auto_update, formatter_kwargs=formatter_kwargs)
    mgl = MagGridliner(ax, apex=apex, apex_height=apex_height, **gl_kwargs)
    ax.add_artist(mgl)

    return mgl


class MagGridliner(Gridliner):

    #def __init__(self, axes, apex=None, apex_height=0., draw_labels=False,
    #             xlocator=None, ylocator=None, collection_kwargs=None,
    #             xformatter=None, yformatter=None, dms=False,
    #             x_inline=None, y_inline=None, auto_inline=True,
    #             xlim=None, ylim=None, rotate_labels=None,
    #             xlabel_style=None, ylabel_style=None, labels_bbox_style=None,
    #             xpadding=5, ypadding=5, offset_angle=25,
    #             auto_update=None, formatter_kwargs=None):
    def __init__(self, axes, apex=None, apex_height=0., **gl_kwargs):

        #self.A = Apex()
        #self.apex_height = 0.

        # Create and Apex object for coordinate conversion if one is not provided
        if apex:
            self.A = apex
        else:
            self.A = Apex()
        self.apex_height = apex_height

        # Current method of blanket passing kwargs directly to super class works, but
        # need to figure out a way to modify such that defaulf values are set

        # Set gridline parameters based on defaults and provided keywords
        for param_key in ['collection_kwargs', 'xlabel_style', 'ylabel_style']:
            line_params = dict(color='orange')
            if param_key in gl_kwargs.keys():
                for key, value in gl_kwargs[param_key].items():
                    line_params[key] = value
            gl_kwargs[param_key] = line_params


       # label_style = dict(color='orange')
       # if 'xlabel_style' in gl_kwargs.keys():
       #     for key, value in gl_kwargs['xlabel_style'].items():
       #         line_params[key] = value
       # gl_kwargs['collection_kwargs'] = line_params

        #proj = ccrs.AzimuthalEquidistant(central_longitude=-147, central_latitude=64)
        proj = PlateCarree()
        #super().__init__(axes, proj, collection_kwargs={'color':'magenta'}, draw_labels=True, xlabel_style={'color':'purple'}, ylabel_style={'color':'green'})
        super().__init__(axes, proj, **gl_kwargs)
        #super().__init__(axes, proj, draw_labels=draw_labels, 
        #xlocator=xlocs, ylocator=ylocs, collection_kwargs=collection_kwargs, dms=dms,
        #x_inline=x_inline, y_inline=y_inline, auto_inline=auto_inline,
        #xformatter=xformatter, yformatter=yformatter,
        #xlim=xlim, ylim=ylim, rotate_labels=rotate_labels,
        #xlabel_style=xlabel_style, ylabel_style=ylabel_style,
        #labels_bbox_style=labels_bbox_style,
        #xpadding=xpadding, ypadding=ypadding, offset_angle=offset_angle,
        #auto_update=auto_update, formatter_kwargs=formatter_kwargs)

        #print(self.xline_artists)

        print('MagGridliner initialized!!')

        #print(self._axes_domain())


    def _axes_domain(self, nx=None, ny=None):
        """Return lon_range, lat_range"""
        DEBUG = False

        transform = self._crs_transform()

        ax_transform = self.axes.transAxes
        desired_trans = ax_transform - transform

        nx = nx or 100
        ny = ny or 100
        x = np.linspace(1e-9, 1 - 1e-9, nx)
        y = np.linspace(1e-9, 1 - 1e-9, ny)
        x, y = np.meshgrid(x, y)

        coords = np.column_stack((x.ravel(), y.ravel()))

        in_data = desired_trans.transform(coords)

        mlat, mlon = self.A.geo2apex(in_data[:,1], in_data[:,0], height=self.apex_height)
        apex_data = np.array([mlon, mlat]).T
        #print(in_data.shape, apex_data.shape)

        ax_to_bkg_patch = self.axes.transAxes - self.axes.patch.get_transform()

        # convert the coordinates of the data to the background patches
        # coordinates
        background_coord = ax_to_bkg_patch.transform(coords)
        ok = self.axes.patch.get_path().contains_points(background_coord)

        if DEBUG:
            import matplotlib.pyplot as plt
            plt.plot(coords[ok, 0], coords[ok, 1], 'or',
                     clip_on=False, transform=ax_transform)
            plt.plot(coords[~ok, 0], coords[~ok, 1], 'ob',
                     clip_on=False, transform=ax_transform)

        #inside = in_data[ok, :]
        inside = apex_data[ok, :]


        # If there were no data points in the axes we just use the x and y
        # range of the projection.
        if inside.size == 0:
            lon_range = self.crs.x_limits
            lat_range = self.crs.y_limits
        else:
            # np.isfinite must be used to prevent np.inf values that
            # not filtered by np.nanmax for some projections
            lat_max = np.compress(np.isfinite(inside[:, 1]),
                                  inside[:, 1])
            if lat_max.size == 0:
                lon_range = self.crs.x_limits
                lat_range = self.crs.y_limits
            else:
                lat_max = lat_max.max()
                lon_range = np.nanmin(inside[:, 0]), np.nanmax(inside[:, 0])
                lat_range = np.nanmin(inside[:, 1]), lat_max
        #print(lon_range, lat_range)

        # Will require careful handling of projections
        # Get back to this later
        ## XXX Cartopy specific thing. Perhaps make this bit a specialisation
        ## in a subclass...
        #crs = self.crs
        #if isinstance(crs, Projection):
        #    lon_range = np.clip(lon_range, *crs.x_limits)
        #    lat_range = np.clip(lat_range, *crs.y_limits)

        #    # if the limit is >90% of the full x limit, then just use the full
        #    # x limit (this makes circular handling better)
        #    prct = np.abs(np.diff(lon_range) / np.diff(crs.x_limits))
        #    if prct > 0.9:
        #        lon_range = crs.x_limits

        #if self.xlim is not None:
        #    if np.iterable(self.xlim):
        #        # tuple, list or ndarray was passed in: (-140, 160)
        #        lon_range = self.xlim
        #    else:
        #        # A single int/float was passed in: 140
        #        lon_range = (-self.xlim, self.xlim)

        #if self.ylim is not None:
        #    if np.iterable(self.ylim):
        #        # tuple, list or ndarray was passed in: (-140, 160)
        #        lat_range = self.ylim
        #    else:
        #        # A single int/float was passed in: 140
        #        lat_range = (-self.ylim, self.ylim)

        return lon_range, lat_range



    def _draw_gridliner(self, nx=None, ny=None, renderer=None):
        """Create Artists for all visible elements and add to our Axes.

        The following rules apply for the visibility of labels:

        - X-type labels are plotted along the bottom, top and geo spines.
        - Y-type labels are plotted along the left, right and geo spines.
        - A label must not overlap another label marked as visible.
        - A label must not overlap the map boundary.
        - When a label is about to be hidden, its padding is slightly
          increase until it can be drawn or until a padding limit is reached.
        """
        # Update only when needed or requested
        if self._drawn and not self._auto_update:
            return
        self._drawn = True

        # Inits
        lon_lim, lat_lim = self._axes_domain(nx=nx, ny=ny)
        print(lon_lim, lat_lim)
        transform = self._crs_transform()
        n_steps = self.n_steps
        crs = self.crs

        # Get nice ticks within crs domain
        # Generates lon_ticks as list but lat_ticks as ndarray??
        # This is probably something in the LatitudeLocator and LongitudeLocator classes in cartopy.mpl.ticker
        lon_ticks = self.xlocator.tick_values(lon_lim[0], lon_lim[1])
        lat_ticks = self.ylocator.tick_values(lat_lim[0], lat_lim[1])

        lon_ticks = list(lon_ticks)
        lat_ticks = list(lat_ticks)

        #print(lat_ticks)
        #print(lon_ticks)

        ## had to do with not exeededing geographic plot limits?
        ## not sure how to translate this properly, get back to it later
        #inf = max(lon_lim[0], crs.x_limits[0])
        #sup = min(lon_lim[1], crs.x_limits[1])
        #lon_ticks = [value for value in lon_ticks if inf <= value <= sup]
        #inf = max(lat_lim[0], crs.y_limits[0])
        #sup = min(lat_lim[1], crs.y_limits[1])
        #lat_ticks = [value for value in lat_ticks if inf <= value <= sup]

        print(lat_ticks)
        print(lon_ticks)

        #####################
        # Gridlines drawing #
        #####################

        collection_kwargs = self.collection_kwargs
        if collection_kwargs is None:
            collection_kwargs = {}
        collection_kwargs = collection_kwargs.copy()
        collection_kwargs['transform'] = transform
        if not any(x in collection_kwargs for x in ['c', 'color']):
            collection_kwargs.setdefault('color',
                                         matplotlib.rcParams['grid.color'])
        if not any(x in collection_kwargs for x in ['ls', 'linestyle']):
            collection_kwargs.setdefault('linestyle',
                                         matplotlib.rcParams['grid.linestyle'])
        if not any(x in collection_kwargs for x in ['lw', 'linewidth']):
            collection_kwargs.setdefault('linewidth',
                                         matplotlib.rcParams['grid.linewidth'])
        collection_kwargs.setdefault('clip_path', self.axes.patch)

        # Meridians
        lat_min, lat_max = lat_lim
        if lat_ticks:
            lat_min = min(lat_min, min(lat_ticks))
            lat_max = max(lat_max, max(lat_ticks))
        lon_lines = np.empty((len(lon_ticks), n_steps, 2))
        lon_lines[:, :, 0] = np.array(lon_ticks)[:, np.newaxis]
        lon_lines[:, :, 1] = np.linspace(
            lat_min, lat_max, n_steps)[np.newaxis, :]

        mlat_lines, mlon_lines, _ = self.A.apex2geo(lon_lines[:, :, 1], lon_lines[:, :, 0], height=self.apex_height)
        for i in range(mlon_lines.shape[0]):
            mlon_lines[i] = np.unwrap(mlon_lines[i], period=360.)
        lon_lines = np.array([mlon_lines, mlat_lines]).transpose((1,2,0))

        if self.xlines:
            nx = len(lon_lines) + 1
            # XXX this bit is cartopy specific. (for circular longitudes)
            # Purpose: omit plotting the last x line,
            # as it may overlap the first.
            if (isinstance(crs, Projection) and
                    isinstance(crs, _RectangularProjection) and
                    abs(np.diff(lon_lim)) == abs(np.diff(crs.x_limits))):
                nx -= 1

            if self.xline_artists:
                # Update existing collection.
                lon_lc, = self.xline_artists
                lon_lc.set(segments=lon_lines, **collection_kwargs)
            else:
                # Create new collection.
                lon_lc = mcollections.LineCollection(lon_lines,
                                                     **collection_kwargs)
                self.xline_artists.append(lon_lc)

        # Parallels
        lon_min, lon_max = lon_lim
        if lon_ticks:
            lon_min = min(lon_min, min(lon_ticks))
            lon_max = max(lon_max, max(lon_ticks))
        lat_lines = np.empty((len(lat_ticks), n_steps, 2))
        lat_lines[:, :, 0] = np.linspace(lon_min, lon_max,
                                         n_steps)[np.newaxis, :]
        lat_lines[:, :, 1] = np.array(lat_ticks)[:, np.newaxis]

        # When grid lines cross the international date line, they plot wierd
        # for normal cartopy coordinates, crossing the dateline at any point in the figure triggers
        #   lon limits of -180 -> +180 to be triggered
        # for magnetic coorinates, sometimes the magcoordinates will not roll over even when geo 
        #   coordinates cross the dateline (over AK is a good example)
        # In this case after converting to geo, coordinates jump from -180 to 180 at the crossing
        # Options:
        # 1. Set something so that when the date line is crossed, coordinates cover the entire globe
        #       and make sure to somehow line up so 180 geo is plotted on both ends
        # 2. Sanitize longitudes so they increase monotonically even crossing the date line
        # Solution: Use unwrap on longitude array after converting to geodetic so it's always increasing
        #   monotonically.  This needs to be done for both the parallels AND meridians.

        mlat_lines, mlon_lines, _ = self.A.apex2geo(lat_lines[:, :, 1], lat_lines[:, :, 0], height=self.apex_height)
        for i in range(mlon_lines.shape[0]):
            mlon_lines[i] = np.unwrap(mlon_lines[i], period=360.)
        lat_lines = np.array([mlon_lines, mlat_lines]).transpose((1,2,0))

        #print(lat_lines[0])

        if self.ylines:
            if self.yline_artists:
                # Update existing collection.
                lat_lc, = self.yline_artists
                lat_lc.set(segments=lat_lines, **collection_kwargs)
            else:
                lat_lc = mcollections.LineCollection(lat_lines,
                                                     **collection_kwargs)
                self.yline_artists.append(lat_lc)

        #################
        # Label drawing #
        #################

        # Clear drawn labels
        self._labels.clear()

        if not any((self.left_labels, self.right_labels, self.bottom_labels,
                    self.top_labels, self.inline_labels, self.geo_labels)):
            return
        self._assert_can_draw_ticks()

        # Inits for labels
        max_padding_factor = 5
        delta_padding_factor = 0.2
        spines_specs = {
            'left': {
                'index': 0,
                'coord_type': "x",
                'opcmp': operator.le,
                'opval': max,
            },
            'bottom': {
                'index': 1,
                'coord_type': "y",
                'opcmp': operator.le,
                'opval': max,
            },
            'right': {
                'index': 0,
                'coord_type': "x",
                'opcmp': operator.ge,
                'opval': min,
            },
            'top': {
                'index': 1,
                'coord_type': "y",
                'opcmp': operator.ge,
                'opval': min,
            },
        }
        for side, specs in spines_specs.items():
            bbox = self.axes.spines[side].get_window_extent(renderer)
            specs['coords'] = [
                getattr(bbox, specs['coord_type'] + idx) for idx in "01"]

        def update_artist(artist, renderer):
            artist.update_bbox_position_size(renderer)
            this_patch = artist.get_bbox_patch()
            this_path = this_patch.get_path().transformed(
                this_patch.get_transform())
            return this_path

        # Get the real map boundaries
        self.axes.spines["geo"].get_window_extent(renderer)  # update coords
        map_boundary_path = self.axes.spines["geo"].get_path().transformed(
            self.axes.spines["geo"].get_transform())
        map_boundary = sgeom.Polygon(map_boundary_path.vertices)

        if self.x_inline:
            y_midpoints = self._find_midpoints(lat_lim, lat_ticks)
        if self.y_inline:
            x_midpoints = self._find_midpoints(lon_lim, lon_ticks)

        # Cache a few things so they aren't re-calculated in the loops.
        crs_transform = self._crs_transform().transform
        inverse_data_transform = self.axes.transData.inverted().transform_point

        # Create a generator for the Label objects.
        generate_labels = self._generate_labels()

        for xylabel, lines, line_ticks, formatter, label_style in (
                ('x', lon_lines, lon_ticks,
                 self.xformatter, self.xlabel_style.copy()),
                ('y', lat_lines, lat_ticks,
                 self.yformatter, self.ylabel_style.copy())):

            x_inline = self.x_inline and xylabel == 'x'
            y_inline = self.y_inline and xylabel == 'y'
            padding = getattr(self, f'{xylabel}padding')
            bbox_style = self.labels_bbox_style.copy()
            if "bbox" in label_style:
                bbox_style.update(label_style["bbox"])
            label_style["bbox"] = bbox_style

            formatter.set_locs(line_ticks)

            for line_coords, tick_value in zip(lines, line_ticks):
                # Intersection of line with map boundary
                line_coords = crs_transform(line_coords)
                infs = np.isnan(line_coords).any(axis=1)
                line_coords = line_coords.compress(~infs, axis=0)
                if line_coords.size == 0:
                    continue
                line = sgeom.LineString(line_coords)
                if not line.intersects(map_boundary):
                    continue
                intersection = line.intersection(map_boundary)
                del line
                if intersection.is_empty:
                    continue
                if isinstance(intersection, sgeom.MultiPoint):
                    if len(intersection) < 2:
                        continue
                    n2 = min(len(intersection), 3)
                    tails = [[(pt.x, pt.y)
                              for pt in intersection[:n2:n2 - 1]]]
                    heads = [[(pt.x, pt.y)
                              for pt in intersection[-1:-n2 - 1:-n2 + 1]]]
                elif isinstance(intersection, (sgeom.LineString,
                                               sgeom.MultiLineString)):
                    if isinstance(intersection, sgeom.LineString):
                        intersection = [intersection]
                    elif len(intersection.geoms) > 4:
                        # Gridline and map boundary are parallel and they
                        # intersect themselves too much it results in a
                        # multiline string that must be converted to a single
                        # linestring. This is an empirical workaround for a
                        # problem that can probably be solved in a cleaner way.
                        xy = np.append(
                            intersection.geoms[0].coords,
                            intersection.geoms[-1].coords,
                            axis=0,
                        )
                        intersection = [sgeom.LineString(xy)]
                    else:
                        intersection = intersection.geoms
                    tails = []
                    heads = []
                    for inter in intersection:
                        if len(inter.coords) < 2:
                            continue
                        n2 = min(len(inter.coords), 8)
                        tails.append(inter.coords[:n2:n2 - 1])
                        heads.append(inter.coords[-1:-n2 - 1:-n2 + 1])
                    if not tails:
                        continue
                elif isinstance(intersection, sgeom.GeometryCollection):
                    # This is a collection of Point and LineString that
                    # represent the same gridline.  We only consider the first
                    # geometries, merge their coordinates and keep first two
                    # points to get only one tail ...
                    xy = []
                    for geom in intersection.geoms:
                        for coord in geom.coords:
                            xy.append(coord)
                            if len(xy) == 2:
                                break
                        if len(xy) == 2:
                            break
                    tails = [xy]
                    # ... and the last geometries, merge their coordinates and
                    # keep last two points to get only one head.
                    xy = []
                    for geom in reversed(intersection.geoms):
                        for coord in reversed(geom.coords):
                            xy.append(coord)
                            if len(xy) == 2:
                                break
                        if len(xy) == 2:
                            break
                    heads = [xy]
                else:
                    warnings.warn(
                        'Unsupported intersection geometry for gridline '
                        f'labels: {intersection.__class__.__name__}')
                    continue
                del intersection

                # Loop on head and tail and plot label by extrapolation
                for i, (pt0, pt1) in itertools.chain.from_iterable(
                        enumerate(pair) for pair in zip(tails, heads)):

                    # Initial text specs
                    x0, y0 = pt0
                    if x_inline or y_inline:
                        kw = {'rotation': 0, 'transform': self.crs,
                              'ha': 'center', 'va': 'center'}
                        loc = 'inline'
                    else:
                        x1, y1 = pt1
                        segment_angle = np.arctan2(y0 - y1,
                                                   x0 - x1) * 180 / np.pi
                        loc = self._get_loc_from_spine_intersection(
                            spines_specs, xylabel, x0, y0)
                        if not self._draw_this_label(xylabel, loc):
                            visible = False
                        kw = self._get_text_specs(segment_angle, loc, xylabel)
                        kw['transform'] = self._get_padding_transform(
                            segment_angle, loc, xylabel)
                    kw.update(label_style)

                    # Get x and y in data coords
                    pt0 = inverse_data_transform(pt0)
                    if y_inline:
                        # 180 degrees isn't formatted with a suffix and adds
                        # confusion if it's inline.
                        if abs(tick_value) == 180:
                            continue
                        x = x_midpoints[i]
                        y = tick_value
                        y, x, _ = self.A.apex2geo(y, x, self.apex_height)
                        kw.update(clip_on=True)
                        y_set = True
                    else:
                        x = pt0[0]
                        y_set = False

                    if x_inline:
                        if abs(tick_value) == 180:
                            continue
                        x = tick_value
                        y = y_midpoints[i]
                        y, x, _ = self.A.apex2geo(y, x, self.apex_height)
                        kw.update(clip_on=True)
                    elif not y_set:
                        y = pt0[1]

                    # Update generated label.
                    label = next(generate_labels)
                    text = formatter(tick_value)
                    artist = label.artist
                    artist.set(x=x, y=y, text=text, **kw)

                    # Update loc from spine overlapping now that we have a bbox
                    # of the label.
                    this_path = update_artist(artist, renderer)
                    if not x_inline and not y_inline and loc == 'geo':
                        new_loc = self._get_loc_from_spine_overlapping(
                            spines_specs, xylabel, this_path)
                        if new_loc and loc != new_loc:
                            loc = new_loc
                            transform = self._get_padding_transform(
                                segment_angle, loc, xylabel)
                            artist.set_transform(transform)
                            artist.update(
                                self._get_text_specs(
                                    segment_angle, loc, xylabel))
                            artist.update(label_style.copy())
                            this_path = update_artist(artist, renderer)

                    # Is this kind label allowed to be drawn?
                    if not self._draw_this_label(xylabel, loc):
                        visible = False

                    elif x_inline or y_inline:
                        # Check that it does not overlap the map.
                        # Inline must be within the map.
                        # TODO: When Matplotlib clip path works on text, this
                        # clipping can be left to it.
                        center = (artist
                                  .get_transform()
                                  .transform_point(artist.get_position()))
                        visible = map_boundary_path.contains_point(center)
                    else:
                        # Now loop on padding factors until it does not overlap
                        # the boundary.
                        visible = False
                        padding_factor = 1
                        while padding_factor < max_padding_factor:

                            # Non-inline must not run through the outline.
                            if map_boundary_path.intersects_path(
                                    this_path, filled=padding > 0):

                                # Apply new padding.
                                transform = self._get_padding_transform(
                                    segment_angle, loc, xylabel,
                                    padding_factor)
                                artist.set_transform(transform)
                                this_path = update_artist(artist, renderer)
                                padding_factor += delta_padding_factor

                            else:
                                visible = True
                                break

                    # Updates
                    label.set_visible(visible)
                    label.path = this_path
                    label.xy = xylabel
                    label.loc = loc
                    self._labels.append(label)

        # Now check overlapping of ordered visible labels
        if self._labels:
            self._labels.sort(
                key=operator.attrgetter("priority"), reverse=True)
            visible_labels = []
            for label in self._labels:
                if label.get_visible():
                    for other_label in visible_labels:
                        if label.check_overlapping(other_label):
                            break
                    else:
                        visible_labels.append(label)
