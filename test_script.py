import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from apexpy import Apex
import magcoordmap as mcm

projections = [
    ccrs.PlateCarree(),
    ccrs.AlbersEqualArea(),
    ccrs.AzimuthalEquidistant(),
    ccrs.LambertConformal(),
    ccrs.LambertCylindrical(),
    ccrs.Mercator(),
    ccrs.Miller(),
    ccrs.Mollweide(),
    ccrs.Orthographic(),
    ccrs.Robinson(),
    ccrs.Sinusoidal(),
    ccrs.Stereographic(),
    ccrs.InterruptedGoodeHomolosine(emphasis='land'),
    ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=36.0, central_rotated_longitude=-106.0),
    ccrs.OSGB(approx=False),
    ccrs.EuroPP(),
    ccrs.Geostationary(),
    ccrs.NearsidePerspective(),
    ccrs.Gnomonic(),
    ccrs.LambertAzimuthalEqualArea(),
    ccrs.NorthPolarStereo(),
    ccrs.OSNI(approx=False),
    ccrs.SouthPolarStereo(),
]

def plot_all():
    fig = plt.figure(figsize=(8,10))
    gs = gridspec.GridSpec(int(len(projections)/4)+1, 4)
    for i, proj in enumerate(projections):
        ax = fig.add_subplot(gs[i], projection=proj)
        ax.coastlines()
        ax.gridlines(draw_labels=['left', 'bottom'])
        mcm.maggridlines(ax, draw_labels=['right', 'top'])
    plt.show()


def plot_single(proj):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines()
    ax.gridlines(draw_labels=['left', 'bottom'])
    mcm.maggridlines(ax, draw_labels=['right', 'top'])
    plt.show()

if __name__ == '__main__':
    plot_single(ccrs.Robinson())
    plot_all()
