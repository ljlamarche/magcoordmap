import pytest
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import matplotlib.ticker as mticker
import cartopy.crs as ccrs
#from apexpy import Apex
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


@pytest.mark.parametrize('proj', projections)
def test_full_map(proj):
    proj_name = type(proj).__name__
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines()
    #ax.gridlines(draw_labels=['left', 'bottom'])
    #mcm.maggridlines(ax, draw_labels=['right', 'top'])
    ax.gridlines(draw_labels=True)
    mcm.maggridlines(ax, draw_labels=True)
    ax.set_title(proj_name)
    plt.savefig(f'test_{proj_name}.png')
    plt.close()


