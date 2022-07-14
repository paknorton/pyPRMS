#!/usr/bin/env python3

from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize     # , LogNorm, PowerNorm
from matplotlib.patches import Polygon
from osgeo import ogr
from typing import Optional, Sequence, Set, Union

import cartopy.crs as ccrs
import copy
import geopandas
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj as prj
import shapely


def plot_line_collection(ax, geoms, values=None, cmap=None, norm=None, vary_width=False, vary_color=True, colors=None,
                         alpha=1.0, linewidth=1.0, **kwargs):
    """ Plot a collection of line geometries.
    """

    lines = []
    for geom in geoms:
        a = np.asarray(geom.coords)

        if geom.has_z:
            a = shapely.geometry.LineString(zip(*geom.xy))

        lines.append(shapely.geometry.LineString(a))

    if vary_width:
        lwidths = ((values / values.max()).to_numpy() + 0.01) * linewidth
        if vary_color:
            lines = LineCollection(lines, linewidths=lwidths, cmap=cmap, norm=norm, alpha=alpha)
        else:
            lines = LineCollection(lines, linewidths=lwidths, colors=colors, alpha=alpha)
    elif vary_color:
        lines = LineCollection(lines, linewidth=linewidth, alpha=alpha, cmap=cmap, norm=norm)

    if vary_color and values is not None:
        lines.set_array(values)
        # lines.set_cmap(cmap)

    ax.add_collection(lines, autolim=True)
    ax.autoscale_view()
    return lines


def plot_polygon_collection(ax, geoms, values=None, cmap=None, norm=None, facecolor=None, edgecolor=None,
                            alpha=1.0, linewidth=1.0, **kwargs):
    """ Plot a collection of Polygon geometries.
    """

    # from https://stackoverflow.com/questions/33714050/geopandas-plotting-any-way-to-speed-things-up
    patches = []

    for poly in geoms:

        a = np.asarray(poly.exterior)
        if poly.has_z:
            a = shapely.geometry.Polygon(zip(*poly.exterior.xy))

        patches.append(Polygon(a))

    # patches = PatchCollection(patches, facecolor=facecolor, linewidth=linewidth, edgecolor=edgecolor,
    #                           alpha=alpha, cmap=cmap, norm=norm)
    patches = PatchCollection(patches, match_original=False,
                              edgecolor='face', linewidth=linewidth, alpha=alpha, cmap=cmap, norm=norm, **kwargs)
    if values is not None:
        patches.set_array(values)
        # patches.set_cmap(cmap)

    ax.add_collection(patches, autolim=True)
    ax.autoscale_view()
    return patches


def get_projection(gdf: geopandas.GeoDataFrame):
    """Get projection of geodataframe.

    :param gdf: GeoDataFrame
    """

    aa = {}
    for yy in gdf.crs.coordinate_operation.params:
        aa[yy.name] = yy.value

    if '9822' in gdf.crs.coordinate_operation.method_code:
        # Albers Equal Area
        crs_proj = ccrs.AlbersEqualArea(central_longitude=aa['Longitude of false origin'],
                                        central_latitude=aa['Latitude of false origin'],
                                        standard_parallels=(aa['Latitude of 1st standard parallel'],
                                                            aa['Latitude of 2nd standard parallel']),
                                        false_easting=aa['Easting at false origin'],
                                        false_northing=aa['Northing at false origin'])
    elif '9802' in gdf.crs.coordinate_operation.method_code:
        # Lambert Conformal Conic
        crs_proj = ccrs.LambertConformal(central_latitude=aa['Latitude of false origin'],
                                         central_longitude=aa['Longitude of false origin'],
                                         standard_parallels=(aa['Latitude of 1st standard parallel'],
                                                             aa['Latitude of 2nd standard parallel']),
                                         false_easting=aa['Easting at false origin'],
                                         false_northing=aa['Northing at false origin'])
    else:
        # We're gonna crash
        crs_proj = None
    return crs_proj


def get_extent(shapefile: str,
               layer_name: Optional[str] = None,
               driver: Optional[str] = 'ESRI Shapefile'):
    """Get the extent from a shapefile.

    :param shapefile: name of shapefile or geodatabase
    :param layer_name: name of geodatabase layer
    :param driver: name of GDAL driver to use
    """

    # Get extent information from the national HRUs shapefile

    # Use gdal/ogr to get the extent information
    # Shapefile can be in projected coordinates
    # Driver can be: OpenFileGDB or ESRI Shapefile
    in_driver = ogr.GetDriverByName(driver)
    in_data_source = in_driver.Open(shapefile, 0)

    if layer_name is None:
        in_layer = in_data_source.GetLayer()
    else:
        in_layer = in_data_source.GetLayerByName(layer_name)

    extent = in_layer.GetExtent()

    # Get the spatial reference information from the shapefile
    spatial_ref = in_layer.GetSpatialRef()

    # Create transformation object using projection information from the shapefile
    xform = prj.Proj(spatial_ref.ExportToProj4())

    west, east, south, north = extent
    # pad = 100000.    # amount to pad the extent values with (in meters)
    # east += pad
    # west -= pad
    # south -= pad
    # north += pad

    ll_lon, ll_lat = xform(west, south, inverse=True)
    ur_lon, ur_lat = xform(east, north, inverse=True)
    print('\tExtent: ({0:f}, {1:f}, {2:f}, {3:f})'.format(west, east, south, north))
    print('\tExtent: (LL: [{}, {}], UR: [{}, {}])'.format(ll_lon, ll_lat, ur_lon, ur_lat))

    extent_dms = [ll_lon, ur_lon, ll_lat, ur_lat]

    # Matplotlib basemap requires the map center (lon_0, lat_0) be in decimal degrees
    # and yet the corners of the extent can be in projected coordinates
    cen_lon, cen_lat = xform((east+west)/2, (south+north)/2, inverse=True)

    print('cen_lon: {}'.format(cen_lon))
    print('cen_lat: {}'.format(cen_lat))

    return extent_dms


def set_colormap(the_var: str,
                 param_data: pd.DataFrame,
                 cmap: Optional[Union[str, colors.LinearSegmentedColormap]] = None,
                 min_val: Optional[Union[int, float]] = None,
                 max_val: Optional[Union[int, float]] = None,
                 **kwargs):
    """Set the colormap for a plot.

    :param the_var: name parameter
    :param param_data: parameter values
    :param cmap: colormap to use
    :param min_val: minimum value for color range
    :param max_val: maximum value for color range
    """

    # Create the colormap
    # cmap = 'BrBG' #'GnBu_r' # for snow
    # cmap = 'GnBu_r'
    # cmap = 'jet'

    if cmap is None:
        if the_var == 'tmax_allsnow':
            cmap = 'coolwarm'
        elif the_var in ['tmax', 'tmin']:
            cmap = 'bwr'
        elif the_var == 'tmax_allrain_offset':
            cmap = 'OrRd'
        elif the_var == 'snarea_thresh':
            cmap = 'tab20'
        elif the_var in ['net_ppt', 'net_rain', 'net_snow']:
            cmap = 'YlGnBu'
        elif the_var in ['tmax_cbh_adj', 'tmin_cbh_adj', 'jh_coef']:
            cmap = 'coolwarm'
        elif the_var in ['snow_cbh_adj', 'rain_cbh_adj']:
            cmap = 'YlGnBu'
        else:
            cmap = 'brg'

    # cmap = copy.copy(mpl.cm.get_cmap("nipy_spectral"))

    # Create the colormap if a list of names is given, otherwise use the given colormap
    if isinstance(cmap, (list, tuple)):
        lscm = mpl.colors.LinearSegmentedColormap
        cmap = lscm.from_list('mycm', cmap)
    else:
        cmap = copy.copy(plt.get_cmap(cmap))
        # if the_var == 'hru_deplcrv':
        #     num_col = abs(param_data.max().max() - param_data.min().min()) + 1
        #     cmap = plt.cm.get_cmap(name=cmap, lut=num_col)
        # else:
        #     cmap = plt.get_cmap(cmap)

    # missing_color = '#ff00cb'   # pink/magenta

    # Get the min and max values for the variable
    if max_val is None:
        max_val = param_data.max().max()

    if min_val is None:
        min_val = param_data.min().min()

    # norm = PowerNorm(gamma=0.05)
    # norm = LogNorm(vmin=min_val, vmax=max_val)

    # if min_val == 0.:
    #     if the_var in ['net_ppt', 'net_rain', 'net_snow']:
    #         # cmap.set_under(color='None')
    #         norm = LogNorm(vmin=0.000001, vmax=max_val)
    #     else:
    #         norm = Normalize(vmin=0.0, vmax=max_val)
    # else:
    # if the_var in ['tmax_hru', 'tmin_hru', 'tmax', 'tmin']:
    #     norm = Normalize(vmin=-max_val, vmax=max_val)
    # elif the_var in ['tmax_cbh_adj', 'tmin_cbh_adj', 'tmax_allsnow', 'jh_coef']:
    #     norm = Normalize(vmin=-max_val, vmax=max_val)
    # else:
    norm = Normalize(vmin=min_val, vmax=max_val)

    # TODO: 2022-06-17 PAN - Categorical variables require entry in two places.
    #       The first place is here for adjusting color boundaries and the second place is in
    #       Parameters.py for the labelling. Need to figure out a more consistent/better
    #       way to do this.
    if the_var in ['hru_deplcrv', 'soil_type']:
        # Adjust the boundaries; useful for qualitative colormaps
        # bnds = np.arange(param_data.min().min(), param_data.max().max()+2) - 0.5
        # num_col = abs(param_data.max().max() - param_data.min().min()) + 1
        bnds = np.arange(min_val, max_val+2) - 0.5
        num_col = abs(max_val - min_val) + 1
        norm = colors.BoundaryNorm(boundaries=bnds, ncolors=num_col)

    return cmap, norm


def main():
    pass
    # from collections import namedtuple
    #
    # # Data source
    # work_dir = '/Users/pnorton/tmp/tmp_paramdb'
    #
    # # Output-related variables
    # output_dir = '/Users/pnorton/tmp/v1_figs'
    # fversion = 'v11'  # version to include in output filenames
    #
    # # GIS-related variables
    # shpfile = '/Users/pnorton/Projects/National_Hydrology_Model/Trans-boundary_HRUs/GIS/GFv1.1_v2e.gdb'
    # layer_name = 'nhru_v11'
    # shape_key = 'nhru_v11'
    #
    # # Parameter information
    # Param = namedtuple('Param', ['name', 'min', 'max'])
    # params = {'tmax_allsnow': Param(name='tmax_allsnow', min=28.2, max=35.8),
    #           'tmax_allrain_offset': Param(name='tmax_allrain_offset', min=0.0, max=11.0)}
    #
    # the_var = params['tmax_allrain_offset']
    # time_index = 0
    #
    # print(f'Parameter name: {the_var.name}')
    # print(f'Starting time index: {time_index}')
    #
    # print('Reading parameter database')
    # pdb = ParamDb(paramdb_dir=work_dir, verbose=True, verify=True)
    #
    # # extent_dms = get_extent(shpfile, layer_name=layer_name, driver='OpenFileGDB')
    #
    # # Read the shapefile
    # print(f'Reading shapefile: {shpfile}')
    # hru_df = geopandas.read_file(shpfile, layer=layer_name)
    #
    # if hru_df.crs.name == 'USA_Contiguous_Albers_Equal_Area_Conic_USGS_version':
    #     print(f'Overriding USGS aea crs with EPSG:5070')
    #     hru_df.crs = 'EPSG:5070'
    #
    # # Get extent information
    # minx, miny, maxx, maxy = hru_df.geometry.total_bounds
    #
    # # param_var = pdb.parameters.get_dataframe(the_var).iloc[:]
    #
    # # Use the following for nhru x nmonths parameters
    # param_var = pdb.parameters.get_dataframe(the_var.name).iloc[:, time_index].to_frame(name=the_var.name)
    #
    # # Override for tmax_allsnow: min_val = 28.2, max_val = 35.8
    # # Override for tmax_allrain_offset: min_val = 0.0, max_val = 11.0
    # cmap, norm = set_colormap(the_var.name, param_var, min_val=the_var.min, max_val=the_var.max)
    #
    # crs_proj = get_projection(hru_df)
    #
    # print('Writing first plot')
    # # This takes care of multipolygons that are in the NHM geodatabase/shapefile
    # geoms_exploded = hru_df.explode().reset_index(level=1, drop=True)
    #
    # df_mrg = geoms_exploded.merge(param_var, left_on=shape_key, right_index=True, how='left')
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
    #
    # ax = plt.axes(projection=crs_proj)
    # ax.coastlines()
    # ax.gridlines()
    #
    # ax.set_extent([minx, maxx, miny, maxy], crs=crs_proj)
    # # ax.set_extent(extent_dms)
    #
    # mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # mapper.set_array(df_mrg[the_var.name])
    # plt.colorbar(mapper, shrink=0.6)
    #
    # # plt.title('Variable: {},  Date: {}'.format(the_var, xdf_df['time'].iloc[0].isoformat()))
    # plt.title(f'Variable: {the_var.name},  Month: {time_index+1}')
    # # plt.title('Variable: {}'.format(the_var))
    #
    # col = plot_polygon_collection(ax, df_mrg.geometry, values=df_mrg[the_var.name], colormap=cmap, norm=norm, linewidth=0.0)
    #
    # plt.savefig(f'{output_dir}/{the_var.name}_{fversion}_{time_index+1:02}.png', dpi=150, bbox_inches='tight')
    #
    # print('Writing remaining plots')
    # for tt in range(1, 12):
    #     print(f'    Index: {tt}')
    #     param_var = pdb.parameters.get_dataframe(the_var.name).iloc[:, tt].to_frame(name=the_var.name)
    #     df_mrg = geoms_exploded.merge(param_var, left_on=shape_key, right_index=True, how='left')
    #
    #     # ax.set_title('Variable: {},  Date: {}'.format(the_var, xdf_df['time'].iloc[0].isoformat()))
    #     ax.set_title(f'Variable: {the_var.name},  Month: {tt+1}')
    #     col.set_array(df_mrg[the_var.name])
    #     # fig
    #     plt.savefig(f'{output_dir}/{the_var.name}_{fversion}_{tt+1:02}.png', dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()
