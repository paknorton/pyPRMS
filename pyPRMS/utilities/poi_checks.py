#!/usr/bin/env python3

import datetime
from collections import OrderedDict

import colorful as cf
# import colorama
# from colorama import Fore, Back, Style
import numpy as np
import pandas as pd
import re
import sys
from io import StringIO

from urllib.request import urlopen, Request
from urllib.error import HTTPError

__author__ = 'Parker Norton (pnorton@usgs.gov)'

# URLs can be generated/tested at: http://waterservices.usgs.gov/rest/Site-Test-Tool.html
base_url = 'http://waterservices.usgs.gov/nwis'

t1 = re.compile('^#.*$\n?', re.MULTILINE)   # remove comment lines
t2 = re.compile('^5s.*$\n?', re.MULTILINE)  # remove field length lines


def read_csv_pairs(filename1, filename2):
    # Given two paramdb files this will return an ordered dictionary of the 2nd columns of each file
    # where the first file is the key, second file is the value
    fhdl = open(filename1)
    rawdata = fhdl.read().splitlines()
    fhdl.close()
    it = iter(rawdata)

    fhdl2 = open(filename2)
    rawdata2 = fhdl2.read().splitlines()
    fhdl2.close()
    it2 = iter(rawdata2)

    data = OrderedDict()
    next(it)
    next(it2)

    for lh, rh in zip(it, it2):
        data[lh.split(',')[1]] = int(rh.split(',')[1])
    return data


def nwis_site_param_cd(poi_id):
    col_names = ['agency_cd', 'site_no', 'station_nm', 'site_tp_cd', 'dec_lat_va', 'dec_long_va',
                 'coord_acy_cd', 'dec_coord_datum_cd', 'alt_va', 'alt_acy_va', 'alt_datum_cd', 'huc_cd',
                 'data_type_cd', 'parm_cd', 'stat_cd', 'ts_id', 'loc_web_ds', 'medium_grp_cd',
                 'parm_grp_cd', 'srs_id', 'access_cd', 'begin_date', 'end_date', 'count_nu']
    col_types = [np.str_, np.str_, np.str_, np.str_, np.float, np.float,
                 np.str_, np.str_, np.float, np.float, np.str_, np.str_,
                 np.str_, np.str_, np.str_, np.int, np.str_, np.str_,
                 np.str_, np.int, np.str_, np.str_, np.str_, np.int]
    cols = dict(zip(col_names, col_types))

    # Retrieve a single station and pull out the field names and data types
    stn_url = f'{base_url}/site/?format=rdb&sites={poi_id}&seriesCatalogOutput=true&siteStatus=all'

    response = urlopen(stn_url)
    encoding = response.info().get_param('charset', failobj='utf8')
    streamgage_site_page = response.read().decode(encoding)

    # Strip the comment lines and field length lines from the result
    streamgage_site_page = t1.sub('', streamgage_site_page, 0)
    streamgage_site_page = t2.sub('', streamgage_site_page, 0)

    df = pd.read_csv(StringIO(streamgage_site_page), sep='\t', dtype=cols, usecols=['site_no', 'parm_cd'])

    return df['parm_cd'].tolist()


def nwis_site_fields():
    # Retrieve a single station and pull out the field names and data types
    stn_url = f'{base_url}/site/?format=rdb&sites=01646500&siteOutput=expanded&siteStatus=active&parameterCd=00060&siteType=ST'

    response = urlopen(stn_url)
    encoding = response.info().get_param('charset', failobj='utf8')
    streamgage_site_page = response.read().decode(encoding)

    # Strip the comment lines and field length lines from the result
    streamgage_site_page = t1.sub('', streamgage_site_page, 0)

    nwis_dtypes = t2.findall(streamgage_site_page)[0].strip('\n').split('\t')
    nwis_fields = StringIO(streamgage_site_page).getvalue().split('\n')[0].split('\t')

    nwis_final = {}
    for fld in nwis_fields:
        code = fld[-2:]
        if code in ['cd', 'no', 'nm', 'dt']:
            nwis_final[fld] = np.str_
        elif code in ['va']:
            nwis_final[fld] = np.float32
        else:
            nwis_final[fld] = np.str_

    return nwis_final


def nwis_site_simple_retrieve():
    cols = nwis_site_fields()

    # Columns to include in the final dataframe
    include_cols = ['agency_cd', 'site_no', 'station_nm', 'dec_lat_va', 'dec_long_va', 'dec_coord_datum_cd',
                    'alt_va', 'alt_datum_cd', 'huc_cd', 'drain_area_va', 'contrib_drain_area_va']

    # Start with an empty dataframe
    nwis_sites = pd.DataFrame(columns=include_cols)

    for region in range(19):
        # region = '01'
        print(f'Region {region+1:02}')
        stn_url = f'{base_url}/site/?format=rdb&huc={region+1:02}&siteOutput=expanded&siteStatus=all&parameterCd=00060&siteType=ST'

        response = urlopen(stn_url)
        encoding = response.info().get_param('charset', failobj='utf8')
        streamgage_site_page = response.read().decode(encoding)

        # Strip the comment lines and field length lines from the result
        streamgage_site_page = t1.sub('', streamgage_site_page, 0)
        streamgage_site_page = t2.sub('', streamgage_site_page, 0)

        # Read the rdb file into a dataframe
        df = pd.read_csv(StringIO(streamgage_site_page), sep='\t', dtype=cols, usecols=include_cols)

        nwis_sites = nwis_sites.append(df, ignore_index=True)

    nwis_sites.set_index('site_no', inplace=True)
    return nwis_sites


def nwis_load_site_por_info():
    col_names = ['agency_cd', 'site_no', 'station_nm', 'site_tp_cd', 'dec_lat_va', 'dec_long_va',
                 'coord_acy_cd', 'dec_coord_datum_cd', 'alt_va', 'alt_acy_va', 'alt_datum_cd', 'huc_cd',
                 'data_type_cd', 'parm_cd', 'stat_cd', 'ts_id', 'loc_web_ds', 'medium_grp_cd',
                 'parm_grp_cd', 'srs_id', 'access_cd', 'begin_date', 'end_date', 'count_nu']
    col_types = [np.str_, np.str_, np.str_, np.str_, np.float, np.float,
                 np.str_, np.str_, np.float, np.float, np.str_, np.str_,
                 np.str_, np.str_, np.str_, np.int, np.str_, np.str_,
                 np.str_, np.int, np.str_, np.str_, np.str_, np.int]
    cols = dict(zip(col_names, col_types))

    nwis_sites_exp = pd.read_csv('/Users/pnorton/Projects/National_Hydrology_Model/datasets/streamflow/nwis_sites_por.tab',
                                 sep='\t', dtype=cols, index_col=0)
    return nwis_sites_exp


def has_multiple_timeseries(df, poi_id, param_cd, stat_cd):
    return df[(df['site_no'] == poi_id) & (df['parm_cd'] == param_cd) &
              (df['stat_cd'] == stat_cd) & (df['loc_web_ds'].notnull())].shape[0] > 1


def has_param_statistic(df, poi_id, param_cd, stat_cd):
    # Returns true if poi_id  has a single entry for the given param_cd and stat_cd
    return df[(df['site_no'] == poi_id) & (df['parm_cd'] == param_cd) &
              (df['stat_cd'] == stat_cd)].shape[0] == 1


def nwis_load_daily_statistics(src_dir):
    col_names = ['agency_cd', 'site_no', 'parameter_cd', 'ts_id', 'loc_web_ds', 'month_nu', 'day_nu',
                 'begin_yr', 'end_yr', 'count_nu', 'mean_va']
    col_types = [np.str_, np.str_, np.str_, np.int, np.str_, np.int, np.int, np.int, np.int, np.int, np.float]
    cols = dict(zip(col_names, col_types))

    # Start with an empty dataframe
    nwis_daily = pd.DataFrame(columns=col_names)

    for region in range(18):
        # region = '01'
        sys.stdout.write(f'\rRegion: {region+1:02}')
        sys.stdout.flush()
        # print(f'Region {region+1:02}')

        # Read the rdb file into a dataframe
        df = pd.read_csv(f'{src_dir}/conus_daily_HUC_{region+1:02}_obs.tab', sep='\t', dtype=cols)

        nwis_daily = nwis_daily.append(df, ignore_index=True)
    print('')
    return nwis_daily


def poi_info(poi_dict, msg=''):
    print('-'*60)
    print(f'{msg} Number of POIs: {len(poi_dict)}')
    print(f'{msg} Number of unique POIs: {len(set(poi_dict))}')
    print(f'{msg} Number of POI segments: {len(poi_dict.values())}')
    print(f'{msg} Number of unique POI segments: {len(set(poi_dict.values()))}')


def print_error(txt):
    print(cf.red('ERROR: ') + txt)
    # print(Fore.RED + 'ERROR: ' + Style.RESET_ALL + txt)


def print_warning(txt):
    print(cf.orange('WARNING: ') + txt)


def main():
    cf.use_256_ansi_colors()
    # colorama.init()

    # The poi_agency fields are: GNIS_Name,Type_Gage,Type_Ref,Gage_Source,poi_segment_v1_1
    workdir = '/Users/pnorton/Projects/National_Hydrology_Model/Trans-boundary_HRUs/GIS'
    gf_poi_filename = f'{workdir}/poi_agency.csv'

    paramdb_dir = '/Users/pnorton/Projects/National_Hydrology_Model/datasets/paramdb_v11/paramdb_v11_gridmet_CONUS'

    v1_pdb_src = '/Users/pnorton/Projects/National_Hydrology_Model/Trans-boundary_HRUs/v1_paramdb'
    tb_pdb_src = '/Users/pnorton/Projects/National_Hydrology_Model/Trans-boundary_HRUs/tbparamdb'

    # Get the POI station IDs and segment from the paramdb
    pdb_pois = read_csv_pairs(f'{paramdb_dir}/poi_gage_id.csv', f'{paramdb_dir}/poi_gage_segment.csv')
    poi_info(pdb_pois, 'paramdb_1.1')

    # Load the version 1.0 POI station IDs and segments
    v1_pdb_pois = read_csv_pairs(f'{v1_pdb_src}/poi_gage_id.csv', f'{v1_pdb_src}/poi_gage_segment.csv')
    poi_info(v1_pdb_pois, 'paramdb_1.0')

    # Load the transboundary POI station IDs and segments
    tb_pdb_pois = read_csv_pairs(f'{tb_pdb_src}/poi_gage_id.csv', f'{tb_pdb_src}/poi_gage_segment.csv')
    poi_info(tb_pdb_pois, 'TB_v1.1')

    # Build ordered dictionary of geospatial fabric POIs
    # key->gage_id, value->gage_seg
    fhdl = open(gf_poi_filename, 'r')
    rawdata = fhdl.read().splitlines()
    fhdl.close()
    it = iter(rawdata)
    next(it)

    gf_pois = OrderedDict()

    for row in it:
        flds = row.strip().split(',')
        gage_id = flds[1]
        gage_seg = int(flds[4])

        gf_pois[gage_id] = gage_seg
    poi_info(gf_pois, 'gf_v1.1')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check consistency of geospatial fabric POIs and if they're in the paramdb v1.1
    fhdl = open(gf_poi_filename, 'r')

    all_gages = []
    ec_gages = []

    print('='*60)
    for row in fhdl:
        flds = row.strip().split(',')
        gage_id = flds[1]
        gage_src = flds[3]

        if gage_id in pdb_pois:
            if gage_src != 'EC':
                if gage_src == '0':
                    print(f'{gage_id} in PARAMDB but has no gage_src')
                try:
                    gage_id_int = int(gage_id)
    #                 print(f'{gage_id} in PARAMDB ({gage_src})')

                    if len(gage_id) > 15:
                        print_error(f'{gage_id} is not a USGS gage')
                        # print(Fore.RED + f'{gage_id} is not a USGS gage' + Style.RESET_ALL)
                except ValueError:
                    print_error(f'{gage_id} incorrectly sourced to {gage_src}')
                    # print(Fore.RED + f'{gage_id} incorrectly sourced to {gage_src}' + Style.RESET_ALL)

                    if gage_id != 'Type_Gage':
                        ec_gages.append(gage_id)
            elif gage_src == 'EC':
                if len(gage_id) > 7:
                    print_error(f'{gage_id} incorrectly sourced to {gage_src}')
                    # print(f'{gage_id} incorrectly sourced to {gage_src}')
                else:
                    ec_gages.append(gage_id)
            all_gages.append(gage_id)
        else:
            # print(f'{gage_id} not included in paramdb ({gage_src})')
            pass
    fhdl.close()

    print('-'*60)
    print(f'Number of GF POIs: {len(all_gages)}')
    print(f'Number of HYDAT stations: {len(ec_gages)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check paramdb v1.1 POIs for incorrect gage segment.
    # Compares paramdb gage segment to GF gage segment
    mis_cnt = 0
    mis_v11_cnt = 0

    print('='*60)
    for nhm_poi, nhm_seg in pdb_pois.items():
        if nhm_poi not in gf_pois:
            # print(f'NHM POI: {nhm_poi} not in GFv11')
            mis_v11_cnt += 1
        else:
            if nhm_seg != gf_pois[nhm_poi]:
                print(f'{nhm_poi} v11:{nhm_seg}, GF:{gf_pois[nhm_poi]}, v10:{v1_pdb_pois.get(nhm_poi)}, tb:{tb_pdb_pois.get(nhm_poi)}')
                mis_cnt += 1

    print('-'*60)
    print(f'Number of POIs in paramdb v1.0: {len(pdb_pois)}')
    print(f'Number of incorrect POIs segments: {mis_cnt}')
    print(f'Number of POIs missing from GF: {mis_v11_cnt}')

    # ==========================================================================
    # ==========================================================================
    # Load NWIS stations
    # Retrieve the station information by HUC2
    print('Loading simple site information')

    # Retrieve simple site information from NWIS water service
    # nwis_sites = nwis_site_simple_retrieve()

    # Load simple site information from file
    col_names = ['poi_id', 'poi_agency', 'poi_name', 'latitude', 'longitude',
                 'dec_coord_datum_cd', 'elevation', 'alt_datum_cd', 'huc_cd',
                 'drainage_area', 'drainage_area_contrib']
    col_types = [np.str_, np.str_, np.str_, np.float, np.float,
                 np.str_, np.float, np.str_, np.str_,
                 np.float, np.float]
    cols = dict(zip(col_names, col_types))

    nwis_sites = pd.read_csv('/Users/pnorton/Projects/National_Hydrology_Model/datasets/streamflow/nwis_sites_simple.tab',
                             sep='\t', dtype=cols)
    nwis_sites.set_index('poi_id', inplace=True)

    site_list = nwis_sites.index.tolist()

    # ==========================================================================
    # ==========================================================================
    # Load POR information
    nwis_sites_por = nwis_load_site_por_info()

    # ==========================================================================
    # ==========================================================================
    print('='*60)
    print('Load NWIS mean daily statistics')
    daily_src = '/Users/pnorton/Projects/National_Hydrology_Model/datasets/streamflow'

    nwis_daily = nwis_load_daily_statistics(daily_src)

    # Get set of stations that have multiple time-series for a single ID
    nwis_multiple = set(nwis_daily[nwis_daily['loc_web_ds'].notnull()]['site_no'].tolist())
    bb = nwis_daily[nwis_daily['loc_web_ds'].notnull()]['site_no'].tolist()

    print(f'Total duplicated site_no: {len(bb)}')
    print(f'nwis_daily rows: {nwis_daily.shape[0]}')
    print(f'Number of stations with multiple time-series: {len(nwis_multiple)}')

    # Remove the rows that have non-null loc_web_ds entries
    # nwis_daily = nwis_daily[~nwis_daily['site_no'].isin(nwis_multiple)]
    nwis_daily = nwis_daily[~nwis_daily['loc_web_ds'].notnull()]
    print(f'nwis_daily rows: {nwis_daily.shape[0]}')

    # daily_sum = daily_df.groupby(['site_no'])['count_nu'].sum()
    nwis_cnt = nwis_daily.groupby(['site_no'])['count_nu'].sum()

    # ==========================================================================
    # Check if GF POIs exist in paramdb and if the NWIS site has discharge information
    not_in_site_list = []

    print('-'*60)
    for poi, seg in gf_pois.items():
        if poi not in site_list:
            try:
                _ = int(poi)
                not_in_site_list.append(poi)

                if has_multiple_timeseries(nwis_sites_por, poi, '00060', '00003'):
                    print_error(f'{poi} not in NWIS streamflow site list but has multiple timeseries')
                else:
                    if has_param_statistic(nwis_sites_por, poi, '00060', '00003'):
                        # if '00060' not in nwis_site_param_cd(poi):
                        if poi in pdb_pois:
                            print_warning(f'{poi} not in NWIS streamflow site list, IS in paramdb, it does have discharge information')
                        else:
                            print_warning(f'{poi} not in NWIS streamflow site list, NOT in paramdb, does have discharge information')
                    else:
                        if poi in pdb_pois:
                            print_error(f'{poi} not in NWIS streamflow site list, IS in paramdb, does NOT have discharge information')
                        else:
                            print_error(f'{poi} not in NWIS streamflow site list, NOT in paramdb, does NOT have discharge information')
                        # print(Fore.RED + f'{poi}' + Style.RESET_ALL + ' does not have discharge information')

            except ValueError:
                # Skipping HYDAT
                pass
        else:
            try:
                # It's in the list of NWIS streamflow sites; show number of obs
                # print(f'{poi}: has {nwis_cnt[poi]} observations')
                _ = nwis_cnt[poi]
            except KeyError:
                if poi not in site_list:
                    print_error(f'{poi} ******** SHOULD NOT HAPPEN***not in site_list')
                    # print(f'                             WARNING: {poi} not in site_list')
                if has_multiple_timeseries(nwis_sites_por, poi, '00060', '00003'):
                    print_error(f'{poi} valid NWIS streamflow site but has multiple timeseries **********')

                if poi in pdb_pois:
                    if has_param_statistic(nwis_sites_por, poi, '00060', '00003'):
                        print_warning(f'{poi} in paramdb, has discharge but not in daily statistics file')
                    else:
                        print_error(f'{poi} in paramdb, not in daily statistics file, does NOT have discharge')
                    # print(f'WARNING: {poi} not in daily statistics')
                else:
                    if has_param_statistic(nwis_sites_por, poi, '00060', '00003'):
                        print_warning(f'{poi} NOT in paramdb, has discharge but not in daily statistics file')
                    else:
                        print_error(f'{poi} NOT in paramdb, not in daily statistics file, does NOT have discharge')

    print('~'*60)
    print(f'Number of POIs not in NWIS streamflow site list: {len(not_in_site_list)}')


if __name__ == '__main__':
    main()
