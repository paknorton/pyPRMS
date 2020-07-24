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


def read_csv_pairs(filename1, filename2, dtype):
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

    if dtype == 'int':
        for lh, rh in zip(it, it2):
            data[lh.split(',')[1]] = int(rh.split(',')[1])
    if dtype == 'float':
        for lh, rh in zip(it, it2):
            data[int(lh.split(',')[1])] = float(rh.split(',')[1]) * 0.0015625

    return data


def nwis_site_type(poi_id):
    # Retrieve a single station and pull out the field names and data types
    stn_url = f'{base_url}/site/?format=rdb&sites={poi_id}&siteStatus=all'

    response = urlopen(stn_url)
    encoding = response.info().get_param('charset', failobj='utf8')
    streamgage_site_page = response.read().decode(encoding)

    # Strip the comment lines and field length lines from the result
    streamgage_site_page = t1.sub('', streamgage_site_page, 0)
    streamgage_site_page = t2.sub('', streamgage_site_page, 0)

    df = pd.read_csv(StringIO(streamgage_site_page), sep='\t', header=0)  # , usecols=['site_type_cd'])
    tp = df['site_tp_cd'].tolist()

    if len(set(tp)) > 1:
        print('MULTIPLE SITE_TYPES '*5)

    return tp[0]


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


def get_date_range(df, poi_id, param_cd, stat_cd):
    # Returns the period of record available for a given param code and statistic
    return df[(df['site_no'] == poi_id) & (df['parm_cd'] == param_cd) &
              (df['stat_cd'] == stat_cd)][['begin_date', 'end_date']].to_numpy()[0].tolist()


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

    reasons = {'multi': 'POI has multiple Q-columns',
               'noda': 'POI segment has no HRUs connected to it and no upstream connected segments (DA=0)',
               'nodis': 'POI does not provide parameter type 00060 (discharge)',
               'por': 'POI observations outside of POR or non-contiguous in POR',
               'bad': 'Bad site_no supplied',
               'del': 'poi_gage_segment points to segment removed in version 1.1',
               'zero': 'poi_gage_segment points to zero (not used)',
               'sitetype': 'POI has a non-streamflow site type'}
    reject_log = {}

    seg_cum_area = read_csv_pairs(f'{paramdb_dir}/nhm_seg.csv', f'{paramdb_dir}/seg_cum_area.csv', 'float')

    # Get the POI station IDs and segment from the paramdb version 1.1
    pdb_pois = read_csv_pairs(f'{paramdb_dir}/poi_gage_id.csv', f'{paramdb_dir}/poi_gage_segment.csv', 'int')
    poi_info(pdb_pois, 'paramdb_1.1')

    # POIs that have no branch in GF v1.1
    rejected_pdb_gone = ['01180000', '06175520', '06354490', '12113347', '12143700',
                         '12157250', '12158010', '12158040', '12202300', '12202420']

    rejected_pdb_multiQ = ['05051500']

    rejected_pdb_zero = []
    for xx, yy in pdb_pois.items():
        if yy == 0:
            rejected_pdb_zero.append(xx)

    print(f'NHM paramdb v1.1 counts')
    print(f'  No branches: {len(rejected_pdb_gone)}')
    print(f'  Segment=0: {len(rejected_pdb_zero)}')
    print(f'  Multiple-Q cols: {len(rejected_pdb_multiQ)}')

    # Remove rejected POIs in pdb_pois
    for xx in rejected_pdb_zero:
        reject_log[xx] = reasons['zero']
        del pdb_pois[xx]

    for xx in rejected_pdb_gone:
        reject_log[xx] = reasons['del']
        del pdb_pois[xx]

    for xx in rejected_pdb_multiQ:
        reject_log[xx] = reasons['multi']
        del pdb_pois[xx]

    # These are additional POIs that need to be manually added to accepted list
    addl_pois = {'04095380': 10024,
                 '06023100': 28828,
                 '06024020': 28894,
                 '06027600': 28980,
                 '06036805': 28650,
                 '06204070': 29526,
                 '06287800': 31545,
                 '06288400': 31537,
                 '06307990': 31461}

    # --------------------------------------------------------------------------
    # Load the version 1.0 POI station IDs and segments
    v1_pdb_pois = read_csv_pairs(f'{v1_pdb_src}/poi_gage_id.csv', f'{v1_pdb_src}/poi_gage_segment.csv', 'int')
    poi_info(v1_pdb_pois, 'paramdb_1.0')

    # Load the transboundary POI station IDs and segments
    tb_pdb_pois = read_csv_pairs(f'{tb_pdb_src}/poi_gage_id.csv', f'{tb_pdb_src}/poi_gage_segment.csv', 'int')
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

    print(f'GF v1.1')
    print(f'  Number of original POIs: {len(gf_pois)}')
    print(f'  Number of Additional POIs: {len(addl_pois)}')

    for xx, yy in addl_pois.items():
        gf_pois[xx] = yy

    print(f'  Total number of POIs: {len(gf_pois)}')

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

    # print('-'*60)
    # print(f'Number of GF POIs: {len(all_gages)}')
    # print(f'Number of HYDAT stations: {len(ec_gages)}')
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # Check paramdb v1.1 POIs for incorrect gage segment.
    # # Compares paramdb gage segment to GF gage segment
    # mis_cnt = 0
    # mis_v11_cnt = 0
    #
    # print('='*60)
    # for nhm_poi, nhm_seg in pdb_pois.items():
    #     if nhm_poi not in gf_pois:
    #         # print(f'NHM POI: {nhm_poi} not in GFv11')
    #         mis_v11_cnt += 1
    #     else:
    #         if nhm_seg != gf_pois[nhm_poi]:
    #             print(f'{nhm_poi} v11:{nhm_seg}, GF:{gf_pois[nhm_poi]}, v10:{v1_pdb_pois.get(nhm_poi)}, tb:{tb_pdb_pois.get(nhm_poi)}')
    #             mis_cnt += 1
    #
    # print('-'*60)
    # print(f'Number of POIs in paramdb v1.0: {len(pdb_pois)}')
    # print(f'Number of incorrect POIs segments: {mis_cnt}')
    # print(f'Number of POIs missing from GF: {mis_v11_cnt}')

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

    print(f'  Total duplicated site_no: {len(bb)}')
    print(f'  nwis_daily rows: {nwis_daily.shape[0]}')
    print(f'  Number of stations with multiple time-series: {len(nwis_multiple)}')

    # Remove the rows that have non-null loc_web_ds entries
    # nwis_daily = nwis_daily[~nwis_daily['site_no'].isin(nwis_multiple)]

    # NOTE: The line below was also stripping out sites that use loc_web_ds for
    #       comments or informationbut only had a single entry for the site
    # nwis_daily = nwis_daily[~nwis_daily['loc_web_ds'].notnull()]
    # print(f'nwis_daily rows: {nwis_daily.shape[0]}')

    # daily_sum = daily_df.groupby(['site_no'])['count_nu'].sum()

    # NOTE: will have to figure out how to get the sum of days by site later
    # nwis_cnt = nwis_daily.groupby(['site_no'])['count_nu'].sum()
    nwis_cnt = nwis_daily.groupby(['site_no'])['loc_web_ds'].nunique()
    nwis_cnt = nwis_cnt[nwis_cnt < 2]

    # ==========================================================================
    # Check if GF POIs exist in paramdb and if the NWIS site has discharge information
    not_in_site_list = []
    lake_site_cnt = 0
    rejected_type = {'ES': [],
                     'LK': [],
                     'SP': [],
                     'ST-TS': []}
    rejected_bad = []
    rejected_noDA = []
    rejected_noQ = []
    rejected_por = []
    rejected_multi = []
    accepted_pois = []
    rejected_log = {}

    print('-'*60)
    for poi, seg in gf_pois.items():
        if poi not in site_list:
            try:
                _ = int(poi)
                not_in_site_list.append(poi)

                if has_multiple_timeseries(nwis_sites_por, poi, '00060', '00003'):
                    # TODO: NOT included in final list
                    rejected_multi.append(poi)
                    reject_log[poi] = reasons['multi']

                    print_error(f'{poi} not in NWIS streamflow site list, has multiple timeseries **********')
                else:
                    if has_param_statistic(nwis_sites_por, poi, '00060', '00003'):
                        date_rng = get_date_range(nwis_sites_por, poi, '00060', '00003')
                        site_type = nwis_site_type(poi)

                        if poi in pdb_pois:
                            # TODO: NOT included in final list
                            if site_type != 'ST':
                                rejected_type[site_type].append(poi)
                                reject_log[poi] = f'{reasons["sitetype"]} ({site_type})'
                            else:
                                rejected_por.append(poi)
                                reject_log[poi] = f'{reasons["por"]} ({date_rng})'

                            print_warning(f'{poi} IS in paramdb, NOT in streamflow site list, HAS discharge information, {site_type}, {date_rng}')
                        else:
                            # TODO: NOT included in final list
                            if site_type != 'ST':
                                rejected_type[site_type].append(poi)
                                reject_log[poi] = f'{reasons["sitetype"]} ({site_type})'
                            else:
                                rejected_por.append(poi)
                                reject_log[poi] = f'{reasons["por"]} ({date_rng})'

                            print_warning(f'{poi} NOT in paramdb, NOT in streamflow site list, HAS discharge information, {site_type}, {date_rng}')
                    else:
                        if poi in pdb_pois:
                            # TODO: NOT included in final list
                            site_type = nwis_site_type(poi)

                            if site_type != 'ST':
                                rejected_type[site_type].append(poi)
                                reject_log[poi] = f'{reasons["sitetype"]} ({site_type})'
                            else:
                                rejected_noQ.append(poi)
                                reject_log[poi] = reasons['nodis']

                            if site_type == 'LK':
                                lake_site_cnt += 1

                            print_error(f'{poi} IS in paramdb, NOT in streamflow site list, does NOT have discharge information, type: {site_type}')
                        else:
                            # TODO: NOT included in final list
                            site_type = nwis_site_type(poi)

                            if site_type != 'ST':
                                rejected_type[site_type].append(poi)
                                reject_log[poi] = f'{reasons["sitetype"]} ({site_type})'
                            else:
                                rejected_noQ.append(poi)
                                reject_log[poi] = reasons['nodis']

                            if site_type == 'LK':
                                lake_site_cnt += 1
                            else:
                                print_error(f'{poi} NOT in paramdb, NOT in streamflow site list, does NOT have discharge, type: {site_type}')
                        # print(Fore.RED + f'{poi}' + Style.RESET_ALL + ' does not have discharge information')
            except HTTPError:
                # TODO: NOT included in final list
                rejected_bad.append(poi)
                reject_log[poi] = reasons['bad']
                print_error(f'*************************** HTTPError on {poi}')
            except ValueError:
                # HYDAT
                if seg_cum_area[seg] == 0.0:
                    # TODO: NOT included in final list
                    rejected_noDA.append(poi)
                    reject_log[poi] = f'{reasons["noda"]} (seg={seg})'
                    print_error(f'{poi} for segment {seg} has not connected HRUs')
                else:
                    # TODO: INCLUDED in final list
                    accepted_pois.append(poi)
                # pass
        else:
            # POIs that ARE in the streamflow site list
            try:
                # It's in the list of NWIS streamflow sites; show number of obs
                # print(f'{poi}: has {nwis_cnt[poi]} observations')
                _ = nwis_cnt[poi]
                if seg_cum_area[seg] == 0.0:
                    # TODO: NOT included in final list
                    rejected_noDA.append(poi)
                    reject_log[poi] = f'{reasons["noda"]} (seg={seg})'
                    print_error(f'{poi} for segment {seg} has no U/S segments and no connected HRUs')
                else:
                    # TODO: INCLUDED in final list
                    accepted_pois.append(poi)
            except KeyError:
                if poi not in site_list:
                    print_error(f'{poi} ******** SHOULD NOT HAPPEN***not in site_list')
                    exit()

                if has_multiple_timeseries(nwis_sites_por, poi, '00060', '00003'):
                    # TODO: NOT included in final list
                    rejected_multi.append(poi)
                    reject_log[poi] = reasons['multi']
                    print_error(f'{poi} IS in streamflow site list, has multiple timeseries **********')

                elif poi in pdb_pois:
                    if has_param_statistic(nwis_sites_por, poi, '00060', '00003'):
                        # TODO: NOT included in final list
                        date_rng = get_date_range(nwis_sites_por, poi, '00060', '00003')
                        site_type = nwis_site_type(poi)

                        rejected_por.append(poi)
                        reject_log[poi] = f'{reasons["por"]} ({date_rng})'
                        print_warning(f'{poi} IS in paramdb, IS in streamflow site list, NOT in daily statistics file, HAS discharge information, {date_rng}')
                    else:
                        # TODO: NOT included in final list
                        rejected_noQ.append(poi)
                        reject_log[poi] = reasons['nodis']
                        print_error(f'{poi} IS in paramdb, IS in streamflow site list, NOT in daily statistics file, does NOT have discharge information')
                    # print(f'WARNING: {poi} not in daily statistics')
                else:
                    if has_param_statistic(nwis_sites_por, poi, '00060', '00003'):
                        # TODO: NOT included in final list
                        date_rng = get_date_range(nwis_sites_por, poi, '00060', '00003')
                        site_type = nwis_site_type(poi)

                        rejected_por.append(poi)
                        reject_log[poi] = f'{reasons["por"]} ({date_rng})'
                        print_warning(f'{poi} NOT in paramdb, IS in streamflow site list, NOT in daily statistics file, HAS discharge information, {date_rng}')
                    else:
                        # TODO: NOT included in final list
                        rejected_noQ.append(poi)
                        reject_log[poi] = reasons['nodis']
                        print_error(f'{poi} NOT in paramdb, IS in streamflow site list, NOT in daily statistics file, does NOT have discharge information')

    print('~'*60)
    print(f'Number of GF v1.1 POIs not in NWIS streamflow site list: {len(not_in_site_list)}')

    print(f'Number of POIs in NHM v1.1 but not in GF v1.1: {len(set(pdb_pois.keys()) - gf_pois.keys())}')

    print('\nNumber of rejected POIs by site type')
    type_labels = {'ES': 'Estuary', 'LK': 'Lake', 'SP': 'Spring', 'ST-TS': 'Tidal stream'}

    for xx, yy in rejected_type.items():
        print(f'  {type_labels[xx]} ({xx}): {len(yy)}')

    print('\nNumber of rejected POIs')
    print(f'  No discharge: {len(rejected_noQ)}')
    print(f'  Incomplete/Outside POR: {len(rejected_por)}')
    print(f'  Multiple Q-cols: {len(rejected_multi)}')
    print(f'  Bad site_no: {len(rejected_bad)}')
    print(f'  No U/S segments or connected HRUs: {len(rejected_noDA)}')

    print(f'\nNumber of accepted POIs: {len(accepted_pois)}')
    print('='*60)

    # Write out the log of rejected sites

    # Check if there are duplicates
    print(f'Rejected sites check: {len(reject_log) - len(set(reject_log.keys()))}')
    print('-'*60)
    outlog = open('rejected_sites_log.tab', 'w')
    outlog.write('poi_id\tdescription\tflag\n')
    for xx, yy in reject_log.items():
        outlog.write(f'{xx}\t{yy}\t1\n')
    outlog.close()

    # Write the POI-related paramdb files
    poi_gage_hdl = open('poi_gage_id.csv', 'w')
    poi_seg_hdl = open('poi_gage_segment.csv', 'w')
    poi_type_hdl = open('poi_type.csv', 'w')

    poi_gage_hdl.write('$id,poi_gage_id\n')
    poi_seg_hdl.write('$id,poi_gage_segment\n')
    poi_type_hdl.write('$id,poi_type\n')

    type_log_hdl = open('poi_type_da.csv', 'w')
    type_log_hdl.write('poi,nwis_da,model_da,da_ratio\n')

    # print('*'*60)
    # print(seg_cum_area)
    # print('*'*60)

    poi_type_cnt = {0: 0,
                    1: 0}

    accepted_pois.sort()
    for nn, pp in enumerate(accepted_pois):
        poi_gage_hdl.write(f'{nn+1},{pp}\n')
        poi_seg_hdl.write(f'{nn+1},{gf_pois[pp]}\n')

        sca = seg_cum_area[gf_pois[pp]]
        poi_type = 0
        try:
            nwis_da = nwis_sites.loc[pp]['drainage_area']

            if sca < nwis_da:
                da_perc = sca / nwis_da
            else:
                da_perc = nwis_da / sca

            if da_perc >= 0.95:
                # Model DA to NWIS DA ratio is within +/- 5%
                poi_type = 1
        except KeyError:
            nwis_da = 0.0
            da_perc = 0.0

        poi_type_cnt[poi_type] += 1

        type_log_hdl.write(f'{pp},{nwis_da},{sca},{da_perc}\n')
        poi_type_hdl.write(f'{nn+1},{poi_type}\n')

    poi_gage_hdl.close()
    poi_seg_hdl.close()
    poi_type_hdl.close()
    type_log_hdl.close()

    print('Count of poi_type')
    print(f'  1 - model DA to NWIS DA +/-5%: {poi_type_cnt[1]} ')
    print(f'  0 - model DA to NWIS DA > +/-5%: {poi_type_cnt[0]}')


if __name__ == '__main__':
    main()
