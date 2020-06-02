#!/usr/bin/env python3

import argparse
from collections import OrderedDict
import networkx as nx
import os
import time

from pyPRMS.ParamDb import ParamDb
# from pyPRMS.ParameterFile import ParameterFile

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Generate headwaters files')
    parser.add_argument('-a', '--area', help='Maximum headwater area in square kilometers',
                        default=3000, type=float)
    parser.add_argument('--src', help='Source parameter database')
    parser.add_argument('--dst', help='Destination directory for output files')

    args = parser.parse_args()

    # workdir = '/Users/pnorton/tmp/check_paramdb_v11'
    max_area = args.area

    # Roland's original script used 0.00404685642 for conversion to sq km
    acre_to_sq_km = 0.0040468564224
    # sq_km_to_acre = 247.10538

    pdb = None

    print('-'*40)
    print('Read parameter database')
    t1 = time.time()
    try:
        pdb = ParamDb(paramdb_dir=args.src, verbose=True, verify=True)
    except FileNotFoundError:
        print(f'{args.src} is not a valid paramdb directory')
        exit(2)

    try:
        print(f'Creating output directory: {args.dst}')
        os.makedirs(args.dst)
    except OSError:
        print('\tUsing existing directory')

    params = pdb.parameters

    hru_segment_nhm = params['hru_segment_nhm'].tolist()
    hru_area = params['hru_area'].tolist()
    nhm_seg = params['nhm_seg'].tolist()

    print(f'Total number of segments: {len(nhm_seg)}')

    # Get mapping of segments to HRU ids
    seg_to_hru = params.seg_to_hru
    if 0 in seg_to_hru:
        # We don't want any non-routed HRUs
        print(f'Number of segments with no attached HRUs: {len(seg_to_hru[0])}')
        del seg_to_hru[0]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sum up the HRU area for each segment
    # This is used as a lookup table when computing the cumulative
    # area by segment
    area_by_seg = OrderedDict()

    for seg, harea in zip(hru_segment_nhm, hru_area):
        if seg in area_by_seg:
            area_by_seg[seg] += harea
        else:
            area_by_seg[seg] = harea
    print(f'\tParamdb read time: {time.time() - t1}')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get the stream network
    print('-'*40)
    print(f'Loading stream network')
    t1 = time.time()
    dag_streamnet = params.stream_network(tosegment='tosegment_nhm', seg_id='nhm_seg')

    print(f'Number of NHM downstream nodes: {dag_streamnet.number_of_nodes()}')
    print(f'Number of NHM downstream edges: {dag_streamnet.number_of_edges()}')
    print(f'Is a DAG: {nx.is_directed_acyclic_graph(dag_streamnet)}')
    print("Isolate nodes:")
    print(list(nx.isolates(dag_streamnet)))
    print(f'\tload network time: {time.time() - t1}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute cumulative drainage area by segment
    # Create upstream graph
    print('-'*40)
    print(f'Generate model subsets for all stream segments <= {max_area} square kilometers')
    t1 = time.time()
    dag_us = dag_streamnet.reverse()

    out_seg_cum_area = OrderedDict()
    out_seg_hw = OrderedDict()

    # Generate a subset for every segment in the parameter set
    for dsmost_seg in nhm_seg:
        uniq_seg_us = set()

        try:
            pred = nx.dfs_predecessors(dag_us, dsmost_seg)
            uniq_seg_us = uniq_seg_us.union(set(pred.keys()).union(set(pred.values())))
        except KeyError:
            print(f'\nKeyError: Segment {dsmost_seg} does not exist in stream network')

        dag_ds_subset = dag_streamnet.subgraph(uniq_seg_us).copy()

        node_outlets = [ee[0] for ee in dag_ds_subset.edges()]
        true_outlets = {dsmost_seg}.difference(set(node_outlets))

        # Add the downstream segments that exit the subgraph
        for xx in true_outlets:
            nhm_outlet = list(dag_streamnet.neighbors(xx))[0]
            dag_ds_subset.add_node(nhm_outlet)
            dag_ds_subset.add_edge(xx, nhm_outlet)

        toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges))

        csum = 0.0
        for xx in toseg_idx:
            csum += area_by_seg.get(xx, 0.0)

        # Convert acres to square kilometers
        csum *= acre_to_sq_km

        if 0.0 < csum <= max_area:
            out_seg_cum_area[dsmost_seg] = csum
            out_seg_hw[dsmost_seg] = toseg_idx
    print(f'\tmodel subset time: {time.time() - t1}')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build a list of HW segments that are ordered from largest headwater to
    # smallest.
    print('-'*40)
    print(f'Order headwaters by area largest to smallest')
    t1 = time.time()
    largest_hw = 0

    for kk, vv in out_seg_hw.items():
        if largest_hw < len(vv):
            largest_hw = len(vv)

    print(f'Number of segments in largest headwater area: {largest_hw}')

    seg_by_size = []

    while largest_hw != 0:
        for kk, vv in out_seg_hw.items():
            if len(vv) == largest_hw:
                seg_by_size.append(kk)
        largest_hw -= 1
    print(f'\tHW sort time: {time.time() - t1}')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove headwaters that are a subset of larger headwaters
    print('-'*40)
    print(f'Select largest unique headwaters')
    t1 = time.time()
    for xx in seg_by_size:
        remove_list = []

        if xx in out_seg_hw:
            for vv in out_seg_hw[xx]:
                if xx != vv and vv in out_seg_hw:
                    remove_list.append(vv)

            for rr in remove_list:
                del out_seg_hw[rr]

    print(f'Number of unique headwater areas less than or equal to {max_area} sq km: {len(out_seg_hw)}')
    print(f'\tUnique HW time: {time.time() - t1}')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output two files:
    # 1) The headwater segment file (used for headwater model extractions)
    # 2) Headwater IDs by HRU (used for mapping purposes)
    print('-'*40)
    print(f'Write hw_segs and hw_hrus files')
    hru_by_hw = OrderedDict()

    fhdl_hw_segs = open(f'{args.dst}/hw_segs.csv', 'w')
    fhdl_hw_segs.write('hw_id,start_seg,child_segs\n')

    fhdl_hru_hw = open(f'{args.dst}/hw_hrus.csv', 'w')
    fhdl_hru_hw.write('nhm_id,hw_id\n')

    fhdl_hw_area = open(f'{args.dst}/hw_area.csv', 'w')
    fhdl_hw_area.write('hw_id,seg_outlet,seg_cum_area_sqkm\n')

    cnt = 1
    for kk, vv in out_seg_hw.items():
        out_seg_str = ','.join([str(xx) for xx in vv])
        fhdl_hw_segs.write(f'{cnt},{kk},{out_seg_str}\n')

        fhdl_hw_area.write(f'{cnt},{kk},{out_seg_cum_area[kk]}\n')

        for ss in vv:
            if ss in seg_to_hru:
                # Some segments have no HRUs. If they have HRUs we'll process them.
                for hh in seg_to_hru[ss]:
                    hru_by_hw[hh] = cnt
                    fhdl_hru_hw.write(f'{hh},{cnt}\n')
        cnt += 1

    fhdl_hw_segs.close()
    fhdl_hru_hw.close()
    fhdl_hw_area.close()
    print('.. done!')
    print('='*40)


if __name__ == '__main__':
    main()
