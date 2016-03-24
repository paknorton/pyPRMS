

# Create date: 2015-05-19
# Author: Parker Norton (pnorton@usgs.gov)
# Description: Misc helper functions for calibration.
#              Most likely many of these will get move elsewhere eventually

import numpy as np
import prms_lib as prms

def read_default_params(filename):
    # Read in the default parameter ranges
    default_rng_file = open(filename, 'r')
    raw_range = default_rng_file.read().splitlines()
    default_rng_file.close()
    it = iter(raw_range)
    
    def_ranges = {}
    it.next()
    for line in it:
        flds = line.split(' ')
        def_ranges[flds[0]] = {'max': float(flds[1]), 'min': float(flds[2])}
    
    return def_ranges


def read_sens_params(filename, include_params=(), exclude_params=()):
    # Read in the sensitive parameters
    
    try:
        sensparams_file = open(filename, 'r')
    except IOError:
        #print "\tERROR: Missing hruSens.csv file for %s... skipping" % bb
        print '\tERROR: Missing %s file... skipping' % filename
        return {}
        
    rawdata = sensparams_file.read().splitlines()
    sensparams_file.close()
    it = iter(rawdata)

    counts = {}
    for line in it:
        flds = line.split(',')
        for ff in flds:
            ff = ff.strip()

            try:
                int(ff)
            except:
                if ff not in exclude_params:
                    if ff not in counts:
                        counts[ff] = 0
                    counts[ff] += 1
    
    # Add in the include_params if they are missing from the sensitive parameter list
    for pp in include_params:
        if pp not in counts:
            counts[pp] = 0
    return counts


def adjust_param_ranges(paramfile, calib_params, default_ranges, outfilename, make_dups=False):
    """Adjust and write out the calibration parameters and ranges"""
    src_params = prms.parameters(paramfile)

    # Write the param_list file
    outfile = open(outfilename, 'w')
    for kk, vv in calib_params.iteritems():
        # Grab the current param (kk) from the .params file and verify the
        # upper and lower bounds. Modify them if necessary.

        src_vals = src_params.get_var(kk)['values']
        src_mean = np.mean(src_vals)
        src_min = np.min(src_vals)
        src_max = np.max(src_vals)
        
        # Set upper and lower bounds
        user_min = default_ranges[kk]['min']
        user_max = default_ranges[kk]['max']
        if user_min > src_min:
            user_min = src_min
        if user_max < src_max:
            user_max = src_max
        
        C = abs(user_min) + 10.
        
        adjMin = ((user_min + C) * (src_mean + C) / (src_min + C)) - C
        adjMax = ((user_max + C) * (src_mean + C) / (src_max + C)) - C
                
        if round(adjMin, 5) != round(default_ranges[kk]['min'], 5):
            print '\t%s: lower bound adjusted (%f to %f)' % (kk, default_ranges[kk]['min'], adjMin)
        if round(adjMax, 5) != round(default_ranges[kk]['max'], 5):
            print '\t%s: upper bound adjusted (%f to %f)' % (kk, default_ranges[kk]['max'], adjMax)
        
        if make_dups:
            # Duplicate each parameter by the number of times it occurred
            # This is for a special use case when calibrating individual values of
            # a parameter.
            for dd in xrange(vv):
                outfile.write('%s %f %f\n' % (kk, adjMax, adjMin))
        else:
            # Output each parameter once
            outfile.write('%s %f %f\n' % (kk, adjMax, adjMin))
    outfile.close()
