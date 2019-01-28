
from __future__ import (absolute_import, division, print_function)


# Order to write control file parameters for printing and writing a new control file
ctl_order = ['start_time', 'end_time', 'executable_desc', 'executable_model', 'model_mode', 'param_file', 'data_file',
             'model_output_file', 'parameter_check_flag', 'print_debug', 'et_module', 'precip_module',
             'soilzone_module',
             'solrad_module',
             'srunoff_module', 'strmflow_module', 'temp_module', 'transp_module', 'prms_warmup', 'init_vars_from_file',
             'save_vars_to_file', 'var_init_file', 'var_save_file', 'cbh_binary_flag', 'cbh_check_flag',
             'gwflow_cbh_flag', 'orad_flag', 'snow_cbh_flag', 'humidity_day', 'potet_day', 'precip_day', 'precip_grid',
             'swrad_day', 'tmax_day', 'tmax_grid', 'tmin_day', 'tmin_grid', 'transp_day', 'windspeed_day', 'csvON_OFF',
             'csv_output_file', 'nhruOutON_OFF', 'nhruOutBaseFileName', 'nhruOutVars', 'nhruOutVar_names',
             'nhruOut_freq', 'mapOutON_OFF', 'nmapOutVars', 'mapOutVar_names', 'statsON_OFF', 'stat_var_file',
             'nstatVars', 'statVar_element', 'statVar_names', 'aniOutON_OFF', 'ani_output_file', 'naniOutVars',
             'aniOutVar_names', 'dispGraphsBuffSize', 'ndispGraphs', 'dispVar_element', 'dispVar_names',
             'dispVar_plot', 'initial_deltat', 'cascade_flag', 'cascadegw_flag', 'dprst_flag', 'dyn_covden_flag',
             'dyn_covtype_flag', 'dyn_dprst_flag', 'dyn_fallfrost_flag', 'dyn_imperv_flag', 'dyn_intcp_flag',
             'dyn_potet_flag',
             'dyn_radtrncf_flag', 'dyn_snareathresh_flag', 'dyn_soil_flag', 'dyn_springfrost_flag',
             'dyn_sro2dprst_imperv_flag', 'dyn_sro2dprst_perv_flag', 'dyn_transp_flag', 'frozen_flag',
             'gwr_swale_flag', 'stream_temp_flag', 'subbasin_flag', 'covden_sum_dynamic', 'covden_win_dynamic',
             'covtype_dynamic', 'dprst_depth_dynamic', 'dprst_frac_dynamic', 'fallfrost_dynamic',
             'imperv_frac_dynamic', 'imperv_stor_dynamic', 'potetcoef_dynamic', 'radtrncf_dynamic',
             'snareathresh_dynamic', 'snow_intcp_dynamic', 'soilmoist_dynamic', 'soilrechr_dynamic',
             'springfrost_dynamic', 'srain_intcp_dynamic', 'sro2dprst_imperv_dynamic',
             'sro2dprst_perv_dynamic', 'transp_flag_dynamic', 'transpbeg_dynamic', 'transpend_dynamic',
             'wrain_intcp_dynamic', 'dprst_transferON_OFF', 'dprst_transfer_file', 'external_transferON_OFF',
             'external_transfer_file', 'gwr_transferON_OFF', 'gwr_transfer_file', 'lake_transferON_OFF',
             'lake_transfer_file', 'segment_transferON_OFF', 'segment_transfer_file', 'segmentOutON_OFF']

ctl_module_params = ['et_module', 'precip_module', 'soilzone_module', 'solrad_module',
                     'srunoff_module', 'strmflow_module', 'temp_module', 'transp_module']

# valtypes = ['', 'integer', 'float', 'double', 'string']

# Constants related to parameter files
DIMENSIONS_HDR = 'Dimensions'
PARAMETERS_HDR = 'Parameters'
CATEGORY_DELIM = '**'  # Delimiter for categories of variables
VAR_DELIM = '####'  # Used to delimit dimensions and parameters
# DATA_TYPES = ['', 'integer', 'float', 'double', 'string']
DATA_TYPES = {1: 'integer', 2: 'float', 3: 'double', 4: 'string'}

# Valid dimensions names for PRMS
DIMENSION_NAMES = ['ncascade', 'ncascdgw', 'nconsumed', 'ndays', 'ndepl', 'ndeplval',
                   'nevap', 'nexternal', 'ngw', 'ngwcell', 'nhru', 'nhrucell', 'nhumid',
                   'nlake', 'nlakeelev', 'nlapse', 'nmonths', 'nobs', 'npoigages', 'nrain',
                   'nratetbl', 'nsegment', 'nsnow', 'nsol', 'nssr', 'nsub', 'ntemp',
                   'numlakes', 'nwateruse', 'nwind', 'one']

HRU_DIMS = ['nhru', 'ngw', 'nssr']  # These dimensions are related and should have same size

# Constants for NhmParamDb
REGIONS = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08', 'r09',
           'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18']
PARAMETERS_XML = 'parameters.xml'
DIMENSIONS_XML = 'dimensions.xml'
NHM_DATATYPES = {'I': 1, 'F': 2, 'D': 3, 'S': 4}
NETCDF_DATATYPES = {1: 'i4', 2: 'f4', 3: 'f4', 4: 'c'}
NETCDF_FILLVAL = {1: 0, 2: 9.9692099683868690e+36, 3: 9.9692099683868690e+36, 4: 0}
PARNAME_DATATYPES = {'long': 1, 'float': 2, 'double': 3, 'string': 4}
