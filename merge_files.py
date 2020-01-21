#!/usr/bin/env python3
import h5py as h5
import numpy as np
import datetime as dt
import netCDF4 as nc
import argparse
import os

parser = argparse.ArgumentParser(description='Merge footprint scene/L1B/L2 retrieval files.')
parser.add_argument('-s', '--scene', metavar='scene file', required=True, help='Scene file')
parser.add_argument('-l', '--l1b', metavar='L1B file', required=True, help='L1B file')
parser.add_argument('-r', '--retr', metavar='L2 file', required=True, help='L2 Retrieval file')
parser.add_argument('-i', '--ils', metavar='ILS file', help='ILS file')
parser.add_argument('-o', '--out', metavar='output', required=True, help='Output file')
parser.add_argument('-m', '--mode', metavar='Slit homogenizer mode', type=str, required=True, help='ILS mode')
args = parser.parse_args()

def format_retrieved_var(retr_var, sounding_ids):
    if(len(retr_var.shape) == 2):
        formatted_var = np.full([scene_size, retr_var.shape[1]], np.nan)
    else:
        formatted_var = np.full(scene_size, np.nan)
    formatted_var[sounding_ids] = retr_var
    return formatted_var

dtor = np.pi/180.

sample_size = {'lamont':2065, 'manaus':2880}
scene_size  = sample_size['manaus']
print("SCENE_SIZE:", scene_size)
scene_fid   = args.scene
l1b_fid     = args.l1b
retr_fid    = args.retr
have_ils    = (args.ils is not None)
if have_ils:
    ils_fid = args.ils
    mode    = args.mode
out_fid     = args.out

if __name__ == '__main__':
    #read in Files
    print('Reading:    ' + scene_fid)
    scene   = h5.File(scene_fid, 'r')
    print('Reading:    ' + l1b_fid)
    l1b     = h5.File(l1b_fid, 'r')
    print('Reading:    ' + retr_fid)
    retr    = h5.File(retr_fid, 'r')
    if have_ils:
        print('Reading:    ' + ils_fid)
        ils     = nc.Dataset(ils_fid, 'r')
    else:
        print('No ILS file')
    #grab sounding ids for retrieval
    common_idx  = retr['RetrievalHeader/exposure_index'][:] -1
    l1b_id      = l1b['SoundingGeometry/sounding_id'][:,0]
    scene_size  = l1b_id.size
    retr_id     = retr['RetrievalHeader/sounding_id_reference'][:]
    retr_size   = retr_id.size
    #assert ( all(l1b_id[common_idx] - retr_id == 0)), 'Problem matching l1b to l2 file'
    #extract lat/lon coords
    lon = scene['Simulation/Geometry/longitude'][:,0,0]
    lat = scene['Simulation/Geometry/latitude'][:,0,0]
    #-------------------
    ## L2 FILE VARIABLES
    #-------------------
    # pressure
    retr_p0         = retr['RetrievalResults/surface_pressure_fph'][:]
    retr_dP         = retr_p0 - retr['RetrievalResults/surface_pressure_apriori_fph'][:]
    # retr_psurf_uncert = retr['RetrievalResults/surface_pressure_uncert_fph'][:]
    retr_pwf        = retr['RetrievalResults/xco2_pressure_weighting_function'][:]
    retr_pLevels    = (retr['RetrievalResults/vector_pressure_levels'][:].T/retr_p0).T
    retr_nLevels    = retr_pLevels.shape[1]
    # co2
    retr_xco2       = retr['RetrievalResults/xco2'][:]*1e6 #ppm
    # retr_xco2_prior = retr['RetrievalResults/xco2_apriori'][:]*1e6 #ppm
    # retr_xco2_uncert= retr['RetrievalResults/xco2_uncert'][:]*1e6 #ppm
    retr_co2_profile_apriori= retr['RetrievalResults/co2_profile_apriori'][:]*1e6
    # retr_co2_profile_akm    = retr['RetrievalResults/co2_profile_averaging_kernel_matrix'][:]
    retr_co2_ak_norm = retr['RetrievalResults/xco2_avg_kernel_norm'][:]
    retr_co2_grad_del       = retr['RetrievalResults/co2_vertical_gradient_delta'][:]*1e6
    # ch4
    retr_xch4       = retr['RetrievalResults/xch4'][:]*1e9 
    # retr_xch4_prior = retr['RetrievalResults/xch4_apriori'][:]*1e9 
    # retr_xch4_uncert= retr['RetrievalResults/xch4_uncert'][:]*1e9 #ppb
    retr_ch4_profile_apriori= retr['RetrievalResults/ch4_profile_apriori'][:]*1e9
    # retr_ch4_profile_akm    = retr['RetrievalResults/ch4_profile_averaging_kernel_matrix'][:]
    retr_ch4_ak_norm = retr['RetrievalResults/xch4_avg_kernel_norm'][:]
    # co
    retr_xco        = retr['RetrievalResults/xco'][:]*1e9 
    # retr_xco_prior  = retr['RetrievalResults/xco_apriori'][:]*1e9
    # retr_xco_uncert = retr['RetrievalResults/xco_uncert'][:]*1e9 #ppb
    retr_co_profile_apriori= retr['RetrievalResults/co_profile_apriori'][:]*1e9
    # retr_co_profile_akm    = retr['RetrievalResults/co_profile_averaging_kernel_matrix'][:]
    retr_co_ak_norm = retr['RetrievalResults/xco_avg_kernel_norm'][:]
    #h2o
    retr_tcwv = retr['RetrievalResults/retrieved_h2o_column'][:]*(0.0180153/6.0221415e23)
    #signal stuff
    # signal_keys =           ['signal_ch4_fph',
    #                          'signal_o2_fph',
    #                          'signal_strong_co2_fph',
    #                          'signal_weak_co2_fph']
    retr_s1 = retr['SpectralParameters/signal_o2_fph'][:]
    retr_s2 = retr['SpectralParameters/signal_weak_co2_fph'][:]
    retr_s3 = retr['SpectralParameters/signal_strong_co2_fph'][:]
    retr_s4 = retr['SpectralParameters/signal_ch4_fph'][:]
    retr_s31 = retr_s3/retr_s1
    retr_s32 = retr_s3/retr_s2
    retr_s34 = retr_s3/retr_s4

    ##format retrieved variables
    xco2        = format_retrieved_var(retr_xco2, common_idx)
    # xco2_prior  = format_retrieved_var(retr_xco2_prior, common_idx)
    xch4        = format_retrieved_var(retr_xch4, common_idx)
    # xch4_prior  = format_retrieved_var(retr_xch4_prior, common_idx)
    xco         = format_retrieved_var(retr_xco, common_idx)
    # xco_prior   = format_retrieved_var(retr_xco_prior, common_idx)
    dP          = format_retrieved_var(retr_dP, common_idx)
    # xco2_uncert = format_retrieved_var(retr_xco2_uncert,common_idx)
    # xco_uncert  = format_retrieved_var(retr_xco_uncert,common_idx)
    # xch4_uncert = format_retrieved_var(retr_xch4_uncert,common_idx)
    # psurf_uncert= format_retrieved_var(retr_psurf_uncert,common_idx)
    psurf       = format_retrieved_var(retr_p0,common_idx)
    co2_grad_del= format_retrieved_var(retr_co2_grad_del, common_idx)
    tcwv        = format_retrieved_var(retr_tcwv, common_idx)
    s31         = format_retrieved_var(retr_s31, common_idx)
    s32         = format_retrieved_var(retr_s32, common_idx)
    s34         = format_retrieved_var(retr_s34, common_idx)

    #----------------------
    ## SCENE FILE VARIABLES
    #----------------------
    #extract pressure levels
    nLayers     = scene['Simulation/Thermodynamic/num_layers'][0]
    pLevels     = scene['Simulation/Thermodynamic/pressure_level'][:,:nLayers+1]
    p0          = np.max(pLevels, axis=1)
    dpLevels    = np.apply_along_axis(func1d=np.ediff1d, arr=pLevels, axis=1)
    pLayers     = ((pLevels[:,1:] - dpLevels/2).T/p0).T
    #extract xco2, xch4, xco
    scene_dryair        = scene['Simulation/Gas/species_density'][:,1,:nLayers]
    scene_co2_profile   = scene['Simulation/Gas/species_density'][:,3,:nLayers] / scene_dryair * 1e6    #ppm
    scene_ch4_profile   = scene['Simulation/Gas/species_density'][:,6,:nLayers] / scene_dryair * 1e9    #ppb
    scene_co_profile    = scene['Simulation/Gas/species_density'][:,7,:nLayers] / scene_dryair * 1e9     #ppb
    scene_albedo_o2     = scene['Simulation/Surface/modis_albedo'][:,0]
    scene_albedo_wco2   = scene['Simulation/Surface/modis_albedo'][:,2]
    scene_albedo_sco2   = scene['Simulation/Surface/modis_albedo'][:,4]
    scene_pwf = (scene_dryair.T/np.sum(scene_dryair,1)).T
    airmass             = 1./np.cos(scene['Simulation/Geometry/sat_zenith_angle'][:,0,0]*dtor) +\
                            1./np.cos(scene['Simulation/Geometry/sun_zenith_angle'][:,0,0]*dtor)
    #--------------
    ## TRUTH VALUES
    #--------------
    # truth_co2_profile   = np.full([scene_size, retr_nLevels], np.nan)
    # truth_co_profile    = np.full([scene_size, retr_nLevels], np.nan)
    # truth_ch4_profile   = np.full([scene_size, retr_nLevels], np.nan)
    truth_xco2      = np.full([scene_size], np.nan)
    truth_xco       = np.full([scene_size], np.nan)
    truth_xch4      = np.full([scene_size], np.nan)
    
    co2_hires_ak_norm = np.empty((retr_size, nLayers))
    co_hires_ak_norm = np.empty((retr_size, nLayers))
    ch4_hires_ak_norm = np.empty((retr_size, nLayers))
    retr_co2_profile_apriori_hires          = np.empty((retr_size, pLayers.shape[1]))
    retr_co_profile_apriori_hires          = np.empty((retr_size, pLayers.shape[1]))
    retr_ch4_profile_apriori_hires          = np.empty((retr_size, pLayers.shape[1]))

    for i in range(retr_size):
        foot_idx = common_idx[i]
        ## DEPRECATED: Now interpolating retrieved profile (20 layers) to simulation resolution (25 layers)
        # truth_co2_profile[foot_idx]     = np.interp(retr_pLevels[i], pLayers[foot_idx], scene_co2_profile[foot_idx])
        # truth_ch4_profile[foot_idx]     = np.interp(retr_pLevels[i], pLayers[foot_idx], scene_ch4_profile[foot_idx])
        # truth_co_profile[foot_idx]      = np.interp(retr_pLevels[i], pLayers[foot_idx], scene_co_profile[foot_idx])
        # truth_co2_profile[foot_idx]     = np.matmul(retr_co2_profile_akm[i], truth_co2_profile[foot_idx]) + \
        #                                     np.matmul(np.eye(retr_nLevels) - retr_co2_profile_akm[i], retr_co2_profile_apriori[i])
        # truth_ch4_profile[foot_idx]     = np.matmul(retr_ch4_profile_akm[i], truth_ch4_profile[foot_idx]) + \
        #                                     np.matmul(np.eye(retr_nLevels) - retr_ch4_profile_akm[i], retr_ch4_profile_apriori[i])
        # truth_co_profile[foot_idx]      = np.matmul(retr_co_profile_akm[i], truth_co_profile[foot_idx]) + \
        #                                     np.matmul(np.eye(retr_nLevels) - retr_co_profile_akm[i], retr_co_profile_apriori[i])
        retr_co2_profile_apriori_hires[i,:] = np.interp(pLayers[foot_idx,:], retr_pLevels[i,:], retr_co2_profile_apriori[i,:])
        retr_ch4_profile_apriori_hires[i,:] = np.interp(pLayers[foot_idx,:], retr_pLevels[i,:], retr_ch4_profile_apriori[i,:])
        retr_co_profile_apriori_hires[i,:]  = np.interp(pLayers[foot_idx,:], retr_pLevels[i,:], retr_co_profile_apriori[i,:])
        co2_hires_ak_norm[i]    = np.interp(pLayers[foot_idx], retr_pLevels[i], retr_co2_ak_norm[i])
        ch4_hires_ak_norm[i]    = np.interp(pLayers[foot_idx], retr_pLevels[i], retr_ch4_ak_norm[i])
        co_hires_ak_norm[i]     = np.interp(pLayers[foot_idx], retr_pLevels[i], retr_co_ak_norm[i])

    truth_xco2[common_idx]  = np.sum(retr_co2_profile_apriori_hires * scene_pwf[common_idx], axis=1) + \
        np.sum(co2_hires_ak_norm * scene_pwf[common_idx] *(scene_co2_profile[common_idx] - retr_co2_profile_apriori_hires), axis=1)
    truth_xch4[common_idx]  = np.sum(retr_ch4_profile_apriori_hires * scene_pwf[common_idx], axis=1) + \
        np.sum(ch4_hires_ak_norm * scene_pwf[common_idx] *(scene_ch4_profile[common_idx] - retr_ch4_profile_apriori_hires), axis=1)
    truth_xco[common_idx]   = np.sum(retr_co_profile_apriori_hires * scene_pwf[common_idx], axis=1) + \
        np.sum(co_hires_ak_norm * scene_pwf[common_idx] *(scene_co_profile[common_idx] - retr_co_profile_apriori_hires), axis=1)
    
    #calculate errors
    xco2_error = xco2 - truth_xco2
    xch4_error = xch4 - truth_xch4
    xco_error  = xco - truth_xco
    merge_vars = {'dP' : dP,
            'airmass':airmass,
            'co2_grad_del':co2_grad_del,
            'xco2': xco2,
            'xco2_truth':truth_xco2,
            'xco2_error':xco2_error,
            'xch4': xch4,
            'xch4_truth':truth_xch4,
            'xch4_error':xch4_error,
            'xco': xco,
            'xco_truth':truth_xco,
            'xco_error':xco_error,
            'psurf' :  psurf,
            'psurf_truth' :  p0,
            'psurf_error' : p0 - psurf,
            's31' : s31,
            's32' : s32,
            's34' : s34,
            'tcwv' : tcwv}
    # profile_vars = {'truth_co_profile' : truth_co_profile,
    #         'truth_ch4_profile' : truth_ch4_profile,
    #         'truth_co2_profile' : truth_co2_profile,
    #         'pwf' : format_retrieved_var(retr_pwf,common_idx),
    #         'retrieved_pressure_levels' : format_retrieved_var(retr_pLevels,common_idx)}
    coords={'lat': lat,
            'lon': lon,
            'sounding_id': l1b_id}
    outfile = h5.File(out_fid, 'a')
    outfile['metadata/scene_file'] = scene_fid.encode('ASCII')
    outfile['metadata/l1b_file'] = l1b_fid.encode('ASCII')
    if have_ils:
        outfile['metadata/ils_file'] = ils_fid.encode('ASCII')
    outfile['metadata/retrieval_file'] = retr_fid.encode('ASCII')
    scene_grp = outfile.create_group('scene')
    h5.h5o.copy(scene.id, b'Simulation', scene_grp.id, b'Simulation')
    l1b_grp = outfile.create_group('l1b')
    for grp in l1b.keys():
        bgrp = grp.encode('ASCII')
        h5.h5o.copy(l1b.id, bgrp, l1b_grp.id, bgrp )
    retr_grp = outfile.create_group('retrieval')
    for grp in retr.keys():
        bgrp = grp.encode('ASCII')
        h5.h5o.copy(retr.id, bgrp, retr_grp.id, bgrp)
    anal_grp = outfile.create_group('analysis')
    retrieval_fit_keys =    ['diverging_steps',
                             'dof_ch4_profile',
                             'dof_co2_profile',
                             'dof_co_profile',
                             'dof_full_vector']
    for key in retrieval_fit_keys:
        anal_grp.create_dataset(key, data=format_retrieved_var(retr['RetrievalResults/'+key][:], common_idx))
    spectral_fit_keys =     ['reduced_chi_squared_ch4_fph',
                             'reduced_chi_squared_o2_fph',
                             'reduced_chi_squared_strong_co2_fph',
                             'reduced_chi_squared_weak_co2_fph',
                             'relative_residual_mean_square_ch4',
                             'relative_residual_mean_square_o2',
                             'relative_residual_mean_square_strong_co2',
                             'relative_residual_mean_square_weak_co2',
                             'residual_mean_square_ch4',
                             'residual_mean_square_o2',
                             'residual_mean_square_strong_co2',
                             'residual_mean_square_weak_co2']
    for key in spectral_fit_keys:
        anal_grp.create_dataset(key.replace('reduced_chi_squared','chi2').replace('_fph','').replace('relative','rel').replace('residual_mean_square','rms'),
            data=format_retrieved_var(retr['SpectralParameters/'+key][:], common_idx))
    for key in merge_vars.keys():
        anal_grp.create_dataset(key, data=merge_vars[key])
    for key in coords.keys():
        anal_grp.create_dataset(key, data=coords[key])
    albedo_keys =           ['brdf_reflectance_slope_ch4',
                             'brdf_reflectance_slope_o2',
                             'brdf_reflectance_slope_strong_co2',
                             'brdf_reflectance_slope_weak_co2',
                             'brdf_reflectance_ch4',
                             'brdf_reflectance_o2',
                             'brdf_reflectance_strong_co2',
                             'brdf_reflectance_weak_co2']
    for key in albedo_keys:
        anal_grp.create_dataset(key.replace('brdf_reflectance', 'albedo'), data=format_retrieved_var(retr['RetrievalResults/'+key][:], common_idx))
    if have_ils:
        A = np.array([np.ones(20), np.arange(0,20)]).T  
        anal_grp.create_dataset('subslit_albedo_slope_o2',          data = np.linalg.lstsq(A, ils['gc_subslit_alb_band2'][:scene_size].T/1000, rcond=None)[0][1])
        anal_grp.create_dataset('subslit_albedo_slope_weak_co2',    data = np.linalg.lstsq(A, ils['gc_subslit_alb_band6'][:scene_size].T/1000, rcond=None)[0][1])
        anal_grp.create_dataset('subslit_albedo_slope_strong_co2',  data = np.linalg.lstsq(A, ils['gc_subslit_alb_band7'][:scene_size].T/1000, rcond=None)[0][1])
        anal_grp.create_dataset('subslit_albedo_cv_o2',         data = ils['gc_footprint_alb_cv_band2'][:scene_size])
        anal_grp.create_dataset('subslit_albedo_cv_weak_co2',   data = ils['gc_footprint_alb_cv_band6'][:scene_size])
        anal_grp.create_dataset('subslit_albedo_cv_strong_co2', data = ils['gc_footprint_alb_cv_band7'][:scene_size])
        #calculate shape error
        unifils = ils[mode+'_unif'][:]
        unifmax = np.max(unifils,1)
        ils_se = np.full((4,scene_size), np.nan)
        for i in range(scene_size):
            ils_se[:,i] = (np.max(np.abs(ils[mode][:, i] - unifils),1).T/np.abs(unifmax)).T
        anal_grp.create_dataset('ils_shape_error_o2',           data = ils_se[0])
        anal_grp.create_dataset('ils_shape_error_weak_co2',     data = ils_se[1])
        anal_grp.create_dataset('ils_shape_error_strong_co2',   data = ils_se[2])
        anal_grp.create_dataset('ils_shape_error_ch4',          data = ils_se[3])
    # prof_grp = anal_grp.create_group('profile')
    # for key in profile_vars.keys():
    #     prof_grp.create_dataset(key, data=profile_vars[key])
    print('Saved to:   ' + out_fid)
    outfile.close()
    scene.close()
    l1b.close()
    retr.close()
    if have_ils:
        ils.close()
