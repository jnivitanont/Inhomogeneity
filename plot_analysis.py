#!/usr/bin/env python3

from plot_analysis_module import *
import os

parser = argparse.ArgumentParser(description='Comparing Surface Pressure/XCO2/XCO/XCH4 of a merge file.')
parser.add_argument('-i', '--shin', required=True, type=str, help='Merged Scene/L1b/L2 file with slit homogenizer installed.')
parser.add_argument('-o', '--shout', required=True, type=str, help='Merged Scene/L1b/L2 file with slit homogenizer installed.')
parser.add_argument('-pd', '--plotdir', default='./', type=str, help='Plotting directory')
parser.add_argument('-pf', '--plotfmt', default='png', type=str, help='Image file format for plots')
parser.add_argument('-il', '--shinlbl', default=r'ILS-with-SH', type=str, help='Image label for ILS w/ SH merge file')
parser.add_argument('-ol', '--shoutlbl', default=r'ILS-without-SH', type=str, help='Image label for ILS w/o SH merge file')
parser.add_argument('-l', '--location', required=True, type=str, help='Target location; "lamont" or "manaus"')
args = parser.parse_args()
print(args)
print(os.getcwd())
GRIDS={"lamont":(35,59), "manaus":(36,80)}
TCCON={'lamont':(-97.486,36.604), 'manaus':(-60.5983, -3.2133)}

if __name__=='__main__':

    #PASS ARGS
    file_id = args.shin
    shout_id = args.shout
    plot_dir_root = args.plotdir
    plotfmt = args.plotfmt
    shoutlbl = args.shoutlbl
    shinlbl = args.shinlbl
    location = args.location
    GRID=GRIDS[location]
    SITE=TCCON[location]
    print('File1:', file_id)
    print('File2:', shout_id)
    print('ILS without SH label:', shoutlbl)
    print('ILS with SH label:', shinlbl)
    mpl.rcParams['savefig.format'] = args.plotfmt
    #OPEN FILES
    shin = h5.File(file_id, 'r')
    shout = h5.File(shout_id, 'r')
    lat = shin['analysis/lat'][:]
    lon = shin['analysis/lon'][:]
    lon_grid = np.reshape(lon, GRID)
    lat_grid = np.reshape(lat, GRID)
    # CALL PLOTTING FUNCS
    # print('Plotting four panel.')
    # plot_dir = plot_dir_root + '/four_panel/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_four_panel(shin, shinlbl)
    # plt.savefig(plot_dir + shinlbl + '-four_panel')
    # plt.close()
    # plot_four_panel(shout, shoutlbl)
    # plt.savefig(plot_dir + shoutlbl + '-four_panel')
    # plt.close()
    # plot_dir = plot_dir_root + '/four_panel_spatial/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_four_panel_spatial(shout, lon_grid, lat_grid, SITE,  title=location + ', ' + shoutlbl, apply_ppf=True)
    # plt.savefig(plot_dir + shoutlbl + '-four_panel_spatial_ppf')
    # plt.close()
    # plot_four_panel_spatial(shin, lon_grid, lat_grid, SITE,  title=location + ', ' + shinlbl, apply_ppf=True)
    # plt.savefig(plot_dir + shinlbl + '-four_panel_spatial_ppf')
    # plt.close()

    # plot_four_panel_spatial(shout, lon_grid, lat_grid, SITE,  title=location + ', ' + shoutlbl)
    # plt.savefig(plot_dir + shoutlbl + '-four_panel_spatial')
    # plt.close()
    # plot_four_panel_spatial(shin, lon_grid, lat_grid, SITE,  title=location + ', ' + shinlbl)
    # plt.savefig(plot_dir + shinlbl + '-four_panel_spatial')
    # plt.close()
    plot_dir = plot_dir_root + '/four_panel_kde/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_four_panel_kde(shin, shout, centered=True, sdfilter=4,title='')
    plt.savefig(plot_dir +'all-four_panel_kde')
    # print('Plotting Coeff. of Var. vs Covariates.')
    # plot_dir = plot_dir_root + '/cv/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # yvars = ['xco2_error','xco_error','xch4_error']
    # xvars = ['subslit_albedo_cv_o2','subslit_albedo_cv_weak_co2','subslit_albedo_cv_strong_co2'] 
    # plot_scatter(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir)
    # plot_medians(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir)
    # plot_scatter(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir, apply_ppf=True)
    # plot_medians(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir, apply_ppf=True)

    # print('Plotting Shape Error vs Covariates.')
    # plot_dir = plot_dir_root + '/shape_error/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # yvars = ['xco2_error','xco_error','xch4_error']
    # xvars = ['ils_shape_error_o2','ils_shape_error_weak_co2','ils_shape_error_strong_co2', 'ils_shape_error_ch4']
    # plot_scatter(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir)
    # plot_medians(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir)

    # print('Plotting Coeff. of Var. vs Shape Error.')
    # plot_dir = plot_dir_root + '/cv_vs_shape_error/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # yvars = ['ils_shape_error_o2','ils_shape_error_weak_co2','ils_shape_error_strong_co2', 'ils_shape_error_ch4']
    # xvars = ['subslit_albedo_cv_o2','subslit_albedo_cv_weak_co2','subslit_albedo_cv_strong_co2'] 
    # plot_scatter(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir)
    # plot_medians(shout, shin, xvars, yvars, shoutlbl, shinlbl, plot_dir)

    # print('Plotting density plots of covariates.')
    # plot_dir = plot_dir_root + '/kde_plots/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_kde(shout, shin, shoutlbl, shinlbl, plot_dir, centered=False)

    # print('Plotting spatial distributions of covariates.')
    # plot_dir = plot_dir_root + '/spatial/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_spatial(shout, shin, lon_grid, lat_grid, SITE, shoutlbl, shinlbl, plot_dir)

    # print('Plotting spatial anomaly distributions of covariates.')
    # plot_dir = plot_dir_root + '/spatial-anomaly/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_spatial_anomaly(shout, shin, lon_grid, lat_grid, SITE , shoutlbl, shinlbl, plot_dir)

    # print('Plotting correlation plots.')
    # plot_dir = plot_dir_root + '/correlation_gas_cv/ils_with_sh/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_error_correlation(shin, ils_cv_keys+gas_keys, plot_dir)
    # plot_dir = plot_dir_root + '/correlation_gas_cv/ils_without_sh/'
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_error_correlation(shout, ils_cv_keys+gas_keys, plot_dir)
    shout.close()
    shin.close()
