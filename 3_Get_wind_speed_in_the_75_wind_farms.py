TIME_START_EXE = perf_counter()

# Get wind speed in the 75 wind farms
ncin = netCDF4.Dataset('./data/wind_residual_all_locations.nc', 'r', format='NETCDF4')
gamma_all_locations = ncin.variables['gamma'][:] # scaling parameter gamma(s) at all 53333 locations
harmonics_coefficients_all_locations = ncin.variables['harmonics_coefficients'][:]  # armonics_coefficients at all 53333 locations
harmonics = ncin.variables['harmonics'][:] # harmonics basis functions for each hour from 2013 to 2016
ncin.close()


ncin = netCDF4.Dataset('./data/wind_farm_data.nc', 'r', format='NETCDF4')
alpha_turbine_location = ncin.variables['alpha_turbine_location'][:] # alpha in the wind power law for 75 wind farms at each hour of the day
index_turbine_location = ncin.variables['index_turbine_location'][:]  # index of the 75 wind farms locations in the all 53333 locations
turbine_height = ncin.variables['turbine_height'][:] # turbine heights in the 75 wind farms
turbine_type = ncin.variables['turbine_type'][:] # turbine types in the 75 wind farms
ncin.close()

index_turbine_location = index_turbine_location.astype('int')
turbine_type = turbine_type.astype('int')

harmonic_mean_turbine_locations = \
   harmonics[(nTime - 365*24):].dot(harmonics_coefficients_all_locations[index_turbine_location,1:].T)
harmonic_mean_turbine_locations += harmonics_coefficients_all_locations[index_turbine_location,0]

true_wind_speed_turbine_location = harmonic_mean_turbine_locations + \
    wind_residual_all_locations[index_turbine_location,(nTime - 365*24):].T * gamma_all_locations[index_turbine_location]
true_wind_speed_turbine_location = true_wind_speed_turbine_location.T **2


if rerun:
    os.system('Rscript spatial/get_wind_farm_krig_weight.R')

krig_weight = pyreadr.read_r('spatial/wind_farm_krig_weight.RDS')
krig_weight =np.array(krig_weight[None])
# PYTORCH 7
krig_weight = torch.tensor(krig_weight.reshape(75, -1, order = 'F'), device = device)
# Original
# krig_weight = krig_weight.reshape(75, -1, order = 'F')
# PYTORCH 3.1
per_wind_speed_forecast_turbine_location = wind_residual_all_locations[np.ix_(index_turbine_location, esn_model.outSampleEmb_index - 2)]
# Original
# per_wind_speed_forecast_turbine_location   = wind_residual_all_locations[np.ix_(index_turbine_location, esn_model.outSampleEmb_index - 2)].data
# PYTORCH 6
esn_wind_speed_forecast_turbine_location = torch.mm(krig_weight, esn_model.forMean[:,:,1].T.double())
# Original
# esn_wind_speed_forecast_turbine_location   = krig_weight.dot(esn_model.forMean[:,:,1].T)
df = np.load('results/arima_wind_farm_predictions.npz')
arima_wind_speed_forecast_turbine_location = df['predictions']
df.close()
# PYTORCH 8
harmonic_mean_turbine_locations = torch.from_numpy(harmonic_mean_turbine_locations).to(device)
gamma_all_locations = torch.tensor(gamma_all_locations, device = device)
# END PYTORCH 8
esn_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations + \
    esn_wind_speed_forecast_turbine_location.T * gamma_all_locations[index_turbine_location]
esn_wind_speed_forecast_turbine_location = esn_wind_speed_forecast_turbine_location.T **2
# PYTORCH 9
per_wind_speed_forecast_turbine_location = torch.from_numpy(per_wind_speed_forecast_turbine_location).to(device)
# END PYTORCH 9
per_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations + \
    per_wind_speed_forecast_turbine_location.T * gamma_all_locations[index_turbine_location]
per_wind_speed_forecast_turbine_location = per_wind_speed_forecast_turbine_location.T **2
# PYTORCH 10
arima_wind_speed_forecast_turbine_location = torch.from_numpy(arima_wind_speed_forecast_turbine_location).to(device)
# END PYTORCH 10
arima_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations + \
    arima_wind_speed_forecast_turbine_location.T * gamma_all_locations[index_turbine_location]
arima_wind_speed_forecast_turbine_location = arima_wind_speed_forecast_turbine_location.T **2

del harmonics_coefficients_all_locations, harmonics
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 3.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
