TIME_START_EXE = perf_counter()

# Transform to wind power
# power_curve: power generated in the two wind turbine models
winds_peed = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,32.5,33,33.5,34,34.5,35,35.5,36,36.5,37])
power_curve = np.array([[0,0,0,0,0,0,33,106,197,311,447,610,804,1032,1298,1601,1936,2292,2635,2901,3091,3215,3281,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,29,68,114,177,243,347,452,595,738,907,1076,1307,1538,1786,2033,2219,2405,2535,2633,2710,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

def get_power_curve(t_m,x):
    if x > 37:
        return 0
    left_ind = int(x*2)
    power = (power_curve[t_m,left_ind+1]-power_curve[t_m,left_ind]) * (x-winds_peed[left_ind])/0.5 +\
        power_curve[t_m,left_ind]
    return(power)

true_power_turbine_height = np.empty(true_wind_speed_turbine_location.shape) * np.nan
A = perf_counter()
for loc in np.arange(75):
    for time in np.arange(1,8760):
        tmp =  true_wind_speed_turbine_location[loc,time] * (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
        true_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)
B = perf_counter()

esn_power_turbine_height   = np.empty((75,8760)) * np.nan
per_power_turbine_height   = np.empty((75,8760)) * np.nan
arima_power_turbine_height = np.empty((75,8760)) * np.nan

# PYTORCH 4.1
esn_wind_speed_forecast_turbine_location = esn_wind_speed_forecast_turbine_location.to("cpu").numpy()
per_wind_speed_forecast_turbine_location = per_wind_speed_forecast_turbine_location.to("cpu").numpy()
arima_wind_speed_forecast_turbine_location = arima_wind_speed_forecast_turbine_location.to("cpu").numpy()
# END PYTORCH 4.1
for loc in np.arange(75):
    for time in np.arange(1,8760):
        tmp = esn_wind_speed_forecast_turbine_location[loc,time] * \
            (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
        esn_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)

        tmp = per_wind_speed_forecast_turbine_location[loc,time] * \
            (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
        per_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)

        tmp = arima_wind_speed_forecast_turbine_location[loc,time] * \
            (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
        arima_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)

esn_wind_energy_diff   = np.nansum(np.abs(esn_power_turbine_height - true_power_turbine_height))
per_wind_energy_diff   = np.nansum(np.abs(per_power_turbine_height - true_power_turbine_height))
arima_wind_energy_diff = np.nansum(np.abs(arima_power_turbine_height - true_power_turbine_height))


print('The annual sum of the absolute differences in wind energy during 2016 at all the 75 wind turbines is\n\
      {:.2E}kW·h for S-ESN,\n\
      {:.2E}kW·h for ARIMA,\n\
  and {:.2E}kW·h for persistence.\n\
Thus, we obtain a {:.0f}% improvement against the ARIMA forecasts \
and an {:.0f}% improvement against the persistence forecasts.'.\
    format(esn_wind_energy_diff, arima_wind_energy_diff, per_wind_energy_diff,
           (arima_wind_energy_diff-esn_wind_energy_diff)/arima_wind_energy_diff*100,
           (per_wind_energy_diff-esn_wind_energy_diff)/per_wind_energy_diff*100))


del esn_power_turbine_height, per_power_turbine_height, arima_power_turbine_height, esn_wind_energy_diff, per_wind_energy_diff, arima_wind_energy_diff
del true_wind_speed_turbine_location, per_wind_speed_forecast_turbine_location, esn_wind_speed_forecast_turbine_location, arima_wind_speed_forecast_turbine_location
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("Time for running 4.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
