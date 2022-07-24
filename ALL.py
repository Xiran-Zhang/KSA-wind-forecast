from time import perf_counter
ALL_START_EXE = perf_counter()
exec(open("0_Setup.py").read())
rerun = False
exec(open("1_Forecast_wind_speed_residual_for_2016.py").read())
rerun = False
exec(open("2_Forecast_for_all_the_53333_locations.py").read())
rerun = False
exec(open("3_Get_wind_speed_in_the_75_wind_farms.py").read())
exec(open("4_Transform_to_wind_power.py").read())
rerun = False
exec(open("5_Obtain_the_probabilistic_forecasts.py").read())
rerun_calibration = False
exec(open("6_Calibrate_prediction_distribution.py").read())
exec(open("7_Show_prediction_interval_coverage.py").read())
rerun = False
exec(open("8_Prediction_Quantiles.py").read())
ALL_END_EXE = perf_counter()
print("\n=======================================================\n")
print("In total, the running time is:", ALL_END_EXE - ALL_START_EXE)
print("\n=======================================================\n")
