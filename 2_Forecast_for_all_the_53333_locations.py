TIME_START_EXE = perf_counter()

# Forecast for all the 53333 locations
ncin = netCDF4.Dataset('./data/wind_residual_all_locations.nc', 'r', format='NETCDF4')
wind_residual_all_locations = ncin.variables['wind_residual_all_locations'][:]
lat_all_locations = ncin.variables['lat_all_locations'][:]
lon_all_locations = ncin.variables['lon_all_locations'][:]
ncin.close()

TIME_1 = perf_counter()
print("Un:", TIME_1 - TIME_START_EXE)
# Run the R script to get the spatial predictions.

if rerun:
    os.system("Rscript spatial/spatial_infer.R")

os.system("Rscript spatial/spatial_compile.R")

TIME_2 = perf_counter()
print("Deux:", TIME_2 - TIME_1)

if rerun:
    os.system('Rscript spatial/get_all_krig_weight.R')
    krig_weight_all = pyreadr.read_r('spatial/all_krig_weight.RDS')
    krig_weight_all = np.array(krig_weight_all[None])
    krig_weight_all = krig_weight_all.reshape(53333, -1, order = 'F')

    forMean2016_all = krig_weight_all.dot(esn_model.forMean)

    forError2016_all = np.zeros_like(forMean2016_all)
    for ahead in range(esn_model.numTimePred):
        forError2016_all[:,:,ahead] = forMean2016_all[:,:,ahead] - \
            wind_residual_all_locations[:, esn_model.outSampleEmb_index]

    esn_mse_all_locations = np.nanmean(forError2016_all ** 2, axis = 1)
    np.savez(file = 'spatial/esn_mse_all_locations.npz', esn_mse_all_locations = esn_mse_all_locations)

else:
    # PYTORCH 2.3
    esn_mse_all_locations = torch.load("spatial/esn_mse_all_locations.pt")
    # Original
    # df = np.load('spatial/esn_mse_all_locations.npz')
    # esn_mse_all_locations = df['esn_mse_all_locations']
    # df.close()

TIME_4 = perf_counter()
print("Quatre:", TIME_4 - TIME_2)

# Draw Figure 3, the nonstationary covariance model
import utils
utils.draw_nonstationary_model("spatial/NS_model.Rdata")
plt.savefig('draw_paper_figures/ns_model.png', dpi = 150)

TIME_5 = perf_counter()
print("Cinq:", TIME_5 - TIME_4)

# Get the persistence MSEi
# PYTORCH 2.1
per_mse_all_locations = torch.zeros_like(esn_mse_all_locations, device = device).T
# Original
# per_mse_all_locations = np.zeros_like(esn_mse_all_locations).T

# PYTORCH 2.4
wind_residual_all_locations = torch.from_numpy(wind_residual_all_locations).to(device)
# END PYTORCH 2.4
print('Spatial interpolating at time (total: 8760): ', end = '')
for time_id, time in enumerate(esn_model.outSampleEmb_index):

    if time_id % 100 == 0: print(time_id+1, end=' ')
    
    # PYTORCH 2.2
    per_for_all_location  =  torch.zeros_like(per_mse_all_locations, device = device)
    # Original
    # per_for_all_location  =  np.zeros_like(per_mse_all_locations)
    for pred_lag in range(esn_model.numTimePred):
        per_for_all_location[pred_lag] = wind_residual_all_locations[:,time-pred_lag-1].data

    if time_id < 2:
        update = time_id + 1
    else:
        update = 3

    tmp = (per_for_all_location - wind_residual_all_locations[:,time])**2
    per_mse_all_locations[:update] += tmp[:update]

# PYTORCH 5
nTest = esn_model.outSampleEmb_index.size(dim=-1)
# Original
# nTest = esn_model.outSampleEmb_index.size
# PYTORCH 2.5
per_mse_all_locations = (per_mse_all_locations.T / torch.as_tensor((nTest, nTest - 1, nTest - 2)).to(device))
# Original
# per_mse_all_locations = (per_mse_all_locations.T / (nTest, nTest - 1, nTest - 2))

esn_mse_all = esn_mse_all_locations.mean(axis = 0)
per_mse_all = per_mse_all_locations.mean(axis = 0)

TIME_6 = perf_counter()
print("\n\nSix:", TIME_6 - TIME_5, "\n")

print("Part of Table 4 for the MSE for y_t at all 53,333 locations \
and time points in 2016 by the S-ESN and persistence methods. \
Note that the S-ESN is a stochastic approach, \
so the MSE may be slightly different than the values reported in the paper.\n")

print("-----------------------------------------------")
print("|     Forecast      |   S-ESN   | Persistence |")
print("-----------------------------------------------")
print("|  One hour ahead   |   {:.3f}   |    {:.3f}    |".format(esn_mse_all[0], per_mse_all[0]))
print("|  Two hours ahead  |   {:.3f}   |    {:.3f}    |".format(esn_mse_all[1], per_mse_all[1]))
print("| Three hours ahead |   {:.3f}   |    {:.3f}    |".format(esn_mse_all[2], per_mse_all[2]))
print("-----------------------------------------------")

# Draw Figure 5, the outperformance of S-ESN to the persistence forecasts at all locations
# PYTORCH 2.6
esn_mse_all_locations = esn_mse_all_locations.cpu().numpy()
per_mse_all_locations = per_mse_all_locations.cpu().numpy()
# Original
utils.draw_outperformance_mse_all(lon_all_locations, lat_all_locations, esn_mse_all_locations, per_mse_all_locations)
plt.savefig('draw_paper_figures/MSE_difference.png', dpi = 150)

wind_residual_all_locations = wind_residual_all_locations.cpu().numpy()
del lat_all_locations, lon_all_locations, esn_mse_all_locations, per_mse_all_locations, nTest, esn_mse_all, per_mse_all
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 2.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
