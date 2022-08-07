TIME_START_EXE = perf_counter()

# Prediction Quantiles for the forecasted generated power at the 75 wind farms
# rerun = False
if rerun:
    # PYTORCH 8.1
    esn_turbine_location = torch.mm(krig_weight, esn_model.forMean[:,:,1].T.double()).T
    # Original
    # esn_turbine_location = krig_weight.dot(esn_model.forMean[:,:,1].T).T
    # PYTORCH 8.2
    esn_turbine_location_quantile = torch.empty((quantile.size, *esn_turbine_location.shape)).to(device)
    # Original
    # esn_turbine_location_quantile = np.empty((quantile.size, *esn_turbine_location.shape))
    
    # PYTORCH 8.3
    forErrorQuantile_all = forErrorQuantile_all.to(device)
    # END PYTORCH 8.3
    for i in range(quantile.size):
        esn_turbine_location_quantile[i] = esn_turbine_location + forErrorQuantile_all[i,index_turbine_location,1]

    print("Here1")
    esn_wind_speed_turbine_location_quantile = harmonic_mean_turbine_locations + \
        esn_turbine_location_quantile * gamma_all_locations[index_turbine_location]
    esn_wind_speed_turbine_location_quantile = esn_wind_speed_turbine_location_quantile ** 2
    print("Here1.5")    
    # PYTORCH 8.4
    esn_power_turbine_height_quantile = torch.zeros_like(esn_wind_speed_turbine_location_quantile)
    # Original
    # esn_power_turbine_height_quantile   = np.zeros_like(esn_wind_speed_turbine_location_quantile)
    print("Here1.6, quantile.size =", quantile.size)
    # Slow 1
    esn_wind_speed_turbine_location_quantile = esn_wind_speed_turbine_location_quantile.cpu().numpy()
    esn_power_turbine_height_quantile = esn_power_turbine_height_quantile.cpu().numpy()
    # END Slow 1
    for i in range(quantile.size):
        print(i)
        for loc in range(75):
            for time in np.arange(1,8760):
                tmp = esn_wind_speed_turbine_location_quantile[i,time,loc] * \
                    (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
                esn_power_turbine_height_quantile[i,time,loc] = get_power_curve(turbine_type[loc],tmp)
    # Slow 2
    esn_power_turbine_height_quantile = torch.from_numpy(esn_power_turbine_height_quantile).to(device)
    # END Slow 2
    print("Here2")
    # PYTORCH 8.5
    true_power_turbine_height = torch.from_numpy(true_power_turbine_height).to(device)
    esn_total_power_turbine_height_quantile_diff = torch.nansum(torch.abs(esn_power_turbine_height_quantile - true_power_turbine_height.T), axis = (1,2)).cpu()
    print("Here3")
    # Original
    # esn_total_power_turbine_height_quantile_diff = np.nansum(np.abs(esn_power_turbine_height_quantile - true_power_turbine_height.T), axis = (1,2))
    # PYTOCH 8.6
    torch.save(esn_power_turbine_height_quantile, "results/power_quantiles.pt")
    # Original
    # np.savez(file = 'results/power_quantiles.npz', esn_power_turbine_height_quantile = esn_power_turbine_height_quantile)
    del esn_turbine_location, esn_turbine_location_quantile, esn_wind_speed_turbine_location_quantile
else:
    # PYTORCH 8.7
    esn_power_turbine_height_quantile = torch.load('results/power_quantiles.pt').cpu()
    # Original
    # df = np.load('results/power_quantiles.npz')
    # esn_power_turbine_height_quantile = df['esn_power_turbine_height_quantile']
    # df.close()
    # PYTORCH 8.8
    esn_total_power_turbine_height_quantile_diff = torch.nansum(torch.abs(esn_power_turbine_height_quantile - torch.from_numpy(true_power_turbine_height).T), axis = (1,2))
    # Original
    # esn_total_power_turbine_height_quantile_diff = np.nansum(np.abs(esn_power_turbine_height_quantile - true_power_turbine_height.T), axis = (1,2))

print("Here4")
# Draw Figure 5, the annual sum of the absolute differences between the S-ESN forecast quantiles and the truth in wind energy during 2016 at all the 75 wind turbines combined
utils.draw_power_quantile(quantile, esn_total_power_turbine_height_quantile_diff)
plt.savefig('draw_paper_figures/power_quantile.pdf')

del esn_power_turbine_height_quantile, esn_total_power_turbine_height_quantile_diff
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 8.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
