TIME_START_EXE = perf_counter()

# Calibrate prediction distribution via the quantiles between the truth and the ensemble mean in 2015 at all locations

if rerun_calibration:
    try:
        krig_weight_all
    except NameError:
        os.system('Rscript spatial/get_all_krig_weight.R')
        krig_weight_all = pyreadr.read_r('spatial/all_krig_weight.RDS')
        krig_weight_all = np.array(krig_weight_all[None])
        krig_weight_all = krig_weight_all.reshape(53333, -1, order = 'F')

    try:
        forMean2015
    except NameError:
        _, forMean2015 = estimate_distribution(data,index,hyperpara)

    forMean2015_all = krig_weight_all.dot(forMean2015)

    forError2015_all = np.zeros_like(forMean2015_all)
    for ahead in range(3):
        forError2015_all[:,:,ahead] = forMean2015_all[:,:,ahead] - \
            wind_residual_all_locations[:,(data.nTime - 365*24*2):(data.nTime - 365*24)]

    forErrorQuantile_all = np.ndarray((quantile.size, forError2015_all.shape[0], 3))
    for i in range(quantile.size):
        forErrorQuantile_all[i] = np.nanquantile(forError2015_all, quantile[i], axis = 1)
    np.savez(file = 'results/quantiles_all.npz', forErrorQuantile_all = forErrorQuantile_all)
    del krig_weight_all, forMean2015, forMean2015_all
else:
    df = np.load('results/quantiles_all.npz')
    forErrorQuantile_all = df['forErrorQuantile_all']
    df.close()

torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 6.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
