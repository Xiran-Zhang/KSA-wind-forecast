TIME_START_EXE = perf_counter()
# Obtain the probabilistic forecasts

quantile = np.arange(0.025,1,0.025)
if rerun:
    from estimate_distribution import estimate_distribution
    forErrorQuantile, forMean2015 = estimate_distribution(data,index,hyperpara)
    # PYTORCH 5.1
    torch.save(forErrorQuantile, "results/quantiles_knots.pt")
    # np.savez(file = 'results/quantiles_knots.npz', forErrorQuantile = forErrorQuantile.to("cpu").numpy())
    # Original
    # np.savez(file = 'results/quantiles_knots.npz', forErrorQuantile = forErrorQuantile)
else:
    # file = np.load('results/quantiles_knots.npz')
    # PYTORCH 5.4
    forErrorQuantile = torch.load("results/quantiles_knots.pt").to(device)
    # forErrorQuantile = torch.from_numpy(file['forErrorQuantile']).to(device)
    # Original
    # forErrorQuantile = file['forErrorQuantile']

# Show prediction interval coverage for 2016 at 3173 knots
lower_ind = np.empty(3).astype('int')
upper_ind = np.empty(3).astype('int')
# 95%
lower_ind[0] = np.where( np.abs(quantile - 0.025) < 1e-12 ) [0][0]
upper_ind[0] = np.where( np.abs(quantile - 0.975) < 1e-12 )[0][0]
# 80%
lower_ind[1] = np.where( np.abs(quantile - 0.1) < 1e-12 ) [0][0]
upper_ind[1] = np.where( np.abs(quantile - 0.9) < 1e-12 )[0][0]
# 60%
lower_ind[2] = np.where( np.abs(quantile - 0.2) < 1e-12 ) [0][0]
upper_ind[2] = np.where( np.abs(quantile - 0.8) < 1e-12 )[0][0]
# PYTORCH 5.3
PI = torch.empty((3,3,3173)).to(device)
# Original
# PI = np.empty((3,3,3173))
for i in range(3):
    # PYTORCH 5.2
    LB = esn_model.forMean + forErrorQuantile[lower_ind[i]]
    UB = esn_model.forMean + forErrorQuantile[upper_ind[i]]
    LB = LB.to(device)
    UB = UB.to(device)
    # Original
    # LB = esn_model.forMean + forErrorQuantile[lower_ind[i]]
    # UB = esn_model.forMean + forErrorQuantile[upper_ind[i]]

    for j in range(3):
        # PYTORCH 5.3
        PI[i,j] = (torch.logical_and(
                                esn_model.data.ts[esn_model.outSampleEmb_index] < UB[:,:,j],
                                esn_model.data.ts[esn_model.outSampleEmb_index] > LB[:,:,j])
                  ).sum(axis = 0) / (~torch.isnan(UB[:,:,j])).sum(axis = 0)
        # Original
        '''
        PI[i,j] = (np.logical_and(
                                esn_model.data.ts[esn_model.outSampleEmb_index] < UB[:,:,j],
                                esn_model.data.ts[esn_model.outSampleEmb_index] > LB[:,:,j])
                  ).sum(axis = 0) / (~np.isnan(UB[:,:,j])).sum(axis = 0)
        '''
# PYTORCH 5.5
PI_mean = torch.mean(PI, axis = 2)
PI_std = torch.std(PI, axis = 2)
# Original
# PI_mean = np.mean(PI, axis = 2)
# PI_std = np.std(PI, axis = 2)


print("Table 2 for the mean prediction interval coverage of y_t^* at all 3,173 knots \
and time points in 2016 by the ESN (standard deviation across knots is shown in parentheses). \
Note that the ESN is a stochastic approach, \
so the values may be slightly different than the values reported in the paper.\n")

print("--------------------------------------------------------------")
print("| Prediction |         Prediction Interval Coverage          |")
print("|  Interval  |    1h ahead    |    2h ahead   |   3h ahead   |")
print("--------------------------------------------------------------")
print("|     95%    |   {:.1%}({:.1%})  |  {:.1%}({:.1%})  |  {:.1%}({:.1%}) |".
      format(PI_mean[0,0], PI_std[0,0], PI_mean[0,1], PI_std[0,1], PI_mean[0,2], PI_std[0,2]))
print("--------------------------------------------------------------")
print("|     80%    |   {:.1%}({:.1%})  |  {:.1%}({:.1%})  |  {:.1%}({:.1%}) |".
      format(PI_mean[1,0], PI_std[1,0], PI_mean[1,1], PI_std[1,1], PI_mean[1,2], PI_std[1,2]))
print("--------------------------------------------------------------")
print("|     60%    |   {:.1%}({:.1%})  |  {:.1%}({:.1%})  |  {:.1%}({:.1%}) |".
      format(PI_mean[2,0], PI_std[2,0], PI_mean[2,1], PI_std[2,1], PI_mean[2,2], PI_std[2,2]))
print("--------------------------------------------------------------")

del forErrorQuantile, PI, LB, UB, PI_mean, PI_std
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 5.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
