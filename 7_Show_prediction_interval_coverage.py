TIME_START_EXE = perf_counter()

# Show prediction interval coverage for 2016 at all 53333 knots
# PYTORH 7.0
krig_weight_all = pyreadr.read_r('spatial/all_krig_weight.RDS')
krig_weight_all = np.array(krig_weight_all[None])
krig_weight_all = krig_weight_all.reshape(53333, -1, order = 'F')
krig_weight_all = torch.from_numpy(krig_weight_all).to(device)
# END PYTORCH 7.0

TIME_1 = perf_counter()
print("Un:", TIME_1 - TIME_START_EXE)

'''
try:
    forMean2016_all
except NameError:
    # try:
        krig_weight_all
    except NameError:
        os.system('Rscript spatial/get_all_krig_weight.R')
        krig_weight_all = pyreadr.read_r('spatial/all_krig_weight.RDS')
        # PYTORCH 7.1
        print("\n\n\nHere1\n\n\n")
        print(krig_weight_all[None])
        print(torch.tensor(krig_weight_all[None]))
        krig_weight_all = torch.tensor(krig_weight_all[None]).to(device)
        # Original
        # krig_weight_all = np.array(krig_weight_all[None])
        krig_weight_all = krig_weight_all.reshape(53333, -1, order = 'F')

    # PYTORCH 7.2
    print("here")
    #print(krig_weight_all)
    forMean2016_all = torch.mm(krig_weight_all, esn_model.forMean)
    # Original
    # forMean2016_all = krig_weight_all.dot(esn_model.forMean)
'''
forMean2016_all = torch.mm(krig_weight_all, esn_model.forMean.view(3173,-1).double()).view(53333, 8760, 3)
# PYTORCH 7.10
del krig_weight_all
torch.cuda.empty_cache()
# END PYTORCH 7.10
# PYTORCH 7.3
forecast = torch.swapaxes(forMean2016_all,0,1).cpu()
del forMean2016_all
torch.cuda.empty_cache()
# Original
# forecast = np.swapaxes(forMean2016_all,0,1)
TIME_2 = perf_counter()
print("Deux:", TIME_2 - TIME_1)
# PYTORCH 7.9
true = torch.from_numpy(wind_residual_all_locations[:,esn_model.outSampleEmb_index].T).cpu()
# Original
# true = wind_residual_all_locations[:,esn_model.outSampleEmb_index].T
TIME_3 = perf_counter()
print("Trois:", TIME_3 - TIME_2)
# PYTORCH 7.4
PI_all = torch.empty((3,3,53333))
# Original
# PI_all = np.empty((3,3,53333))
# PYTORCH 7.5
if not torch.is_tensor(forErrorQuantile_all):
    forErrorQuantile_all = torch.from_numpy(forErrorQuantile_all)

TIME_4 = perf_counter()
print("Quatre:", TIME_4 - TIME_3)
# END PYTORCH 7.5
for i in range(3):
    TIME_5 = perf_counter()
    # PYTORCH 7.8
    LB = forecast + forErrorQuantile_all[lower_ind[i]]
    UB = forecast + forErrorQuantile_all[upper_ind[i]]
    # LB = LB.to(device)i
    # UB = UB.to(device)
    # Original
    # LB = forecast + forErrorQuantile_all[lower_ind[i]]
    # UB = forecast + forErrorQuantile_all[upper_ind[i]]
    TIME_6 = perf_counter()
    print("Cinq:", TIME_6 - TIME_5)
    for j in range(3):
        # Slow
        # PYTORCH 7.6
        PI_all[i,j] = (torch.logical_and(true.data < UB[:,:,j],
                                         true.data > LB[:,:,j])
                      ).sum(axis = 0) / (~torch.isnan(UB[:,:,j])).sum(axis = 0)
        # Original
        # PI_all[i,j] = (np.logical_and(true < UB[:,:,j],
        #                               true > LB[:,:,j])
        #               ).sum(axis = 0) / (~np.isnan(UB[:,:,j])).sum(axis = 0)
        
    TIME_7 = perf_counter()
    print("Six:", TIME_7 - TIME_6)
# PYTORCH 7.7
PI_all_mean = torch.mean(PI_all, axis = 2)
PI_all_std = torch.std(PI_all, axis = 2)
# Original
# PI_all_mean = np.mean(PI_all, axis = 2)
# PI_all_std = np.std(PI_all, axis = 2)

print("Table S2 for the mean prediction interval coverage of y_t at all 53,333 locations \
and time points in 2016 using the ESN (standard deviation across knots is shown in parentheses). \
Note that the ESN is a stochastic approach, \
so the values may be slightly different than the values reported in the paper.\n")

print("--------------------------------------------------------------")
print("|     95%    |   {:.1%}({:.1%})  |  {:.1%}({:.1%})  |  {:.1%}({:.1%}) |".
      format(PI_all_mean[0,0], PI_all_std[0,0], PI_all_mean[0,1], PI_all_std[0,1], PI_all_mean[0,2], PI_all_std[0,2]))
print("--------------------------------------------------------------")
print("|     80%    |   {:.1%}({:.1%})  |  {:.1%}({:.1%})  |  {:.1%}({:.1%}) |".
      format(PI_all_mean[1,0], PI_all_std[1,0], PI_all_mean[1,1], PI_all_std[1,1], PI_all_mean[1,2], PI_all_std[1,2]))
print("--------------------------------------------------------------")
print("|     60%    |   {:.1%}({:.1%})  |  {:.1%}({:.1%})  |  {:.1%}({:.1%}) |".
      format(PI_all_mean[2,0], PI_all_std[2,0], PI_all_mean[2,1], PI_all_std[2,1], PI_all_mean[2,2], PI_all_std[2,2]))
print("--------------------------------------------------------------")

del forecast, true, PI_all, LB, UB, PI_all_mean, PI_all_std
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 7.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
