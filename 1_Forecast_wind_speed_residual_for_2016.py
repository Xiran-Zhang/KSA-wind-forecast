TIME_START_EXE = perf_counter()

# Forecast wind speed residual for 2016

index.test_start = nTime - 365*24 # 2016-1-1 00:00
index.test_end = nTime - 1 # 2016-12-31 23:00 (inclusive)
esn_model = ESN(data,index,device)
# esn_model.ensembleLen = 1


if rerun:
    esn_model.train(hyperpara.parameter)
    print('ESN model trained with parameters: ', hyperpara.parameter)


    t0 = datetime.now()
    esn_model.forecast()
    t1 = datetime.now()

    print('Elapased time: ', t1-t0)

    esn_model.compute_forecast_mean()
    
    # PYTORCH 1.1
    torch.save(esn_model.forMean, "results/2016_GPU_mean-Ahead.pt")
    # Original
    # for timeAhead in range(esn_model.numTimePred):
        # PYTORCH 00
    #    df = pd.DataFrame(esn_model.forMean[:,:,timeAhead].cpu().numpy())
        # Original
        # df = pd.DataFrame(esn_model.forMean[:,:,timeAhead].to("cpu"))
    #    pyreadr.write_rds("results/2016_GPU_mean-{}hourAhead.RDS".format(timeAhead+1),df)

else:
    # PYTORCH 0
    esn_model.outSampleEmb_index = torch.arange(esn_model.index.test_start, esn_model.index.test_end+1)
    # PYTORCH 1.2
    esn_model.forMean = torch.load("results/2016_GPU_mean-Ahead.pt").to(device)
    # Original
    # esn_model.forMean = np.ndarray((index.test_end - index.test_start + 1, END_knots - START, 3))
    # esn_model.forMean[:,:,0] = np.array(pyreadr.read_r('results/2016_GPU_mean-1hourAhead.RDS')[None])
    # esn_model.forMean[:,:,1] = np.array(pyreadr.read_r('results/2016_GPU_mean-2hourAhead.RDS')[None])
    #esn_model.forMean[:,:,2] = np.array(pyreadr.read_r('results/2016_GPU_mean-3hourAhead.RDS')[None])
    # esn_model.forMean = torch.from_numpy(esn_model.forMean).to(device)

    #esn_model.forMean = torch.tensor(np.ndarray((index.test_end - index.test_start + 1, END_knots - START, 3)), device = device)
    #esn_model.forMean[:,:,0] = torch.from_numpy(np.array(pyreadr.read_r('results/2016_GPU_mean-1hourAhead.RDS')[None]).astype(np.double)).to(device)
    #esn_model.forMean[:,:,1] = torch.from_numpy(np.array(pyreadr.read_r('results/2016_GPU_mean-2hourAhead.RDS')[None]).astype(np.double)).to(device)
    #esn_model.forMean[:,:,2] = torch.from_numpy(np.array(pyreadr.read_r('results/2016_GPU_mean-3hourAhead.RDS')[None]).astype(np.double)).to(device)
    # Original
    # esn_model.outSampleEmb_index = np.arange(esn_model.index.test_start, esn_model.index.test_end+1)
    # esn_model.forMean = np.ndarray((index.test_end - index.test_start + 1, END - START, 3))
    # esn_model.forMean[:,:,0] = np.array(pyreadr.read_r('results/2016_GPU_mean-1hourAhead.RDS')[None])
    # esn_model.forMean[:,:,1] = np.array(pyreadr.read_r('results/2016_GPU_mean-2hourAhead.RDS')[None])
    # esn_model.forMean[:,:,2] = np.array(pyreadr.read_r('results/2016_GPU_mean-3hourAhead.RDS')[None])
    esn_model.forMeanComputed = True


# Calculate the MSE on knots for residuals
esn_mse = esn_model.compute_MSPE()

# PYTORCH 1
per_err = torch.zeros_like(esn_model.forMean)
# Original
# per_err = np.zeros_like(esn_model.forMean)

for i in range(esn_model.numTimePred):
    # PYTORCH 2
    per_err[:,:,i] = data.ts[esn_model.outSampleEmb_index] - data.ts[esn_model.outSampleEmb_index - i - 1]
    # Original
    # per_err[:,:,i] = data.ts[esn_model.outSampleEmb_index] - data.ts[esn_model.outSampleEmb_index - i - 1]
# PYTORCH 3
per_err[torch.isnan(esn_model.forMean)] = torch.nan
# Original
#per_err[np.isnan(esn_model.forMean)] = np.nan
# PYTORCH 4
per_mse = torch.nanmean(per_err**2,axis=(0,1))
# Original
# per_mse = np.nanmean(per_err**2,axis=(0,1))

print("Part of Table 3 for the MSE for y_t^* at all 3,173 knots \
and time points in 2016 by the ESN and persistence methods. \
Note that the ESN is a stochastic approach, \
so the MSE may be slightly different than the values reported in the paper.\n")

print("---------------------------------------------")
print("|     Forecast      |   ESN   | Persistence |")
print("---------------------------------------------")
print("|  One hour ahead   |  {:.3f}  |    {:.3f}    |".format(esn_mse[0],per_mse[0]))
print("|  Two hours ahead  |  {:.3f}  |    {:.3f}    |".format(esn_mse[1],per_mse[1]))
print("| Three hours ahead |  {:.3f}  |    {:.3f}    |".format(esn_mse[2],per_mse[2]))
print("---------------------------------------------")

del esn_mse, per_err
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 1.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
