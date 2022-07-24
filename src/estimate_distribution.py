from model import ESN
from datetime import datetime
import statsmodels.api as sm
import numpy as np

# PYTORCH 0: Prepare GPU computing
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device is: ")
print(device)
print("\n")

def estimate_distribution(data,index,hyperpara):
    index.test_start = data.nTime - 365*24*2 # 2015-1-1 00:00
    index.test_end = data.nTime - 365*24 - 1 # 2015-12-31 23:00 (inclusive)
    
    # PYTORCH 1
    esn_model = ESN(data, index, device)
    # esn_model.ensembleLen = 1
    # Original
    # esn_model = ESN(data,index)

    esn_model.train(hyperpara.parameter)
    print('ESN model trained with parameters: ', hyperpara.parameter)
    
    t0 = datetime.now()
    esn_model.forecast()
    t1 = datetime.now()

    print('Elapased time: ', t1-t0)
    
    esn_model.compute_forecast_error()
    
    # PYTORCH 2
    quantile = torch.arange(0.025,1,0.025).to(device)
    # Original 
    # quantile = np.arange(0.025,1,0.025)
    
    # PYTORCH 3
    forErrorQuantile = torch.zeros((quantile.size(dim=-1), *esn_model.forError.shape[1:]), device = device)
    # Original
    # forErrorQuantile = np.ndarray((quantile.size, *esn_model.forError.shape[1:]))
    
    # PYTORCH 4
    for i in range(quantile.size(dim=-1)):
    # Original
    # for i in range(quantile.size):
        # PYTORCH 5
        forErrorQuantile[i] = torch.nanquantile(esn_model.forError, quantile[i], axis = 0)
        # Original
        # forErrorQuantile[i] = np.nanquantile(esn_model.forError, quantile[i], axis = 0)
    
    return forErrorQuantile, esn_model.forMean
