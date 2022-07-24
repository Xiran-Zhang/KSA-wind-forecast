from time import perf_counter
TIME_START_EXE = perf_counter()

import sys
sys.path.insert(1, 'src')

import os
import matplotlib.pyplot as plt
from datetime import datetime
import netCDF4
import numpy as np
import pyreadr
import pandas as pd

# Prepare GPU computing
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device is: ")
print(device)
print("\n")

# Read data
ncin = netCDF4.Dataset('./data/wind_residual.nc', 'r', format='NETCDF4')
START = 0
END_wind_residual = 11181920 # 10
END_knots = 3173
wind_residual = ncin.variables['wind_residual'][START:END_wind_residual]
# lon_knots = ncin.variables['lon'][START:END_knots]
# lat_knots = ncin.variables['lat'][START:END_knots]
ncin.close()

# These are masked arrays
# print(str(wind_residual))
# wind_residual = wind_residual.to()

# We now convert them into tensors and put them on GPU
wind_residual = torch.from_numpy(wind_residual).to(device)
# lon_knots = torch.from_numpy(lon_knots).to(device)
# lat_knots = torch.from_numpy(lat_knots).to(device)


from model import ESN
from data import Data
from index import Index
from hyperpara import Hyperpara

hyperpara = Hyperpara()

nTime = wind_residual.shape[1]
print('The total number of hours (2013-2016) considered:', nTime)

data = Data(nTime, wind_residual)
index = Index()

del wind_residual
# del lon_knots, lat_knots
torch.cuda.empty_cache()

TIME_END_EXE = perf_counter()
print("\nTime for running 0_setup.py is:", TIME_END_EXE - TIME_START_EXE, "\n")
