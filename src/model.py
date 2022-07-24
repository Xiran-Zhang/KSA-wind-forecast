from time import perf_counter

import torch
import numpy as np
from scipy.linalg import eigh

class ESN:
    
    def __init__(self,data,index,device):
        
        self.data = data
        self.index = index
        
        self.ensembleLen = 100
        self.numTimePred = 3
        
        self.tauEmb = 1 # number of lead time embedded
    
        self.forMeanComputed = False
        self.forErrorComputed = False

        # PYTORCH 4
        self.device = device

        # PYTORCH 42 
        self.forMat = 2
        
    def standardize_in_sample(self, is_validation = False):
        if(is_validation):
            self.inSampleEmb_len = self.index.validate_start - self.m *self.tauEmb
        else:
            self.inSampleEmb_len = self.index.test_start - self.m *self.tauEmb
            
        #### X
        # PYTORCH 1: convert it into tensor
        self.inSampleX = torch.from_numpy(np.repeat(np.nan,self.inSampleEmb_len * self.m * self.numLocs).reshape(self.inSampleEmb_len, self.m, -1)).to(self.device)
        # Original
        # self.inSampleX = np.repeat(np.nan,self.inSampleEmb_len * self.m * self.numLocs).reshape(self.inSampleEmb_len, self.m, -1)
        for i in range(self.inSampleEmb_len):
            self.inSampleX[i,] = self.data.ts[range(i,(self.m * self.tauEmb + i), self.tauEmb)]

        self.inSampleX_mean = self.inSampleX.mean(axis=0)
        self.inSampleX_std = self.inSampleX.std(axis=0)
            
        self.inSampleX = (self.inSampleX -self.inSampleX_mean)/self.inSampleX_std
        # PYTORCH 5
        self.inSampleDesignMatrix = torch.column_stack([torch.ones(self.inSampleEmb_len, dtype = torch.int8, device = self.device), self.inSampleX.reshape(self.inSampleEmb_len,-1)])
        # Original
        # self.inSampleDesignMatrix = np.column_stack([np.repeat(1,self.inSampleEmb_len),self.inSampleX.reshape(self.inSampleEmb_len,-1)])
        
        #### Y
        self.inSampleY = self.data.ts[range(self.m * self.tauEmb,self.inSampleEmb_len + (self.m * self.tauEmb))]
        
        self.inSampleY_mean = self.inSampleY.mean(axis=0)
        self.inSampleY_std=self.inSampleY.std(axis=0)

        self.inSampleY = (self.inSampleY-self.inSampleY_mean)/self.inSampleY_std
        
        
    def standardize_out_sample(self, is_validation = False):
        if(is_validation):
            self.outSampleEmb_index = np.arange(self.index.validate_start, self.index.validate_end+1)
        else:
            self.outSampleEmb_index = np.arange(self.index.test_start, self.index.test_end+1)
            
        self.outSampleEmb_len = len(self.outSampleEmb_index)
        
        #### X
        # PYTORCH 2: convert it into tensor
        self.outSampleX = torch.zeros((self.outSampleEmb_len, self.m, self.numLocs), device = self.device) * torch.nan
        # Original
        # self.outSampleX = np.zeros((self.outSampleEmb_len, self.m, self.numLocs)) * np.nan
        for i,ind in enumerate(self.outSampleEmb_index):
            self.outSampleX[i,] = self.data.ts[range(ind - self.tauEmb * self.m,ind, self.tauEmb)]
        self.outSampleX = (self.outSampleX - self.inSampleX_mean)/self.inSampleX_std
        # PYTORCH 6
        self.outSampleDesignMatrix = torch.column_stack([torch.ones(self.outSampleEmb_len, dtype = torch.int8, device = self.device), self.outSampleX.reshape(self.outSampleEmb_len,-1)])
        # Original
        # self.outSampleDesignMatrix=np.column_stack([np.repeat(1,self.outSampleEmb_len),self.outSampleX .reshape(self.outSampleEmb_len,-1)])

        #### Y
        self.outSampleY = (self.data.ts[self.outSampleEmb_index] - self.inSampleY_mean)/self.inSampleY_std
        
    def get_w_and_u(self):
        # PYTORCH 7
        wMat = torch.FloatTensor(self.nh*self.nh).to(self.device).uniform_(-self.wWidth,self.wWidth).reshape(self.nh, -1)
        # Original
        # wMat = np.random.uniform(-self.wWidth,self.wWidth,self.nh*self.nh).reshape(self.nh,-1)
        # PYTORCH 8
        uMat = torch.FloatTensor(self.nh*self.nColsU).to(self.device).uniform_(-self.uWidth,self.uWidth).reshape(self.nColsU, -1)
        # Original
        # uMat = np.random.uniform(-self.uWidth,self.uWidth,self.nh*self.nColsU).reshape(self.nColsU,-1)
        #Make W Matrix Sparse 
        for i in range(self.nh):
            # PYTORCH 44

            # Original
            numReset=self.nh-np.random.binomial(self.nh,self.wSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            wMat[resetIndex,i]=0

        #Make U Matrix Sparse 
        for i in range(self.nColsU):
            numReset=self.nh-np.random.binomial(self.nh,self.uSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            uMat[i,resetIndex]=0
        
        #Scale W Matrix
        # PYTORCH 9
        v = torch.linalg.eigvalsh(wMat)
        # Original
        # v = eigh(wMat.cpu(),eigvals_only=True)
        spectralRadius = max(abs(v))
        wMatScaled=wMat*self.delta/spectralRadius

        return wMatScaled, uMat
    
    def get_hMat(self,wMat,uMat):
        #Create H Matrix in-sample
        hMatDim = 2*self.nh
        # PYTORCH 10
        uProdMat = torch.mm(self.inSampleDesignMatrix.double(), uMat.double())
        # Original
        # uProdMat=self.inSampleDesignMatrix.dot(uMat);
        
        # PYTORCH 11
        hMat = torch.zeros((hMatDim,self.inSampleEmb_len), device = self.device)
        # Original
        # hMat = np.zeros((hMatDim,self.inSampleEmb_len))
        
        xTemp = uProdMat[0,:]
        # PYTORCH 12
        xTemp = torch.tanh(xTemp)
        # Original
        # xTemp = np.tanh(xTemp)

        hMat[0:self.nh,0] = xTemp
        hMat[self.nh:,0] = xTemp*xTemp
        
        for t in range(1,self.inSampleEmb_len):
            # PYTORCH 13
            xTemp = torch.mm(wMat.double(), xTemp.reshape(-1,1).double()).reshape(1,-1) + uProdMat[t,:]
            # Original
            # xTemp = wMat.dot(xTemp)+uProdMat[t,:]
            # PYTORCH 14
            xTemp = torch.tanh(xTemp)
            # Original
            # xTemp = np.tanh(xTemp)
            hMat[0:self.nh,t] = xTemp*self.alpha + hMat[0:self.nh,t-1]*(1-self.alpha)
            hMat[self.nh:,t] = hMat[0:self.nh,t]*hMat[0:self.nh,t]

        #Create H Matrix out-sample
        # PYTORCH 15
        uProdMatOutSample = torch.mm(self.outSampleDesignMatrix.double(), uMat.double())
        # Original
        # uProdMatOutSample = self.outSampleDesignMatrix.dot(uMat)
        # PYTORCH 16
        hMatOutSample = torch.zeros((self.outSampleEmb_len,hMatDim), device = self.device)
        # Original
        # hMatOutSample = np.zeros((self.outSampleEmb_len,hMatDim))
        # PYTORCH 17
        xTemp = torch.mm(wMat.double(), xTemp.reshape(-1,1).double()).reshape(1,-1) + uProdMatOutSample[0,:]
        # Original
        # xTemp = wMat.dot(xTemp)+uProdMatOutSample[0,:]
        # PYTORCH 18
        xTemp = torch.tanh(xTemp)
        # Original
        # xTemp = np.tanh(xTemp)
        hMatOutSample[0,0:self.nh] = xTemp
        hMatOutSample[0,self.nh:] = xTemp*xTemp
        for t in range(1,self.outSampleEmb_len):
            # PYTORCH 19
            xTemp = torch.mm(wMat.double(), xTemp.reshape(-1,1).double()).reshape(1,-1) + uProdMatOutSample[t,:]
            # Original
            # xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
            # PYTORCH 20
            xTemp = torch.tanh(xTemp)
            # Original
            # xTemp = np.tanh(xTemp)

            hMatOutSample[t,0:self.nh] = xTemp*self.alpha + hMatOutSample[t-1,0:self.nh]*(1-self.alpha)
            hMatOutSample[t,self.nh:] = hMatOutSample[t,0:self.nh]*hMatOutSample[t,0:self.nh]

        return hMat, hMatOutSample
        
    def train(self,hyper_para):
        
        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity = hyper_para; 
        
        self.m = int(self.m)
        self.nh = int(self.nh)
        
        self.numLocs = self.data.ts.shape[1]
        
        self.standardize_in_sample()
            
    def forecast(self):
        '''
            Forecast.
        
            Obtain the forecast matrix, forMat: 
                * dimension: (#ensemble, #forecast time points, #locations, #prediction ahead time)
                * forMat[e,t,s,p] is the (p+1)-time ahead forecast for time t (instead of t+p+1!!) 
                    at location s from e-th ensemble
        '''
        print("Forecasting, ensemble: ", end="")
        
        self.standardize_out_sample()
        # PYTORCH 3
        self.forMat = torch.ones((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred), device = "cpu") * torch.nan
        # Original
        # self.forMat = np.ones((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred)) * np.nan
        self.nColsU = self.numLocs * self.m + 1
        
        for iEnsem in range(self.ensembleLen):
            print(iEnsem+1,end=" ")
            
            # PYTORCH 1.forMat
            forMat_iEnsem___0 = torch.zeros_like(self.forMat[iEnsem,:,:,0], device = self.device)
            # END PYTORCH 1.forMat
            
            wMat, uMat = self.get_w_and_u();

            hMat, hMatOutSample = self.get_hMat(wMat,uMat);
            # Ridge Regression to get out-sample forecast
            # PYTORCH 21
            tmp = torch.mm(hMat, hMat.T)
            # Original
            # tmp = hMat.dot(hMat.transpose())
            # PYTORCH 22
            tmp[range(len(tmp)), range(len(tmp))] = tmp.diagonal()+self.ridge
            # Original
            # np.fill_diagonal(tmp,tmp.diagonal()+self.ridge)

            # PYTORCH 23
            forMat_iEnsem___0 = torch.mm(hMatOutSample.double(), torch.linalg.solve(tmp.double(), torch.mm(hMat.double(), self.inSampleY)))
            # Original
            # self.forMat[iEnsem,:,:,0] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))

            #Transform to the original scale
            # PYTROCH 39
            forMat_iEnsem___0 = forMat_iEnsem___0 * self.inSampleY_std + self.inSampleY_mean
            self.forMat[iEnsem,:,:,0] = forMat_iEnsem___0.cpu()
            # Original
            # self.forMat[iEnsem,:,:,0] = self.forMat[iEnsem,:,:,0] * self.inSampleY_std + self.inSampleY_mean

            #### Prediction at t + pred_lag + 1 where t is the current time
            hMatDim = 2*self.nh
            for pred_lag in range(1,self.numTimePred):
                
                TIME_FORE_5 = perf_counter()

                #Create H Matrix out-sample for prediction more than one lead time
                # PYTORCH 24
                self.outSampleX_mixed = self.outSampleX.clone()
                # Original
                # self.outSampleX_mixed = self.outSampleX.copy()
                for i in range(min(pred_lag,self.m)):
                    ii = i+1
                    # PYTORCH 40
                    self.outSampleX_mixed[pred_lag:,-ii,:] = (self.forMat[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii].to(self.device) - self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]
                    # Original
                    # self.outSampleX_mixed[pred_lag:,-ii,:] = (self.forMat[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii] - self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]

                # PYTORCH 25
                self.outSampleX_mixed[0:pred_lag,] = torch.nan
                # Original
                # self.outSampleX_mixed[0:pred_lag,] = np.nan
                # PYTORCH 26
                self.outSampleDesignMatrix_mixed = torch.column_stack([torch.ones(self.outSampleEmb_len, device = self.device),
                                                    self.outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])
                # Original
                # self.outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,self.outSampleEmb_len),
                #                                    self.outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])
    
                # PYTORCH 27
                uProdMatOutSample = torch.mm(self.outSampleDesignMatrix_mixed, uMat.double())
                # Original
                # uProdMatOutSample = self.outSampleDesignMatrix_mixed.dot(uMat)

                # PYTORCH 28
                hMatOutSample_new = torch.zeros((self.outSampleEmb_len,hMatDim), device = self.device) * torch.nan
                # Original
                # hMatOutSample_new = np.zeros((self.outSampleEmb_len,hMatDim)) * np.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                # PYTORCH 29
                xTemp = torch.mm(wMat, xTemp.reshape(-1,1)).reshape(1,-1) + uProdMatOutSample[pred_lag,:]
                # Original
                # xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
                # PYTORCH 30
                xTemp = torch.tanh(xTemp)
                # Original
                # xTemp = np.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    # PYTORCH 31
                    xTemp = torch.mm(wMat, xTemp.reshape(-1,1)).reshape(1,-1) + uProdMatOutSample[t,:]
                    # Original
                    # xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
                    # PYTORCH 32
                    xTemp = torch.tanh(xTemp)
                    # Original
                    # xTemp = np.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh] * hMatOutSample_new[t,0:self.nh]
                
                # PYTORCH 33
                hMatOutSample = hMatOutSample_new.clone()
                # Original
                # hMatOutSample = hMatOutSample_new.copy()
                
                # PYTORCH 34
                self.forMat[iEnsem,:,:,pred_lag] = torch.mm(hMatOutSample.double(), torch.linalg.solve(tmp.double(), torch.mm(hMat.double(), self.inSampleY))).to("cpu")
                # Original
                # self.forMat[iEnsem,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
                               
                #Transform to the original scale
                # PYTORCH 41
                self.forMat[iEnsem,:,:,pred_lag] = self.forMat[iEnsem,:,:,pred_lag] * self.inSampleY_std.to("cpu") + self.inSampleY_mean.to("cpu")
                # Original
                # self.forMat[iEnsem,:,:,pred_lag] = self.forMat[iEnsem,:,:,pred_lag] * self.inSampleY_std + self.inSampleY_mean


    def cross_validation(self,cv_para,mChanged = True):
        '''
            Input: 
                cv_para: the cross-validation parameter [m, nh, ridge, delta, alpha, wWidth, uWidth, wSparsity, uSparsity]
                mChange: if m in this cross-validation is different than the last one. If no, there is no need to 
                        re-standardize the in-sample and out-sample data
            Output:
                MSE: vector of MSE with dimension self.numTimePred, which are the mean forecast square error for the different
                     time ahead forecast
        '''
    
        print("Cross Validation with Multiple Lead Times:")

        self.numLocs = self.data.ts.shape[1]
        
        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity = cv_para; 
    
        self.m = int(self.m)
        self.nh = int(self.nh)
        
        self.nColsU = self.numLocs * self.m + 1
        
        if(mChanged):
            self.standardize_in_sample(True)
            self.standardize_out_sample(True)
        
        forMatCV = np.zeros((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred))

        for iEnsem in range(self.ensembleLen):
            wMat, uMat = self.get_w_and_u();

            hMat, hMatOutSample = self.get_hMat(wMat,uMat);

            #Ridge Regression to get out-sample forecast
            tmp = hMat.dot(hMat.transpose())
            np.fill_diagonal(tmp,tmp.diagonal()+self.ridge)

            forMatCV[iEnsem,:,:,0] += hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
            
            #### Prediction at t + pred_lag + 1 where t is the current time
            hMatDim = 2*self.nh
            for pred_lag in range(1,self.numTimePred):

                #Create H Matrix out-sample for prediction more than one lead time
                outSampleX_mixed = self.outSampleX.copy()

                for i in range(min(pred_lag,self.m)):
                    ii = i+1
                    forMatCV_scaled_back  = forMatCV[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii] * self.inSampleY_std + self.inSampleY_mean
                    outSampleX_mixed[pred_lag:,-ii,:] = (forMatCV_scaled_back - self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]

                outSampleX_mixed[0:pred_lag,] = np.nan
                outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,self.outSampleEmb_len),
                                                    outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])
    
                uProdMatOutSample = outSampleDesignMatrix_mixed.dot(uMat)

                hMatOutSample_new = np.zeros((self.outSampleEmb_len,hMatDim)) * np.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
                xTemp = np.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
                    xTemp = np.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh] * hMatOutSample_new[t,0:self.nh]

                hMatOutSample = hMatOutSample_new.copy()
                
                forMatCV[iEnsem,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
        
        
        forMatCVmean = forMatCV.mean(axis = 0)

        diff = np.ndarray(shape = forMatCVmean.shape) * np.nan

        for i in range(self.numTimePred):
            diff[:,:,i] = forMatCVmean[:,:,i] - self.outSampleY

        MSPE = np.nanmean(diff**2,axis=(0,1))
        
        return MSPE
    
    def compute_forecast_mean(self):
        '''
            Compute the ensemble forecast mean, forMean:
                * dimension: (#forecast time points, #locations, #prediction ahead time)
                * forMean[t,s,p] is the (p+1)-time ahead forecast mean for time t (instead of t+p+1!!) at location s 
        '''
        self.forMean = self.forMat.mean(axis=0)
        self.forMeanComputed = True
        
    def compute_forecast_error(self):
        '''
            Compute the error by the ensemble forecast mean, forError:
                * dimension: (#forecast time points, #locations, #prediction ahead time)
                * forError[t,s,p] is the (p+1)-time ahead forecast error for time t (instead of t+p+1!!) at location s 
        '''
        if(not self.forMeanComputed):
            self.compute_forecast_mean()
        
        # PYTORCH 35
        self.forError = torch.zeros_like(self.forMean, device = self.device)
        # Original
        # self.forError = np.zeros_like(self.forMean)
        # PYTORCH 36
        self.forError.fill_(torch.nan)
        # Original
        # self.forError.fill(np.nan)
        
        for ahead in range(self.numTimePred):
            # PYTORCH 43
            self.forError[:,:,ahead] = self.forMean[:,:,ahead].to(self.device) -  self.data.ts[self.outSampleEmb_index]
            # Original
            # self.forError[:,:,ahead] = self.forMean[:,:,ahead] -  self.data.ts[self.outSampleEmb_index]

        self.forErrorComputed = True

    def compute_MSPE(self):
        if(not self.forErrorComputed):
            self.compute_forecast_error()
        
        # PYTORCH 37
        return torch.nanmean(self.forError**2,axis = (0,1))
        # Original
        # return np.Nanmean(self.forError**2,axis = (0,1))
