import numpy as np
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm

import rpy2.robjects as robjects
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import norm

m = Basemap(resolution='l',llcrnrlon=33.6, llcrnrlat=15.4,\
    urcrnrlon=56.6,urcrnrlat=33.2)

meridians = np.arange(35,59,5)
parallels = np.arange(17,33,5)

def draw_power_quantile(quantile, esn_total_power_turbine_height_quantile_diff):
    fig = plt.figure(figsize = (10,6))
    
    ax = plt.subplot(111)
    plt.plot(quantile, esn_total_power_turbine_height_quantile_diff,
            'ko-', linewidth = 2)
    
    plt.xticks([0.025, *np.arange(0.1,1,0.1), 0.975], 
               [2.5, *np.arange(10,100,10), 97.5], 
               fontsize = 15)
    plt.yticks(fontsize = 15)
    tx = ax.yaxis.get_offset_text()
    tx.set_fontsize(15)

    plt.ylabel('Sum of Absolute Differences\nin Wind Energy (kW'+ r'$\cdot$' +'h)', fontsize = 20)
    plt.xlabel('Quantiles (%)', fontsize = 20)
    plt.tight_layout()
    
def draw_nonstationary_model(ns_model_file):
    robjects.r['load'](ns_model_file);
    model = robjects.r['NS_model']
    
    names = np.array(model.names)
    mc_kernels = np.array(model[int(np.where(names == "mc.kernels")[0][0])])
    mc_locations = np.array(model[int(np.where(names == "mc.locations")[0][0])])
    coords = np.array(model[int(np.where(names == "coords")[0][0])])
    kappa_est = np.array(model[int(np.where(names == "kappa.est")[0][0])])
    sigmasq_est = np.array(model[int(np.where(names == "sigmasq.est")[0][0])])    

    plt.figure(figsize = (30,8))
    ax = plt.subplot(131)
    m.scatter(mc_locations[:,0],mc_locations[:,1],marker='.', color = 'red', s = 10)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0, fontsize = 30)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=0, fontsize = 30)
    m.drawcoastlines(linewidth=1, color="black")
    m.drawcountries(linewidth=1, color="black")

    for i in range(mc_locations.shape[0]):
        cov = mc_kernels[:,:,i]
        (lambda1, lambda2), _ = np.linalg.eig(cov)
        eta = np.arcsin(cov[0,1]*2/(lambda1-lambda2))/2*180

        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        ellipse = Ellipse((0,0),  
                          width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor = 'none', edgecolor = 'blue')

        n_std = -norm.ppf((1-0.7)/2)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std

        transf = transforms.Affine2D().rotate_deg(45)\
            .scale(scale_x, scale_y)\
            .translate(mc_locations[i,0],mc_locations[i,1])

        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    ax.set_title('(A) Ellipses of ' +r'$\hat\Sigma$', fontsize = 40)
    ax.set_aspect('auto')

    ax = plt.subplot(132)
    imc = m.scatter(coords[:,0],coords[:,1],marker='.', c= sigmasq_est, cmap = cm.rainbow) 
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0, fontsize = 30)
    m.drawcoastlines(linewidth=1, color="black")
    m.drawcountries(linewidth=1, color="black")
    ax.set_title('(B) Partial Sill ' +r'$\hat\sigma^2$', fontsize = 40)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(imc, cax=cax)
    cbar.ax.tick_params(labelsize = 30)
    ax.set_aspect('auto')

    ax = plt.subplot(133)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0, fontsize = 30)
    imc = m.scatter(coords[:,0],coords[:,1],marker='.', c= kappa_est, cmap = cm.rainbow) 
    m.drawcoastlines(linewidth=1, color="black")
    m.drawcountries(linewidth=1, color="black")
    ax.set_title('(C) Smoothness ' +r'$\hat\nu$', fontsize = 40)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(imc, cax=cax)
    cbar.ax.tick_params(labelsize = 30)
    ax.set_aspect('auto')
    plt.tight_layout()

def draw_outperformance_mse_all(lon_all_locations, lat_all_locations, esn_mse_all_locations, per_mse_all_locations):
    outperformance_all_locations = esn_mse_all_locations - per_mse_all_locations
    
    fig = plt.figure(figsize = (20,6))
    grid = ImageGrid(fig, 111,
                    nrows_ncols = (1,3),
                    axes_pad = 0.2,
                    cbar_location = "right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.1
                    )

    vmin = 0.8
    m.scatter(lon_all_locations,lat_all_locations,c=outperformance_all_locations[:,0],marker='.', s = 1, vmin = (-vmin,vmin), cmap = cm.RdBu_r, ax = grid[0])
    m.drawmeridians(meridians,labels=[True,False,False,True],color='gray', ax = grid[0], linewidth=0, fontsize = 20)
    m.drawparallels(parallels,labels=[True,False,False,True],color='gray', ax = grid[0], linewidth=0, fontsize = 20)
    m.drawcoastlines(linewidth=1, color="black", ax = grid[0])
    m.drawcountries(linewidth=1, color="black", ax = grid[0])
    grid[0].set_title('One Hour Ahead',fontsize = 30)
    grid[0].set_xlabel('Longitude',fontsize = 25, labelpad = 20)
    grid[0].set_ylabel('Latitude',fontsize = 25, labelpad = 60)

    m.scatter(lon_all_locations,lat_all_locations,c=outperformance_all_locations[:,1],marker='.', s = 1, vmin = (-vmin,vmin), cmap = cm.RdBu_r, ax = grid[1])
    m.drawmeridians(meridians,labels=[True,False,False,True],color='gray', ax = grid[1], linewidth=0, fontsize = 20)
    m.drawparallels(parallels,labels=[False,False,False,True],color='gray', ax = grid[1], linewidth=0)
    m.drawcoastlines(linewidth=1, color="black", ax = grid[1])
    m.drawcountries(linewidth=1, color="black", ax = grid[1])
    grid[1].set_title('Two Hours Ahead',fontsize = 30)
    grid[1].set_xlabel('Longitude',fontsize = 25, labelpad = 20)

    imc = m.scatter(lon_all_locations,lat_all_locations,c=outperformance_all_locations[:,2],marker='.', s = 1, vmin = (-vmin,vmin), cmap = cm.RdBu_r, ax = grid[2])
    m.drawmeridians(meridians,labels=[True,False,False,True],color='gray', ax = grid[2], linewidth = 0, fontsize = 20)
    m.drawparallels(parallels,labels=[False,False,False,True],color='gray', ax = grid[2], linewidth = 0)
    m.drawcoastlines(linewidth=1, color="black", ax = grid[2])
    m.drawcountries(linewidth=1, color="black", ax = grid[2])
    grid[2].set_title('Three Hours Ahead',fontsize = 30)
    grid[2].set_xlabel('Longitude',fontsize = 25, labelpad = 20)

    cbar = plt.colorbar(imc, cax=grid.cbar_axes[0])
    cbar.ax.tick_params(labelsize=18) 
    plt.tight_layout()
    
def draw_outperformance_mse_knots(lon_knots, lat_knots, esn_err, per_err):
    outperformance_knots = np.nanmean(esn_err**2,axis = 0) - np.nanmean(per_err**2,axis = 0) 
    
    fig = plt.figure(figsize = (20,6))
    grid = ImageGrid(fig, 111,
                    nrows_ncols = (1,3),
                    axes_pad = 0.2,
                    cbar_location = "right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.1
                    )

    vmin = 0.5
    m.scatter(lon_knots,lat_knots,c=outperformance_knots[:,0],marker='.', s = 10, vmin = (-vmin,vmin), cmap = cm.coolwarm, ax = grid[0])
    m.drawmeridians(meridians,labels=[True,False,False,True],color='gray', linewidth=0, fontsize = 20, ax = grid[0])
    m.drawparallels(parallels,labels=[True,False,False,True],color='gray', linewidth=0, fontsize = 20, ax = grid[0])
    m.drawcoastlines(linewidth=1, color="black", ax = grid[0])
    m.drawcountries(linewidth=1, color="black", ax = grid[0])
    grid[0].set_title('One Hour Ahead',fontsize = 30)
    grid[0].set_xlabel('Longitude',fontsize = 25, labelpad = 20)
    grid[0].set_ylabel('Latitude',fontsize = 25, labelpad = 60)


    m.scatter(lon_knots,lat_knots,c=outperformance_knots[:,1],marker='.', s = 10, vmin = (-vmin,vmin), cmap = cm.coolwarm, ax = grid[1])
    m.drawmeridians(meridians,labels=[True,False,False,True],color='gray', linewidth=0, fontsize = 20, ax = grid[1])
    m.drawparallels(parallels,labels=[False,False,False,True],color='gray', linewidth=0, ax = grid[1])
    m.drawcoastlines(linewidth=1, color="black", ax = grid[1])
    m.drawcountries(linewidth=1, color="black", ax = grid[1])
    grid[1].set_title('Two Hours Ahead',fontsize = 30)
    grid[1].set_xlabel('Longitude',fontsize = 25, labelpad = 20)

    imc = m.scatter(lon_knots,lat_knots,c=outperformance_knots[:,2],marker='.', s = 10, vmin = (-vmin,vmin), cmap = cm.coolwarm, ax = grid[2])
    m.drawmeridians(meridians,labels=[True,False,False,True],color='gray', linewidth = 0, fontsize = 20, ax = grid[2])
    m.drawparallels(parallels,labels=[False,False,False,True],color='gray', linewidth = 0, ax = grid[2])
    m.drawcoastlines(linewidth=1, color="black", ax = grid[2])
    m.drawcountries(linewidth=1, color="black", ax = grid[2])
    grid[2].set_title('Three Hours Ahead',fontsize = 30)
    grid[2].set_xlabel('Longitude',fontsize = 25, labelpad = 20)

    cbar = plt.colorbar(imc, cax=grid.cbar_axes[0])
    cbar.ax.tick_params(labelsize=18)
    plt.tight_layout()
