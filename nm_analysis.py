#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:22:37 2024

@author: yatangli
"""

# -*- coding: utf-8 -*-

#%%
import sys, os, scipy, matplotlib, hdf5storage, timeit, moten
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

if dname not in sys.path: 
    sys.path.append(dname)

from nm_utils import *
#%% define file path
folder_path = os.path.dirname(dname)+'/'

folder_mov = folder_path+'natural_movies/'
video_file_list = ['1_baby_owl-converted.mp4',
                   '2_running_cats-converted.mp4',
                   '3_foraging-converted.mp4',
                   '4_optical_flow.avi']



folder_data = folder_path+'data/'

data_path = {
            'neuron':'neuron.mat',
            'chirp':'chirp.mat',
            'dl_lr':'dl_lr.mat',
            'mb':'mb.mat',
            'color':'color.mat',
            'st':'st.mat',
            'rf':'rf.mat',
            'nm':'nm.mat'
            }

# save plot to the figures folder
folder_figure = folder_path+'figures/'
save_fig = True
fig_width = 8
fig_height = 18
#%% analyze movies
fps = 30        
load_mov_results = True
if load_mov_results:
    motion_energy_arr = np.loadtxt(folder_data+'motion_energy.csv')   
    brightness_arr = np.loadtxt(folder_data+'brightness.csv')
    contrast_arr = np.loadtxt(folder_data+'contrast.csv')
    len_ls = [1886,898,503,225]
else:
    down_sample_ls = [10,10,3,10]
    len_ls = []
    for i in range(len(video_file_list)):
        video_file = folder_mov + video_file_list[i]
        luminance_images_raw = moten.io.video2luminance(video_file)
        luminance_images = luminance_images_raw[:,::down_sample_ls[i],::down_sample_ls[i]]
    
        #% Create a pyramid of spatio-temporal gabor filters
        nimages, vdim, hdim = luminance_images.shape
        len_ls.append(nimages)
        pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=fps)
        
        # Compute motion energy features
        moten_features = pyramid.project_stimulus(luminance_images)
        motion_energy_mean = np.mean(moten_features,axis=1)
        
        brightness_mean = np.mean(luminance_images,axis=(1,2))
        contrast_mean = np.std(luminance_images,axis=(1,2))

        if i==0:
            motion_energy_arr = np.copy(motion_energy_mean)
            brightness_arr = np.copy(brightness_mean)
            contrast_arr = np.copy(contrast_mean)
        else:
            motion_energy_arr = np.concatenate((motion_energy_arr,motion_energy_mean))
            brightness_arr = np.concatenate((brightness_arr,brightness_mean))
            contrast_arr = np.concatenate((contrast_arr,contrast_mean))
            
    np.savetxt(folder_data+'motion_energy.csv',motion_energy_arr)
    np.savetxt(folder_data+'brightness.csv',brightness_arr)
    np.savetxt(folder_data+'contrast.csv',contrast_arr)
#%
len_arr = np.array(len_ls)
len_cumsum = np.cumsum(len_arr)
n = 3
n_mov = 4
out_arr = np.stack((brightness_arr,contrast_arr,motion_energy_arr),axis=1)
t = np.arange(out_arr.shape[0])/fps
# fig, ax = plt.subplots(n,n_mov,figsize=(20, 4))
fig = plt.figure(figsize=(13.5, 4))
margin = 0.03
width_arr = (1-margin*(n_mov-1))*(len_arr/len_cumsum[-1])
width_all = 1+margin*n_mov
for i in range(n):
    for j in range(n_mov):
        width = width_arr[j]
        if j==0:
            left = 0
        else:
            left = np.sum(width_arr[:j])+j*margin
        
        if i==0:
            print([left,width])

        bottom = 1-i/3
        height = 1/3-margin
        ax = fig.add_axes([left,bottom,width,height])
        t = np.arange(len_arr[j])/fps
        if j == 0:
            ax.plot(t,out_arr[:len_cumsum[j],i],'k')
        elif j == n_mov-1:
            ax.plot(t,out_arr[len_cumsum[j-1]:,i],'k')
        else:
            ax.plot(t,out_arr[len_cumsum[j-1]:len_cumsum[j],i],'k')
        if i<2:
            ax.set_xticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(t[0]-0.5,t[-1]+0.5)

savefig(folder_figure+'nm_fig_1C')

#%% load data, data are re-sampled at 5Hz
sampling_rate = 5 #Hz
neuron,chirp,dl_lr,mb,color,st,rf,nm = load_data_new(folder_data, data_path)
num_neuron = neuron['soma_size'].size
num_image = np.unique(neuron['image_id']).size
#%% extract features from chirp data using sparse PCA;  
# note mean(pc)=0 but std(pc)!=1
pc_spca_chirp, pv_spca_chirp = sparse_pca_analysis(chirp['chirp_temporal_norm'], k_spca=20)
pc_spca_mb, pv_spca_mb = sparse_pca_analysis(mb['mb_temporal_sorted_truncated_norm'], k_spca=6)
pc_spca_dl_lr, pv_spca_dl_lr = sparse_pca_analysis(dl_lr['dl_lr_temporal_sorted_truncated_norm'], k_spca=6)
pc_spca_color, pv_spca_color = sparse_pca_analysis(color['color_temporal_norm'], k_spca=8)
pc_spca_st, pv_spca_st = sparse_pca_analysis(st['st_temporal_norm'], k_spca=10)

#%% make the feature matrix X and its temporal profile X_temporal
feature_other_list = [dl_lr['dl_hi'], mb['mb_gdsi'], mb['mb_gosi'], mb['msi']]

for i, m in enumerate(feature_other_list):
    if len(m.shape) == 1:
        m = np.expand_dims(m, axis=1)
    if i == 0:
        feature_other_raw = m
    else:
        feature_other_raw = np.append(feature_other_raw, m, axis=1)

feature_other = StandardScaler().fit_transform(feature_other_raw)
feature_other_norm = norm_x(feature_other_raw)

X_sorted = np.concatenate((pc_spca_chirp,pc_spca_mb,pc_spca_dl_lr,pc_spca_color,
                           pc_spca_st,feature_other), axis=1)

X_temporal = np.concatenate((mb['mb_temporal_sorted_truncated_norm'],
                             dl_lr['dl_lr_temporal_sorted_truncated_norm'], 
                             chirp['chirp_temporal_norm'], st['st_temporal_norm'], 
                             color['color_temporal_norm'], feature_other_norm), axis=1)

X = StandardScaler().fit_transform(X_sorted)
_,num_feature = X.shape

#%% find the optimal cluster number using GMM, this takes 1-2 days.
load_gmm_data = True
if load_gmm_data:
    bic =  np.loadtxt(folder_path+'data/bic.csv',delimiter=',')
    n_cluster_optimal = np.argmin(bic)+2
    labels_arr = np.loadtxt(folder_path+'data/labels_arr.csv',delimiter=',')
    # labels_arr_optimal = np.loadtxt(folder_path+'data/labels_arr_optimal.csv',delimiter=',')
else:
    start = timeit.default_timer()
    n_cluster_max = 50
    # [n_cluster_optimal, aic, bic, silhouette_score,labels_arr] = optimal_cluster_gmm(X,n_cluster_max,random_init=True)
    [n_cluster_optimal, _, bic_all, _,labels_arr_all,_] = optimal_cluster_gmm(X,n_cluster_max,random_init=False,n_init=1000)
    stop = timeit.default_timer()
    print(stop-start)
    labels_arr_all = labels_arr_all.astype(dtype=np.dtype('uint8'))
    bic_optimal_all = bic_all[n_cluster_optimal-2,:]
    bic_optimal_all_sorted = np.sort(bic_optimal_all)
    bic = np.min(bic_all,axis=1)
    bic_idx = np.argmin(bic_all,axis=1)
    labels_arr = np.zeros((num_neuron,n_cluster_max-1),dtype=np.dtype('uint8'))
    for i in range(n_cluster_max-1):
        labels_arr[:,i] = labels_arr_all[:,i,bic_idx[i]]
    # labels_arr_optimal = labels_arr_all[:,n_cluster_optimal-2,:]
    np.savetxt(folder_path+'data/bic.csv', bic, delimiter=',')
    np.savetxt(folder_path+'data/labels_arr.csv', labels_arr, delimiter=',')
    # np.savetxt(folder_path+'data/labels_arr_optimal.csv', labels_arr_optimal, delimiter=',')

num_type_art = n_cluster_optimal    

#%% cluster neurons based on labels    
labels_gmm = labels_arr[:,num_type_art-2]    
# show clusters and their distance in feature space
X_cluster_gmm,_,_,_ = plot_cluster(X,labels_gmm,[],sampling_rate=1)    
cluster_dist_gmm = plot_cluster_dist(X_cluster_gmm)
# show cluster in temporal space
X_temporal_cluster_gmm,X_temporal_cluster_gmm_error,_,_ = plot_cluster(X_temporal,labels_gmm,[])

#%% plot dendrogram as 2d image and resort the temproal matrix
fig_params_dendro = {'ax_dendro':[0,0,0.1,1],
                     'ax_matrix':[0.132,0,0.8,1],
                     'ax_colorbar':[0.94,0,0.02,1],
                     'text_offset':[0,3]}

idx_sorted, labels_gmm_dendro, cluster_size_gmm = dendrogram_x(X_cluster_gmm, X_temporal_cluster_gmm, labels_gmm, 
                                                               fig_params_dendro, [], fig_zoom=2.5,plot_bool=False) 

X_cluster_gmm_dendro, X_cluster_gmm_dendro_error, _, X_cluster_gmm_dendro_num = plot_cluster(
    X,labels_gmm_dendro,[],sampling_rate=1)     

pv_k_X_cluster_gmm_dendro, lamda_cumsum_X_cluster_gmm_dendro, pc_k_X_cluster_gmm_dendro = pca_x(X_cluster_gmm_dendro)
pc_k_X_cluster_gmm_dendro_reverse = pc_k_X_cluster_gmm_dendro[::-1,:]

#%% clustering neurons based on natural movie

nm_temporal_raw = nm['nm_temporal']
nm_temporal_se_raw = nm['nm_temporal_se']
nm_snr_raw = nm['nm_snr']

nm_temporal_0 = nm_temporal_raw#[idx_good_snr,:]
nm_temporal_se_0 = nm_temporal_se_raw#[idx_good_snr,:]
nm_snr_0 = nm_snr_raw#[idx_good_snr,0]
# swith the order of foraging and running cats
# the new order is [baby owls, running cats, foraging, and optic flows]
nm_duration_0 = np.asarray([311,83,148,35],dtype=int) #seconds from the video files
nm_cumsum_0 = np.cumsum(nm_duration_0)

nm_temporal = np.concatenate((nm_temporal_0[:,:nm_cumsum_0[0]],nm_temporal_0[:,nm_cumsum_0[1]:nm_cumsum_0[2]],
                       nm_temporal_0[:,nm_cumsum_0[0]:nm_cumsum_0[1]],nm_temporal_0[:,nm_cumsum_0[2]:]),axis=1)
nm_temporal_se = np.concatenate((nm_temporal_se_0[:,:nm_cumsum_0[0]],nm_temporal_se_0[:,nm_cumsum_0[1]:nm_cumsum_0[2]],
                       nm_temporal_se_0[:,nm_cumsum_0[0]:nm_cumsum_0[1]],nm_temporal_se_0[:,nm_cumsum_0[2]:]),axis=1)
nm_snr = np.concatenate((nm_snr_0[:nm_cumsum_0[0]],nm_snr_0[nm_cumsum_0[1]:nm_cumsum_0[2]],
                       nm_snr_0[nm_cumsum_0[0]:nm_cumsum_0[1]],nm_snr_0[nm_cumsum_0[2]:]))
nm_duration = np.asarray([311,148,83,35],dtype=int) #seconds from the video files
nm_cumsum = np.cumsum(nm_duration)
nm_temporal_norm = norm_x(nm_temporal)
# imshow_X(nm_temporal,'natural_movie_raw') # amplitudes:[-5e3,2e4], need to correct it
imshow_X(nm_temporal_norm[2000:3000,:],'natural_movie_norm')
sns.histplot(nm_snr)


#%% cluster neurons according to nm responses
# if only select neurons with non-nan rf, cluster responding to optic flow does not emerge
rf_position = np.stack((rf['azimuth_rf_flash'],rf['elevation_rf_flash']),axis=1)
idx_rf_position = np.logical_and(np.logical_not(np.isnan(rf_position[:,0])),np.logical_not(np.isnan(rf_position[:,1])))
rf_position_not_nan = rf_position[idx_rf_position,:]
plt.plot(rf_position_not_nan[:,0],rf_position_not_nan[:,1],'.')


# fill nan
row_nan, col_nan = np.where(np.isnan(nm_temporal_norm))
mean_col_nan = np.nanmean(nm_temporal_norm[:,col_nan[0]:col_nan[-1]+1],axis=0)
for row in range(row_nan.min(),row_nan.max()+1):
    nm_temporal_norm[row,col_nan[0]:col_nan[-1]+1] = mean_col_nan + np.random.rand() - 0.5
k_nm, pc_k_nm, pv_scaled_nm, nm_temporal_norm_new, lamda_sum = pca_analysis(nm_temporal_norm, 0.7)
_,num_feature_nm = np.shape(pc_k_nm)

#%% find the optimal cluster number for naturl movie respones

load_gmm_nm_data = True
if load_gmm_nm_data:
    bic_all_nm =  np.loadtxt(folder_path+'data/bic_all_nm.csv',delimiter=' ')
    bic_nm = np.min(bic_all_nm,axis=1)
    n_cluster_optimal_nm = np.argmin(bic_nm)+2
    labels_arr_nm = np.loadtxt(folder_path+'data/labels_arr_nm.csv',delimiter=' ')
else:
    start = timeit.default_timer()
    n_cluster_max = 50
    # [n_cluster_optimal_nm, aic_nm, bic_nm, bic_delta_nm, silhouette_score_nm, labels_list_nm] = optimal_cluster_gmm(pc_k_nm,n_cluster_max)
    [n_cluster_optimal_nm, aic_all_nm, bic_all_nm, silhouette_score_all_nm,labels_arr_all_nm,_] = optimal_cluster_gmm(pc_k_nm,n_cluster_max,random_init=False,n_init=1000)
    stop = timeit.default_timer()
    print(stop-start)
    labels_arr_all_nm = labels_arr_all_nm.astype(dtype=np.dtype('uint8'))
    
    bic_optimal_all_nm = bic_all_nm[n_cluster_optimal_nm-2,:]
    bic_optimal_all_sorted_nm = np.sort(bic_optimal_all_nm)
    bic_nm = np.min(bic_all_nm,axis=1)
    bic_idx_nm = np.argmin(bic_all_nm,axis=1)
    labels_arr_nm = np.zeros((num_neuron,n_cluster_max-1),dtype=np.dtype('uint8'))
    for i in range(n_cluster_max-1):
        labels_arr_nm[:,i] = labels_arr_all_nm[:,i,bic_idx_nm[i]]
    
    np.savetxt(folder_path+'data/bic_all_nm.csv',bic_all_nm)
    np.savetxt(folder_path+'data/bic_nm.csv',bic_nm)
    np.savetxt(folder_path+'data/labels_arr_nm.csv',labels_arr_nm)
    
n_cluster_optimal_nm = 16 # manually selected
num_type = n_cluster_optimal_nm
# plot figure number of cluster versus BIC, nm_Fig_2A
fig,ax = plt.subplots(1,1,figsize=(6,4))   
ax.plot(np.arange(2,51),(bic_nm-np.min(bic_nm))/np.max((bic_nm-np.min(bic_nm))),'-o',color='black',markersize=5)
if save_fig:
    savefig(folder_figure+'nm_Fig_2A')
plt.show()
#%% clustering with GMM 
labels_gmm_nm = labels_arr_nm[:,n_cluster_optimal_nm-2]
# X_cluster_gmm_nm = plot_cluster(nm_temporal_norm_rf_not_nan,labels_gmm_nm)
X_cluster_gmm_nm,X_cluster_gmm_nm_error,_,_ = plot_cluster(nm_temporal_norm,labels_gmm_nm,[])
cluster_dist_gmm_nm = plot_cluster_dist(X_cluster_gmm_nm)
# neuron_dist_nm = plot_cluster_dist(pc_k_nm)

#%% plot dendrogram as 2d image and resort the temproal matrix for natural movie data
fig_params_dendro_nm = {'ax_dendro':[0,0,0.1,1],
                     'ax_matrix':[0.132,0,0.8,1],
                     'ax_colorbar':[0.94,0,0.02,1],
                     'text_offset':[0,3]}

file_save_dendro_temporal_nm = ''
idx_dendro_nm, labels_gmm_nm_dendro, cluster_size_gmm_nm = dendrogram_x(X_cluster_gmm_nm, X_cluster_gmm_nm, \
                                                            labels_gmm_nm, fig_params_dendro_nm, file_save_dendro_temporal_nm, fig_zoom=3,plot_bool=False)  
    
file_save_neurons_temporal_labels_dendro_nm = folder_figure + 'neurons_temporal_labels_dendro_nm'
X_temporal_cluster_gmm_nm_dendro,_,_,_ = plot_cluster(nm_temporal_norm,labels_gmm_nm_dendro,file_save_neurons_temporal_labels_dendro_nm,plot_bool=False)


X_cluster_gmm_nm_dendro, X_cluster_gmm_nm_dendro_error, _, X_cluster_gmm_nm_dendro_num = plot_cluster(
    pc_k_nm,labels_gmm_nm_dendro,[],sampling_rate=1)

# nm_Fig_2B
fig,ax = plt.subplots(1,1,figsize=(9,6))   
norm = colors.TwoSlopeNorm(vcenter=0,vmin=-2.8,vmax=1.9)
# axs.pcolormesh
im = ax.pcolormesh(np.arange(k_nm)+1,np.arange(num_type)+1,X_cluster_gmm_nm_dendro,cmap='bwr',norm=norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_aspect('equal', 'box')
plt.colorbar(im, cax=cax)
ax.set_xticks(np.arange(5,k_nm+1,5))
ax.set_yticks(np.arange(2,num_type+1,2))
savefig(folder_figure+'nm_Fig_2B')     


#%% subsampling to test the cluster stability
num_sub = 100
n_init = 1000
sub_perc = 0.9
load_sub_data = True
if load_sub_data:
    X_cluster_gmm_sub_2d = np.loadtxt(folder_path+'data/X_cluster_gmm_nm_sub.csv',delimiter=',')
    X_cluster_gmm_sub = X_cluster_gmm_sub_2d.reshape(num_type,num_feature_nm,num_sub)
    labels_gmm_sub = np.loadtxt(folder_path+'data/labels_gmm_nm_sub.csv',delimiter=',')
    idx_sub = np.loadtxt(folder_path+'data/idx_nm_sub.csv',delimiter=',')
    
else:
    X_cluster_gmm_sub,labels_gmm_sub,idx_sub = cal_sub_sampling(pc_k_nm,n_cluster_optimal_nm,
                                                            sub_perc=sub_perc,num_sub=num_sub,n_init=n_init,random_init=False)

    np.savetxt(folder_path+'data/X_cluster_gmm_nm_sub.csv', X_cluster_gmm_sub.reshape(num_type,-1), delimiter=',')
    np.savetxt(folder_path+'data/labels_gmm_nm_sub.csv', labels_gmm_sub, delimiter=',')
    np.savetxt(folder_path+'data/idx_nm_sub.csv', idx_sub, delimiter=',')
#%% calcualte the co-association matrix
plot_co_association_matrix = True
if plot_co_association_matrix: 
    labels_gmm_sub_all = np.ones((num_neuron,num_sub),dtype=int)*np.nan
    idx_sub = idx_sub.astype(int)
    for i in range(num_sub):
        labels_gmm_sub_all[idx_sub[:,i],i] = labels_gmm_sub[:,i]
        
    load_co_mat = True
    if load_co_mat:
        co_mat_mean_sub = np.loadtxt(folder_path+'data/co_mat_mean_sub_nm.csv',delimiter=',')
    else:
        # it takes ~4 min
        start = timeit.default_timer()
        co_mat_mean_sub = cal_co_mat_gmm(pc_k_nm,labels_gmm_sub_all)
        stop = timeit.default_timer()
        print(stop-start)
        np.savetxt(folder_path+'data/co_mat_mean_sub_nm.csv', co_mat_mean_sub, delimiter=',')
    # sort it with clusters
    co_mat_mean_sub_sorted_cluster,neuron_sorted_idx = sort_co_mat_cluster(pc_k_nm,labels_gmm_nm_dendro,co_mat_mean_sub)
    
    # nm_fig_2C
    fig, axs = plt.subplots(1,1,figsize=[12,12])
    im = axs.pcolormesh(co_mat_mean_sub_sorted_cluster[::-1,::-1],cmap='viridis')
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    axs.set_aspect('equal', 'box')
    axs.set_xticks([])
    axs.set_yticks([])
    cbar.ax.tick_params(labelsize=15)
    plt.savefig(folder_figure+'nm_fig_2C.png',bbox_inches='tight')
    
# nm_fig_2D
load_co_cluster_sub = True
if load_co_cluster_sub:
    co_cluster_sub =  np.loadtxt(folder_data+'co_cluster_sub.csv',delimiter=',')
else:
    co_cluster_sub = cal_co_cluster(co_mat_mean_sub_sorted_cluster,cluster_size_gmm_nm,plot_bool=False)
    np.savetxt(folder_data+'co_cluster_sub.csv', co_cluster_sub, delimiter=',')
_, axs = plt.subplots(1,1,figsize=(8,8))
im = axs.pcolormesh(np.arange(num_type)+1,np.arange(num_type)+1,co_cluster_sub[::-1,::-1],cmap='viridis')
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar=plt.colorbar(im, cax=cax)
axs.set_aspect('equal', 'box')
axs.set_xticks(np.arange(2,num_type+1,2))
axs.set_yticks(np.arange(2,num_type+1,2))
axs.tick_params(axis='x', labelsize=10)
axs.tick_params(axis='y', labelsize=10)
cbar.ax.tick_params(labelsize=10)
savefig(folder_figure+'nm_fig_2D') 

co_cluster_sub_diag = np.zeros(num_type)
for i in range(num_type):
    co_cluster_sub_diag[i] = co_cluster_sub[i,i]
    
# %%calculate the correlation between clusters in original dataset
X_cluster_gmm_nm_dendro_reverse = X_cluster_gmm_nm_dendro[::-1,:]
corr_original, corr_original_full, corr_sub_match = cal_corr_sub(X_cluster_gmm_nm_dendro,X_cluster_gmm_sub,num_sub=num_sub)
corr_original_full_reverse = corr_original_full[::-1,::-1]
corr_original_full_nan = np.copy(corr_original_full)
corr_original_full_nan[corr_original_full==1]=np.nan
corr_original_max = np.nanmax(corr_original_full_nan,axis=0)
corr_original_max_reverse = corr_original_max[::-1]
corr_sub_match_reverse = corr_sub_match[:,::-1]
corr_sub_match_median = np.median(corr_sub_match_reverse,axis=1) 
corr_sub_match_median_cluster = np.median(corr_sub_match_reverse,axis=0) 

# nm_fig_2G
fig,ax = plt.subplots(1,1,figsize=(3,4))
ax.hist(corr_sub_match_median.ravel(),bins=np.linspace(0.95,1,21),align='mid',color='grey',edgecolor='black') 
savefig(folder_figure+'nm_fig_2G')
plt.show()

idx_low_cc = np.where(corr_sub_match_median_cluster<np.nanmax(corr_original))[0]
corr_sub_match_mean_cluster = np.mean(corr_sub_match_reverse,axis=0) 
corr_sub_match_std_cluster = np.std(corr_sub_match_reverse,axis=0) 
corr_sub_match_sem_cluster = corr_sub_match_std_cluster/np.sqrt(num_sub)

#%% calculate the distance between clusters in original dataset
dist_original_full, dist_original_full_nan, dist_sub_match = cal_dist_sub(X_cluster_gmm_nm_dendro,X_cluster_gmm_sub,num_sub=num_sub)
dist_original_min = np.nanmin(dist_original_full_nan,axis=0)
dist_original_min_reverse = dist_original_min[::-1]
dist_original_full_reverse = dist_original_full[::-1,::-1]
dist_sub_match_reverse =  dist_sub_match[:,::-1]
dist_sub_match_median = np.median(dist_sub_match_reverse,axis=1) 
dist_sub_match_median_cluster = np.median(dist_sub_match_reverse,axis=0) 

idx_long_dist = num_type-np.where(dist_sub_match_median_cluster<np.nanmax(dist_original_full_nan))[0][::-1]
dist_sub_match_mean_cluster = np.mean(dist_sub_match_reverse,axis=0) 
dist_sub_match_std_cluster = np.std(dist_sub_match_reverse,axis=0) 
dist_sub_match_sem_cluster = dist_sub_match_std_cluster/np.sqrt(num_sub)

# nm_fig_2F
fig,ax = plt.subplots(1,1,figsize=(6,4))
ax.errorbar(np.array(range(1,num_type+1)),dist_sub_match_mean_cluster,yerr=dist_sub_match_std_cluster,
             color='black',marker='o',capsize=5,ls='none')
ax.plot(np.array(range(1,num_type+1)),dist_original_min_reverse,'o',color='gray')
savefig(folder_figure+'nm_fig_2F')
plt.show()
#%% calculate the jaccard similarity
load_jaccard=False
if load_jaccard:
    jaccard_sub_match = np.loadtxt(folder_path+'data/jaccard_sub_match.csv',delimiter=',')
else:
    jaccard_sub_match = cal_jaccard_sub(labels_gmm_nm_dendro,labels_gmm_sub,idx_sub,num_sub=num_sub)
    np.savetxt(folder_path+'data/jaccard_sub_match.csv', jaccard_sub_match, delimiter=',')
jaccard_sub_match_reverse = jaccard_sub_match[:,::-1]

jaccard_sub_match_mean_cluster = np.mean(jaccard_sub_match_reverse,axis=0) 
jaccard_sub_match_std_cluster = np.std(jaccard_sub_match_reverse,axis=0) 
jaccard_sub_match_sem_cluster = jaccard_sub_match_std_cluster/np.sqrt(num_sub)
# # nm_fig_2F
fig,ax = plt.subplots(1,1,figsize=(6,4))
ax.errorbar(np.array(range(1,num_type+1)),jaccard_sub_match_mean_cluster,yerr=jaccard_sub_match_std_cluster,
             color='black',marker='o',capsize=5,ls='none')
ax.plot(np.array(range(1,num_type+1)),np.ones(num_type)*0.5,'r')
savefig(folder_figure+'nm_fig_2E')
plt.show()


#%% for natural-movie data, plot dendrogram, temporal profiels, 
# and histograms showing how each nm cluster is composed by other-stimuli cluster

fig_params = {'space_sub': 1, # arbituray unit
              'height_sub': 10, # arbituray unit
              'width_sub_hist': 16, # arbituray unit
              'margin': 0, # relative in [0,1], margin is not useful
              'dendro_width': 0.1,
              'zoom': 2.5,
              'dendro_text_pos': -0.2,
              'offset': 0.04
              }

_idx_nan = np.logical_or(np.isnan(rf['azimuth_rf_flash']),np.isnan(rf['elevation_rf_flash']))
rf['rf_size'][_idx_nan] = np.nan 
rf['azimuth_rf_flash'][_idx_nan] = np.nan
rf['elevation_rf_flash'][_idx_nan] = np.nan

df_results = pd.DataFrame(np.column_stack( (rf['rf_size']/np.log(10)*np.log(2), mb['mb_gdsi'], mb['mb_gosi'], dl_lr['dl_hi'], 
                                            dl_lr['lsi'], st['best_size'], st['ssi'], color['color_bgi'],
                                            chirp['fpr'], chirp['freq_si'], chirp['amp_si']) ),
                   columns=['rf_size', 'gdsi', 'gosi', 'dl_hi', 'lsi', 'best_size', 'ssi','bgi',
                            'fpr','freq_si','amp_si'])
df_results['rf_azimuth'] = rf['azimuth_rf_flash']
df_results['rf_elevation'] = rf['elevation_rf_flash']
df_results['rf_size_log10'] = np.log10(df_results['rf_size'])
df_results['st_cv'] = st['st_cv']
df_results['msi'] = mb['msi']
df_results['on_off_si'] = rf['rf_on_off_si']
df_results['depth'] = neuron['soma_depth']
df_results['off_resp_amp'] = chirp['off_resp_amp']
df_results['on_resp_amp'] = chirp['on_resp_amp']
df_results['after_freq_mod_resp_amp'] = chirp['after_freq_mod_resp_amp']
df_results['after_amp_mod_resp_amp'] = chirp['after_amp_mod_resp_amp']
df_results['mb_resp_amp'] = mb['mb_resp_amp']
df_results['cluster_label_nm_num'] = labels_gmm_nm
df_results['cluster_label_nm_dendro_num'] = labels_gmm_nm_dendro
df_results['cluster_label_dendro_num'] = labels_gmm_dendro


df_results['depth'] = neuron['soma_depth']
df_results['image_id'] = neuron['image_id']
df_results['mouse_id'] = neuron['mouse_id']
df_results['soma_pos_x'] = neuron['soma_pos'][:,0]
df_results['soma_pos_y'] = neuron['soma_pos'][:,1]


fig_params_nm = fig_params.copy()
fig_params_nm['dendro_width'] = 0.1
fig_params_nm['dendro_text_pos'] = -0.2
fig_params_nm['zoom'] = 2.5
fig_params_nm['offset'] = 0.04
temporal_list_nm = [('nm_temporal',X_cluster_gmm_nm,X_cluster_gmm_nm_error)]
hist_list_nm = [] #['FAC'] #functional clusters by artifical stimuli
temporal_zooms_nm = [0.3]
temporal_zooms = [0.6, 0.6, 0.6, 0.3, 0.3, 0.3]
columns_label =['msi','gdsi','gosi','dl_hi','lsi','on_off_si','best_size','ssi','bgi','rf_size']
columns_name = ['MSI','DSI', 'OSI', 'HI', 'LSI', 'CSI','BSS', 'SSI','BGI','RFS']
bins_num = 20+1
bins_rf_size = np.linspace(0,round_up(np.max(df_results['rf_size']),-1),bins_num)
bins_rf_size_log10 = np.logspace(1,round_up(np.log10(np.max(df_results['rf_size'])),1),bins_num)
bins_msi = np.linspace(-1, 1, bins_num)
bins_gdsi = np.linspace(round_down(np.min(df_results['gdsi']),1), round_up(np.max(df_results['gdsi']),1), bins_num)
bins_gosi = np.linspace(round_down(np.min(df_results['gosi']),1), round_up(np.max(df_results['gosi']),1), bins_num)
bins_dl_hi = np.linspace(-1, 1, bins_num)
bins_lsi = np.linspace(-1, 1, bins_num)
bins_csi = np.linspace(-1, 1, bins_num)
bins_st = np.linspace(round_down(np.min(df_results['st_cv']),1), round_up(np.max(df_results['st_cv']),1), bins_num)
bins_best_size = np.linspace(1,6,6)
bins_ssi = np.linspace(round_down(np.min(df_results['ssi']),1), round_up(np.max(df_results['ssi']),1), bins_num)
bins_bgi = np.linspace(-1, 1, bins_num)
bins_list = [bins_msi, bins_gdsi, bins_gosi,bins_dl_hi, bins_lsi, bins_csi, bins_best_size, bins_ssi, bins_bgi,bins_rf_size_log10] 

box_flag = ['best_size']

hist_list = columns_label.copy()
#%% plot dendrogram based on clusters from nm data
hist_list_nm = ['FAC']
file_save_cluster_nm_dendro = []
freq_all = figure_cluster_nm_temporal_hist(df_results, X_cluster_gmm_nm, X_cluster_gmm_nm, cluster_size_gmm_nm, 
                                  temporal_list_nm, hist_list_nm, temporal_zooms_nm,
                                  file_save_cluster_nm_dendro, fig_params_nm, sampling_rate=5)

freq_fac_nm_mat = np.asarray(freq_all)
# nm_fig_4C
fig,ax = plt.subplots(1,1,figsize=(6,4))
im = ax.imshow(freq_fac_nm_mat[::-1,:], cmap='viridis',aspect='auto',extent=[0.5,24.5,num_type+0.5,0.5])
ax.set_xticks(np.arange(1,24,2))
ax.set_yticks(np.arange(1,num_type+1,2)) 

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
# ax.set_aspect('equal', 'box')
fig.colorbar(im, cax=cax)
savefig(folder_figure+'nm_fig_5C_left')   
plt.show()


## plot the cumulative curve for each cluster
freq_fac_nm_cum = np.zeros_like(freq_fac_nm_mat)
for i in range(num_type):
    freq_fac_nm_cum[i,:] = np.cumsum(np.sort(freq_fac_nm_mat[i,:])[::-1])
    
fig,ax = plt.subplots(1,1,figsize=(6,4))
cmap = matplotlib.colormaps.get_cmap('viridis').copy()
n_step = int(256/num_type)
label_colors = cmap.colors[::n_step]
for i in range(num_type):
    ax.plot(np.arange(1,num_type_art+1),freq_fac_nm_cum[num_type-i-1,:],'-o',color=label_colors[i])

n_rand = 1000
freq_fac_rand = np.zeros((num_type_art,n_rand))
_freq_rand = np.cumsum(np.random.rand(num_type_art,n_rand),axis=0)
for i in range(n_rand):
    freq_fac_rand[:,i] = _freq_rand[:,i]/_freq_rand[:,i].max()*100

freq_fac_rand_mean = np.mean(freq_fac_rand,axis=1)
freq_fac_rand_sd = np.std(freq_fac_rand,axis=1)
ax.plot(np.arange(1,num_type_art+1),freq_fac_rand_mean,'r')
ax.set_xlim([0.5,num_type_art+0.5])
ax.set_ylim([0,105])

_legend = ['C'+str(x) for x in np.arange(1,17)]
_legend.append('rand')

ax.legend(_legend,loc='best',ncol=3,labelspacing=0.3)

savefig(folder_figure+'nm_fig_5D')   
plt.show()


freq_fac_nm_group = np.zeros((num_type,2))
for i in range(num_type):
    freq_fac_nm_group[i,0] = np.sum(freq_fac_nm_mat[i,:10]) 
    freq_fac_nm_group[i,1] = np.sum(freq_fac_nm_mat[i,10:]) 
    
fig,ax = plt.subplots(1,1,figsize=(2,4))
im = ax.imshow(freq_fac_nm_group[::-1,:], cmap='viridis',aspect='auto',extent=[0.5,2.5,num_type+0.5,0.5])
ax.set_xticks(np.arange(1,3))
ax.set_yticks(np.arange(1,num_type+1,2)) 

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="10%", pad=0.1)
# ax.set_aspect('equal', 'box')
fig.colorbar(im, cax=cax)


savefig(folder_figure+'nm_fig_5C_right')   
plt.show()



#%% nature movies: temporal plot with error, nm_fig_3A
file_save_cluster_dendro_temporal= folder_figure + 'nm_fig_3A'
data_box_all, data_box_clusters,_ = figure_cluster_temporal_box(df_results, 'cluster_label_nm_dendro_num', X_cluster_gmm_nm, cluster_size_gmm_nm, 
                                 temporal_list_nm, [], temporal_zooms_nm,
                                 columns_name.copy(), bins_list,
                                 file_save_cluster_dendro_temporal, fig_params, 
                                 fac=True, genetic_comp=False, sampling_rate=5, violin=False, norm_flag=False, error=True)


#%% nature movies: normalized temporal plot, nm_fig_3B
file_save_cluster_dendro_temporal= folder_figure + 'nm_fig_3B'
data_box_all, data_box_clusters,_ = figure_cluster_temporal_box(df_results, 'cluster_label_nm_dendro_num', X_cluster_gmm_nm, cluster_size_gmm_nm, 
                                 temporal_list_nm, [], temporal_zooms_nm,
                                 columns_name.copy(), bins_list,
                                 file_save_cluster_dendro_temporal, fig_params, 
                                 fac=True, genetic_comp=False, sampling_rate=5, violin=False, norm_flag=True, error=False)


#%% violin plot for natural-movies defined clusters.
file_save_cluster_dendro_violin= folder_figure + 'nm_fig_5A'
data_box_all, data_box_clusters,_ = figure_cluster_temporal_box(df_results, 'cluster_label_nm_dendro_num', X_cluster_gmm_nm, cluster_size_gmm_nm, 
                                 [], columns_label, temporal_zooms_nm,
                                 columns_name.copy(), bins_list,
                                 file_save_cluster_dendro_violin, fig_params, 
                                 fac=True, genetic_comp=False, sampling_rate=5, violin=True, norm_flag=False, error=False,box_flag=box_flag)
#%%plot cell types versus functional properties for natural movies *** nm_fig_4B
si_list = ['mb_resp_amp',  'gdsi', 'gosi', 'dl_hi', 'lsi', 
           'msi', 'on_off_si','fpr','freq_si', 'after_freq_mod_resp_amp', 
           'after_amp_mod_resp_amp', 'best_size', 'ssi', 'bgi','rf_size_log10']
ref_list = [0, 0.15, 0.15, 0,  0,   
            0, 0, 0.5,  0,   0,       
            0, 3, 0, 0, 1.953]
si_n = len(si_list)
p_thr_fun = 0.05
si_cluster, p_cluster, corr_fp, corr_p_fp = si_corr(df_results,si_list,labels_gmm_nm_dendro,ref_list=ref_list)
si_cluster_reverse = si_cluster[:,::-1].T

si_cluster_reverse_norm = norm_x(si_cluster_reverse.T,formula=1).T 


p_cluster_reverse = p_cluster[:,::-1].T
si_cluster_reverse_sig = si_cluster_reverse*(p_cluster_reverse<p_thr_fun)
si_cluster_reverse_sig_norm = norm_x(si_cluster_reverse_sig.T,formula=1).T 
si_cluster_reverse_sig_norm[p_cluster_reverse>=p_thr_fun] = np.nan

fig,ax = plt.subplots(1,1,figsize=(16,4))
if compareVersion(matplotlib.__version__,'3.7') == 1:
    cmap = matplotlib.colormaps.get_cmap("bwr").copy() #this is for newer versions
else:
    cmap = plt.cm.get_cmap("bwr").copy()
cmap.set_bad(color='grey',alpha=1)
norm = colors.TwoSlopeNorm(vcenter=0)
si_cluster_reverse_sig_norm_masked = np.ma.masked_where(np.isnan(si_cluster_reverse_sig_norm),si_cluster_reverse_sig_norm)
im=ax.imshow(si_cluster_reverse_sig_norm_masked,cmap=cmap,norm=norm,interpolation='nearest',origin='upper',aspect='auto',extent=[0.5,15.5,16.5,0.5])
ax.set_xticks(np.arange(1,16))
ax.set_yticks(np.arange(1,num_type+1,2)) 

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
# ax.set_aspect('equal', 'box')
plt.colorbar(im, cax=cax, ticks=[-1,0,1])

if save_fig:
    savefig(folder_figure+'nm_fig_5B')  
plt.show()
 

# %% calculate the imaging area for each experiment ***
# calculate the distance between two types and their p values with bootstrap
image_outline = []
image_area = np.zeros(num_image)
image_area_ftypes = np.zeros((num_image,num_type))
image_area_ratio_ftypes = np.zeros((num_image,num_type))
dist_mean_all_types = np.ones((num_image,num_type,num_type))*np.nan
p_all_types = np.ones((num_image,num_type,num_type))*np.nan
n_all_types = np.zeros((num_image,num_type))
n_norm_all_types = np.zeros((num_image,num_type))
n_btsp = 1000
n_neuron_type_thr = 5
for i in range(num_image):
    pts = neuron['soma_pos'][neuron['image_id']==i,:]
    for j in range(num_type):
        pts_j = neuron['soma_pos'][np.logical_and(neuron['image_id']==i,labels_gmm_dendro==j),:]
        n_all_types[i,j] = pts_j.shape[0]
        n_norm_all_types[i,j] = n_all_types[i,j]/pts.shape[0]
n_all_types_reverse = n_all_types[:,::-1]
n_all_types_reverse_T = n_all_types_reverse.T
n_norm_all_types_reverse = n_norm_all_types[:,::-1]
n_norm_all_types_reverse_T = n_norm_all_types_reverse.T
for i in range(num_image):
    pts = neuron['soma_pos'][neuron['image_id']==i,:]
    hull = ConvexHull(pts)
    pts_sorted = SortVertice(pts[hull.vertices,:])
    image_outline.append(np.insert(pts_sorted,pts_sorted.shape[0],pts_sorted[0,:],axis=0))
    image_area[i] = PolyArea(image_outline[-1])

#%%    
# image_area_ratio_ftypes[image_area_ratio_ftypes==0]=np.nan  
# image_area_ratio_ftypes_0 = image_area_ratio_ftypes.copy()
# image_area_ratio_ftypes_0[np.isnan(image_area_ratio_ftypes)]=0 
# image_area_ratio_ftypes_reverse = image_area_ratio_ftypes[:,::-1]
# dist_mean_all_types_reverse = dist_mean_all_types[:,::-1,::-1]  

# p_all_types_max = np.ones((num_type,num_type))
# dist_mean_all_types_mean = np.ones((num_type,num_type))*np.nan
# dist_mean_all_types_std = np.ones((num_type,num_type))*np.nan
# for i in range(num_type):
#     for j in range(i,num_type):
#         idx_select_p = np.where(np.logical_and(np.sum(n_all_types,axis=1)>50,
#                                                np.min(n_all_types[:,[i,j]],axis=1)>=10))[0]
#         if idx_select_p.size>0:
#             p_all_types_max[i,j] = np.nanmax(p_all_types[idx_select_p,i,j])
#             p_all_types_max[j,i] = p_all_types_max[i,j]
#             dist_mean_all_types_mean[i,j] = np.nanmean(dist_mean_all_types[idx_select_p,i,j], axis=0)
#             dist_mean_all_types_mean[j,i] = dist_mean_all_types_mean[i,j]
#             dist_mean_all_types_std[i,j] = np.nanstd(dist_mean_all_types[idx_select_p,i,j], axis=0)
#             dist_mean_all_types_std[j,i] = dist_mean_all_types_std[i,j]
# p_all_types_max_reverse = p_all_types_max[::-1,::-1]
# p_all_types_max_reverse_005 = np.zeros_like(p_all_types_max_reverse)
# p_all_types_max_reverse_005[p_all_types_max_reverse<0.05]=1
# p_all_types_max_reverse_005_sum = np.sum(p_all_types_max_reverse_005,axis=0)


# x_p_all_types_max_reverse_005,y_p_all_types_max_reverse_005 = np.where(p_all_types_max_reverse_005==1)
# x_y_p_all_types_max_reverse_005 = np.stack((x_p_all_types_max_reverse_005,y_p_all_types_max_reverse_005),axis=1)

# dist_mean_all_types_mean_reverse = dist_mean_all_types_mean[::-1,::-1]
# dist_mean_all_types_std_reverse = dist_mean_all_types_std[::-1,::-1]
# dist_mean_all_types_mean_reverse_mean = np.nanmean(dist_mean_all_types_mean_reverse,axis=0)

# dist_mean_all_types_mean_reverse_p = dist_mean_all_types_mean_reverse*p_all_types_max_reverse_005
# dist_mean_all_types_mean_reverse_p[np.isnan(dist_mean_all_types_mean_reverse_p)] = 0
# num_separated = np.sum(dist_mean_all_types_mean_reverse_p!=0,axis=0)


#%% analyze the cell types within 50um from a neuron ***
# only select types which has >=5 neurons in a image
neurons_per_image = np.zeros(num_image)
for i in range(num_image):
    neurons_per_image[i] = np.sum(neuron['image_id']==i)
image = {'image_area':image_area,
         'neurons_per_image':neurons_per_image}

#%% plot the density recovery profile in anatomical space ***
bin_size_anatomy = 10 # set bin size to 10um; change 20 run this again
dist_max_anatomy = 1500 #set the largest distance is 1mm
bins_max_anatomy = np.arange(0, dist_max_anatomy+bin_size_anatomy, bin_size_anatomy)
bin_num_anatomy = bins_max_anatomy.size-1
density_all_types_anatomy_mean = np.zeros((bin_num_anatomy, num_type))
density_all_types_anatomy_sd = np.zeros((bin_num_anatomy, num_type))
density_all_types_anatomy_sem = np.zeros((bin_num_anatomy, num_type))
# calculate the image window ffor all imaging planes
w_density = np.zeros(num_image)
h_density = np.zeros(num_image)
idx_select_50 = np.where(neurons_per_image>50)[0]
idx_del_50 = np.where(neurons_per_image<=50)[0]
for i in range(num_image):
    w_density[i] = neuron['soma_pos'][neuron['image_id']==i,0].max() - neuron['soma_pos'][neuron['image_id']==i,0].min()
    h_density[i] = neuron['soma_pos'][neuron['image_id']==i,1].max() - neuron['soma_pos'][neuron['image_id']==i,1].min()

density_all_type_anatomy = np.ones((num_type, num_image, bin_num_anatomy))*np.nan
for i in range(num_type):
    density_one_type_anatomy = np.ones((bin_num_anatomy, num_image))*np.nan
    for j in range(num_image):
        image_ij = np.logical_and(neuron['image_id']==j, df_results['cluster_label_nm_dendro_num']==i)
        if np.sum(image_ij)>=5: 
            _,density_anatomy_temp,_ = Analyze_Dist_Poly(neuron['soma_pos'][image_ij,:],bin_size_anatomy,image_outline[j])
            density_one_type_anatomy[:density_anatomy_temp.shape[0],j] = density_anatomy_temp
    density_one_type_anatomy[:,idx_del_50]=np.nan # assign nan to images that have <=50 neurons
    density_all_type_anatomy[i,:,:] = density_one_type_anatomy.T
    density_one_type_anatomy_mean = np.nanmean(density_one_type_anatomy,axis=1)
    density_one_type_anatomy_sd = np.nanstd(density_one_type_anatomy,axis=1)
    density_one_type_anatomy_sem = scipy.stats.sem(density_one_type_anatomy, axis=1, ddof=1, nan_policy='omit')
    density_all_types_anatomy_mean[:,i] = density_one_type_anatomy_mean
    density_all_types_anatomy_sd[:,i] = density_one_type_anatomy_sd
    density_all_types_anatomy_sem[:,i] = density_one_type_anatomy_sem
         
density_all_types_mean_anatomy = np.mean(density_all_types_anatomy_mean,axis=1)

neurons_per_cluster_density = np.zeros(num_type)
for i in range(num_type):
    neurons_per_cluster_density[i] = np.sum(df_results['cluster_label_nm_dendro_num']==i)
    
# cluster_label_5_example_all_density = neurons_per_cluster_density.argsort()[-5:][::-1]
# cluster_label_5_example_all_density_plot = num_type - cluster_label_5_example_all_density

# cluster_label_5_example_all_density =  num_type - cluster_label_5_example_all_density_plot
# density_all_types_anatomy_example_all_mean_sem = np.zeros((bin_num_anatomy,cluster_label_5_example_all_density.size*2))
# density_all_types_anatomy_example_all_mean_sem[:,::2] = density_all_types_anatomy_mean[:,cluster_label_5_example_all_density]
# density_all_types_anatomy_example_all_mean_sem[:,1::2] = density_all_types_anatomy_sem[:,cluster_label_5_example_all_density]

# # Figure 4B
# n_drp_example = 5
# fig, ax = plt.subplots(5,1,sharex=True,figsize = (10,8))
# for i in range(n_drp_example):
#     ax[i].bar(bins_max_anatomy[1:26],density_all_types_anatomy_example_all_mean_sem[:25,i*2]*1e6,
#         yerr=density_all_types_anatomy_example_all_mean_sem[:25,i*2+1]*1e6,width=10,color='grey',edgecolor='black',capsize=5)
#     ax[i].text(0,100,str(cluster_label_5_example_all_density_plot[i]),color='red')
# if save_fig:
#     plt.savefig(folder_figure+'figure_4B.png',bbox_inches='tight')
# plt.show()

density_all_types_anatomy_all_mean_sem = np.zeros((bin_num_anatomy,num_type*2))
density_all_types_anatomy_all_mean_sem[:,::2] = density_all_types_anatomy_mean
density_all_types_anatomy_all_mean_sem[:,1::2] = density_all_types_anatomy_sem
density_all_types_anatomy_mean_T = density_all_types_anatomy_mean[:26,:].T
density_all_types_anatomy_mean_T_50um_bin = np.zeros((num_type,5))
for i in range(5):
    density_all_types_anatomy_mean_T_50um_bin[:,i] = np.nanmean(density_all_types_anatomy_mean_T[:,i*5:(i+1)*5],axis=1)
density_all_types_anatomy_mean_T_norm = np.zeros_like(density_all_types_anatomy_mean_T_50um_bin)
for i in range(num_type):
    density_all_types_anatomy_mean_T_norm[i,:] = density_all_types_anatomy_mean_T_50um_bin[i,:]/np.nanmax(density_all_types_anatomy_mean_T_50um_bin[i,:])


drp_peak_idx = np.nanargmax(density_all_types_anatomy_mean_T_norm,axis=1)
drp_peak_half = 0.51
drp_peak_half_idx = np.ones_like(drp_peak_idx)*np.nan
for i in range(num_type):
    if drp_peak_idx[i]<=2:
        if np.where(density_all_types_anatomy_mean_T_norm[i,:]<drp_peak_half)[0].size>0:
            drp_peak_half_idx[i] = np.where(density_all_types_anatomy_mean_T_norm[i,:]<drp_peak_half)[0][0]
        else:
            drp_peak_half_idx[i] = 5
        
#nm_fig_4F
fig, ax = plt.subplots(1,1,figsize = (4,4))
im=ax.imshow(density_all_types_anatomy_mean_T_norm,aspect='auto',origin='lower',extent=[0,250,0.5,16.5])
ax.plot((drp_peak_half_idx)*50,np.arange(1,num_type+1),'r*')  
ax.set_xlim([0,255]) 
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
# ax.set_aspect('equal', 'box')
plt.colorbar(im, cax=cax)
if save_fig:
    savefig(folder_figure+'nm_fig_4F')
plt.show()
#%% Scatter plot for images with types color-coded ***

cluster_label_5_example_density_plot = np.arange(num_type)+1
# nm_fig_4E
color_scatter = ['red','magenta','green','blue','cyan']
image_example_density = 30 #vgat2
soma_position_example_density,soma_position_5_example_density,cluster_label_5_example_density_plot=scatter_clusters_nm(image_example_density,df_results,neuron,cluster_label_5_example_density_plot)
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(soma_position_example_density[:,0],soma_position_example_density[:,1],'^',color='grey')
cluster_selected = num_type - np.array([3,4,6,8,9])
for i in cluster_selected:
    ax.plot(soma_position_5_example_density[:,i*2],soma_position_5_example_density[:,i*2+1],'^')
ax.set_aspect('equal') 
ax.legend(['C3','C4','C6','C8','C9'],loc='lower right')
if save_fig:
    savefig(folder_figure+'nm_fig_4E')       
plt.show()


#%% plot the percentage of neurons with flashing-mapped RFs for each cluster ***
rf_perc_nm = np.ones(num_type)*np.nan
rf_count_nm = np.ones(num_type)*np.nan
rf_size_nm = np.ones((num_type,2))*np.nan
for i in range(num_type):    
    temp = rf['rf_size'][df_results['cluster_label_nm_dendro_num']==num_type-i-1]/np.log(10)*np.log(2)
    rf_count_nm[i] = (temp.size - np.sum(np.isnan(temp))) 
    rf_perc_nm[i] = rf_count_nm[i] / temp.size*100
    temp_1 = temp[np.logical_not(np.isnan(temp))]
    rf_size_nm[i,:] = [np.mean(temp_1),np.std(temp_1)/np.sqrt(len(temp_1))]

# Figure 10C, related to Figure 3
fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.bar(np.asarray(range(num_type))+1,rf_perc_nm,color='grey',edgecolor='black',width=1)   
ax.hlines(y=80,xmin=0.3,xmax=num_type+0.7,color='b',linestyles='--')
plt.xlim([0.3,num_type+0.7])
if save_fig:
    savefig(folder_figure+'nm_fig_4A')   
plt.show()


fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.bar(np.asarray(range(num_type))+1,rf_size_nm[:,0],yerr=rf_size_nm[:,1],color='grey',edgecolor='black',width=1,capsize=5)    
# ax.plot(np.asarray(range(num_type))+1,np.ones(num_type)*50,'r')
plt.xlim([0.3,num_type+0.7])
if save_fig:
    savefig(folder_figure+'nm_fig_4B')   
plt.show()



visual_area = np.zeros(num_type)
visual_outline = []
for i in range(num_type):
    _temp = np.logical_and(df_results['cluster_label_nm_dendro_num']==i,~np.isnan(df_results['rf_elevation']))
    pts = np.array([df_results['rf_azimuth'][_temp],df_results['rf_elevation'][_temp]]).T
    # pts = neuron['soma_pos'][neuron['image_id']==i,:]
    hull = ConvexHull(pts)
    pts_sorted = SortVertice(pts[hull.vertices,:])
    visual_outline.append(np.insert(pts_sorted,pts_sorted.shape[0],pts_sorted[0,:],axis=0))
    visual_area[i] = PolyArea(visual_outline[-1])

idx_sorted = np.argsort(visual_area)
num_type_arr = np.arange(num_type)+1
num_type_rev_arr = num_type_arr[::-1]
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.bar(num_type_arr,visual_area[idx_sorted],width=1,color='grey',edgecolor='black')
ax.set_xticks(num_type_arr, labels=num_type_rev_arr[idx_sorted])
ax.set_xlim([1-0.7,num_type+0.7])
ax.set_ylim([0,3680])
ax.hlines(y=900,xmin=1-0.7,xmax=num_type+0.7,color='b',linestyles='--')
ax.vlines(x=8.5,ymin=0,ymax=3680,color='r',linestyles='--')
savefig(folder_figure+'nm_fig_4D')   
plt.show()

#%% apply natural movies clusters to rf position ***

fig_zoom = 2.5
hwr = 0.618 #height to width ratio
plt.figure(figsize=(fig_width*fig_zoom,fig_width*fig_zoom*hwr))
df_rf = df_results[['rf_azimuth', 'rf_elevation', 'rf_size', 'cluster_label_nm_dendro_num']].copy()
fig_params_rf_scatter = dict.copy(fig_params)
fig_params_rf_scatter['space'] = 0.02
fig_params_rf_scatter['zoom'] = 2.5
fig_params_rf_scatter['margin'] = 0
fig_params_rf_scatter['hwr'] = 0.69
fig_params_rf_scatter['fig_col'] = 4


neuron_count = np.arange(num_type)
for i in range(num_type):
    azi = df_results['rf_azimuth'][df_results['cluster_label_nm_dendro_num']==i]
    neuron_count[i] = azi.size

rf_pos_cluster = num_type-np.array([3,8,9,13,14,15])
fig, ax = plt.subplots(1,1,figsize=(6,4))
neuron_count_1 = 0
neuron_count_no_nan_1 = 0 
for i in range(rf_pos_cluster.size):
    azi = df_results['rf_azimuth'][df_results['cluster_label_nm_dendro_num']==rf_pos_cluster[i]]
    ele = df_results['rf_elevation'][df_results['cluster_label_nm_dendro_num']==rf_pos_cluster[i]]
    ax.scatter(azi, ele, s=2)
    neuron_count_1 = neuron_count_1+azi.size
    neuron_count_no_nan_1 = neuron_count_no_nan_1 + sum(~np.isnan(azi))

xlim_min = round_down(np.min(df_rf['rf_azimuth']),-1)
xlim_max =  round_up(np.max(df_rf['rf_azimuth']),-1)
xlim_range = xlim_max - xlim_min 
xlim = [xlim_min, xlim_max]
ylim_min = round_down(np.min(df_rf['rf_elevation']),-1)
ylim_max = round_up(np.max(df_rf['rf_elevation']),-1)
ylim = [ylim_min, ylim_max]
ylim_range = ylim_max - ylim_min     
ax.plot([xlim_min,xlim_max],[np.nanmedian(df_rf['rf_elevation']),np.nanmedian(df_rf['rf_elevation'])],color='red',linewidth=0.2)
ax.plot([np.nanmedian(df_rf['rf_azimuth']),np.nanmedian(df_rf['rf_azimuth'])],[ylim_min,ylim_max],color='red',linewidth=0.2)
ax.set_aspect('equal')
ax.set_xlim(xlim) 
ax.set_ylim(ylim)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.legend(['C3','C8','C9','C13','C14','C15'],loc='upper left')
ax.grid()
savefig(folder_figure+'nm_fig_4C_1')   
plt.show()


rf_pos_cluster = num_type-np.array([1,4,6,7,10,16])
neuron_count_2 = 0
neuron_count_no_nan_2 = 0 
fig, ax = plt.subplots(1,1,figsize=(6,4))
for i in range(rf_pos_cluster.size):
    azi = df_results['rf_azimuth'][df_results['cluster_label_nm_dendro_num']==rf_pos_cluster[i]]
    ele = df_results['rf_elevation'][df_results['cluster_label_nm_dendro_num']==rf_pos_cluster[i]]
    ax.scatter(azi, ele, s=2)
    neuron_count_2 = neuron_count_2+azi.size
    neuron_count_no_nan_2 = neuron_count_no_nan_2 + sum(~np.isnan(azi))
   
ax.plot([xlim_min,xlim_max],[np.nanmedian(df_rf['rf_elevation']),np.nanmedian(df_rf['rf_elevation'])],color='red',linewidth=0.2)
ax.plot([np.nanmedian(df_rf['rf_azimuth']),np.nanmedian(df_rf['rf_azimuth'])],[ylim_min,ylim_max],color='red',linewidth=0.2)
ax.set_aspect('equal')
ax.set_xlim(xlim) 
ax.set_ylim(ylim)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.legend(['C1','C4','C6','C7','C10','C16'],loc='upper left')
labels = [item.get_text() for item in ax.get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax.set_yticklabels(empty_string_labels)
ax.grid()
savefig(folder_figure+'nm_fig_4C_2')   
plt.show()


rf_pos_cluster = num_type-np.array([2,5,11,12])
neuron_count_3 = 0
neuron_count_no_nan_3 = 0 
fig, ax = plt.subplots(1,1,figsize=(6,4))
for i in range(rf_pos_cluster.size):
    azi = df_results['rf_azimuth'][df_results['cluster_label_nm_dendro_num']==rf_pos_cluster[i]]
    ele = df_results['rf_elevation'][df_results['cluster_label_nm_dendro_num']==rf_pos_cluster[i]]
    ax.scatter(azi, ele, s=2)
    neuron_count_3 = neuron_count_3+azi.size
    neuron_count_no_nan_3 = neuron_count_no_nan_3 + sum(~np.isnan(azi))
   
ax.plot([xlim_min,xlim_max],[np.nanmedian(df_rf['rf_elevation']),np.nanmedian(df_rf['rf_elevation'])],color='red',linewidth=0.2)
ax.plot([np.nanmedian(df_rf['rf_azimuth']),np.nanmedian(df_rf['rf_azimuth'])],[ylim_min,ylim_max],color='red',linewidth=0.2)
ax.set_aspect('equal')
ax.set_xlim(xlim) 
ax.set_ylim(ylim)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.legend(['C2','C5','C11','C12'],loc='upper left')
labels = [item.get_text() for item in ax.get_xticklabels()]
empty_string_labels = ['']*len(labels)
ax.set_yticklabels(empty_string_labels)
ax.grid()
savefig(folder_figure+'nm_fig_4C_3')
plt.show()


#%%

genetic_label = neuron['genetic_id']
genetic_name = ['All types','Vglut2+','Vgat+','Tac1+','Rorb+', 'Ntsr1+']
genetic_num = len(genetic_name)
genetic_label_text = [str(x) for x in genetic_label]
genetic_label_text = [label.replace('0.0', genetic_name[0]) for label in genetic_label_text]
genetic_label_text = [label.replace('1.0', genetic_name[1]) for label in genetic_label_text]
genetic_label_text = [label.replace('2.0', genetic_name[2]) for label in genetic_label_text]
genetic_label_text = [label.replace('3.0', genetic_name[3]) for label in genetic_label_text]
genetic_label_text = [label.replace('4.0', genetic_name[4]) for label in genetic_label_text]
genetic_label_text = [label.replace('5.0', genetic_name[5]) for label in genetic_label_text]

df_results['genetic_label'] = genetic_label_text
df_results['genetic_label_num'] = genetic_label

X_cluster_genetic,_,_,_ = plot_cluster(X,df_results['genetic_label_num'],[])


#%% nm_fig_6A relationship to genetic label and depth ***
file_save_temporal_depth = folder_figure + 'nm_fig_6B'
fig_params['height_sub'] = 25
X_cluster_gmm_nm_genetic,X_cluster_gmm_nm_genetic_error,_,_ = plot_cluster(nm_temporal_norm,df_results['genetic_label_num'],[])
temporal_list_nm_genetic = [('nm_temporal',X_cluster_gmm_nm_genetic[1:,:],X_cluster_gmm_nm_genetic_error[1:,:])]
figure_cluster_genetic_temporal_box(df_results, 'genetic_label',  X_cluster_genetic[1:,:],
                                 temporal_list_nm_genetic, [], temporal_zooms_nm,
                                 columns_name.copy(), bins_list,
                                 file_save_temporal_depth, fig_params, fac=True, sampling_rate=5)

freq_nm_genetic_ls = []
cluster_num_nm = np.unique(df_results['cluster_label_nm_dendro_num']).size
bins_nm = np.linspace(1,cluster_num_nm+1,cluster_num_nm+1)
for i in range(genetic_num):
    data_nm_i = df_results['cluster_label_nm_dendro_num'][df_results['genetic_label']==genetic_name[i]]
    hist_nm_i, edges_nm_i = np.histogram(cluster_num_nm-data_nm_i, bins_nm)
    freq_nm_genetic_ls.append(hist_nm_i/float(hist_nm_i.sum())*100)
freq_nm_genetic = np.asarray(freq_nm_genetic_ls)

fig,ax = plt.subplots(1,1,figsize=(6,4))
im = ax.imshow(freq_nm_genetic[1:], cmap='viridis',aspect='auto',origin='upper',extent=[0.5,num_type+0.5,5.5,0.5])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax,ticks=np.arange(0,50,10))
# axs.set_aspect('equal', 'box')
# axs.set_xticks([])
ax.set_yticks([])
# cbar.ax.tick_params(labelsize=15)
if save_fig:
    savefig(folder_figure+'nm_fig_6D')   
plt.show()


#%%
soma_depth_groups = np.zeros_like(neuron['soma_depth'])
soma_depth_groups[np.logical_and(neuron['soma_depth']>0.1,neuron['soma_depth']<=0.2)] = 1
soma_depth_groups[neuron['soma_depth']>0.2] = 2
df_results['depth_label_num'] = soma_depth_groups
depth_name = ['D1', 'D2', 'D3']
depth_num = len(depth_name)
soma_depth_label = []
for i in range(num_neuron):
    soma_depth_label.append(depth_name[int(soma_depth_groups[i])])
df_results['depth_label'] = soma_depth_label



X_temporal_cluster_nm_depth,X_temporal_cluster_nm_depth_error,_,_ = plot_cluster(nm_temporal_norm,df_results['depth_label_num'],[])
temporal_list_nm_depth= [('nm_temporal',X_temporal_cluster_nm_depth,X_temporal_cluster_nm_depth_error)]
X_cluster_depth,_,_,_ = plot_cluster(X,df_results['depth_label_num'],[])

## nm_fig_6B: temporal across the depth
file_save_temporal_depth = folder_figure + 'nm_fig_6A'
figure_cluster_genetic_temporal_box(df_results, 'depth_label',  X_cluster_depth,
                                 temporal_list_nm_depth, [], temporal_zooms_nm,
                                 columns_name.copy(), bins_list,
                                 file_save_temporal_depth, fig_params, fac=True, sampling_rate=5)



freq_nm_depth_ls = []
cluster_num_nm = np.unique(df_results['cluster_label_nm_dendro_num']).size
bins_nm = np.linspace(1,cluster_num_nm+1,cluster_num_nm+1)
for i in range(depth_num):
    data_nm_i = df_results['cluster_label_nm_dendro_num'][df_results['depth_label']==depth_name[i]]
    hist_nm_i, edges_nm_i = np.histogram(cluster_num_nm-data_nm_i, bins_nm)
    freq_nm_depth_ls.append(hist_nm_i/float(hist_nm_i.sum())*100)
freq_nm_depth = np.asarray(freq_nm_depth_ls)

fig,ax = plt.subplots(1,1,figsize=(6.6,4))
im = ax.imshow(freq_nm_depth, cmap='viridis',aspect='auto',origin='upper',extent=[0.5,num_type+0.5,3.5,0.5])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax,ticks=np.arange(0,100,10))
# axs.set_aspect('equal', 'box')
# axs.set_xticks([])
ax.set_yticks([])
# cbar.ax.tick_params(labelsize=15)
if save_fig:
    savefig(folder_figure+'nm_fig_6C')   
plt.show()


#%% plot excample for each cluster from all images, pick the neuron with the highest corr with the mean *****

snr_stack = np.loadtxt(folder_data+'snr_stack.csv',delimiter=',')
snr_stack_mean = np.mean(snr_stack,axis=1)

cell_corr = np.zeros(num_neuron)
for i in range(num_neuron):
    cell_corr[i],_ = scipy.stats.pearsonr(X[i,:],X_cluster_gmm_dendro[int(df_results['cluster_label_nm_dendro_num'][i]),:])

example_all_rois_list_0 = []
snr_thr_example = 0.25
for i in range(num_type):
    idx_temp = np.where(df_results['cluster_label_nm_dendro_num']==i)[0]
    idx_temp_1 = np.argsort(cell_corr[idx_temp])[::-1] 
    snr_sorted_temp = snr_stack_mean[idx_temp[idx_temp_1]]
    idx_cand = idx_temp[idx_temp_1[snr_sorted_temp>snr_thr_example]]
    if idx_cand.size>0:
        for j in range(idx_cand.size):
            ylim = np.nanmax(nm_temporal,axis=1)-np.nanmin(nm_temporal,axis=1)
            if np.max(nm_temporal_se,axis=1)[idx_cand[j]]>0 and ylim[idx_cand[j]]<5:
                example_all_rois_list_0.append(idx_cand[j])
                break


example_all_rois_list = list.copy(example_all_rois_list_0)
example_all_rois = np.asarray(example_all_rois_list)
example_all_rois_cluster = df_results['cluster_label_nm_dendro_num'].iloc[example_all_rois]
example_all_rois_cluster_plot = np.asarray(num_type - example_all_rois_cluster)


example_all_rois = np.delete(example_all_rois, [-3,-4,-6,-10,-13,-15], axis=0)

example_all_rois_temporal = nm_temporal[example_all_rois,:]
example_all_rois_temporal_se = nm_temporal_se[example_all_rois,:]
example_temporal_len_list = [nm_temporal.shape[1]]

fig_params_example = fig_params.copy()
fig_params_example['offset'] = 0.04
fig_params_example['height_sub'] = 18
# Figure 9, related to Figure 2
file_save_example_all_corr = folder_figure + 'nm_fig_1D' 
ylimit_example = figure_example_temporal_nm(example_all_rois_temporal, example_all_rois_temporal_se, 0.2,
                                 fig_params_example, file_save_example_all_corr, sampling_rate=5)


    


