# crop the full `220501_wb3_co2_15_5z18R_merfish5` slice(s)
# -- 10k cells

K = 2

import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import random
import math
import timeit
# start = timeit.default_timer()
# stop = timeit.default_timer()
# print(f'Time: {stop - start}') 
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
pd.options.mode.chained_assignment = None  # default='warn'

red = '#c0362c'
lightgreen = '#93c572'
darkgreen = '#4c9141'
lightblue = '#5d8aa8'
darkblue = '#2e589c'
white = '#fafafa'
lightgray = '#d3d3d3'
darkgray ='#545454'
lightorange = '#fabc2e'
darkorange = '#fb9912'
yellow = '#e4d00a'

color_list = [red, darkorange, lightorange, lightgreen, lightblue]


experiment_name = '220501_wb3_co2_15_5z18R_merfish5'


path = '../processed_data/decoded_spots/mouse2_coronal/spots_'+experiment_name+'.csv'
expr_df = pd.read_csv(path)

print(f'expr_df.shape {expr_df.shape}')
print(f'xmin={expr_df.x.min()} xmax={expr_df.x.max()}')
print(f'ymin={expr_df.y.min()} ymax={expr_df.y.max()}')
print(f'global xmin={expr_df.global_x.min()} xmax={expr_df.global_x.max()}')
print(f'global ymin={expr_df.global_y.min()} ymax={expr_df.global_y.max()}')
print(f'global z unique {expr_df.global_z.unique()}')
print(f'#fov {expr_df.fov.nunique()}')
print(f'#gene {expr_df.target_gene.nunique()}')

Z = expr_df.global_z.nunique()
print(f'Z={Z}')


path = '../processed_data/cell_boundaries/mouse2_coronal/'+experiment_name+'.csv'
seg_df = pd.read_csv(path)

N = seg_df['Unnamed: 0'].nunique()
print(f'#cells={N}')


# load data
path = '../processed_data/counts_mouse2_coronal.h5ad'
counts = anndata.read_h5ad(path)

path = '../processed_data/raw_counts_mouse2_coronal.h5ad'
raw_counts = anndata.read_h5ad(path)

print(counts.X.shape)
print(counts.obs.shape)


### cell list with seg at all panels
# remove rows with NA 
seg_df_rmna = seg_df.dropna(axis='index').copy()
cl_seg_all = seg_df_rmna['Unnamed: 0'].unique().tolist()
print(f"#cells with seg at all panels {len(cl_seg_all)}")

### cell list with cell centroid available
centroid_df = (counts.obs)[counts.obs.sample_id=='co2_sample15'].copy() #!!!<<<
cl_centroid = centroid_df.index.to_list()
print(f'#cells with centroid available {len(cl_centroid)}')

### common list of `cl_seg_all` and `cl_centroid`
cl__seg_all__centroid = list(set(cl_seg_all).intersection(set(cl_centroid)))
print(f'#cells in common {len(cl__seg_all__centroid)}')

# subset dfs
seg_df_com = seg_df_rmna[seg_df_rmna['Unnamed: 0'].isin(cl__seg_all__centroid)]
centroid_df_com = centroid_df[centroid_df.index.isin(cl__seg_all__centroid)]

### cell list with ONLY 1 centroid based on z=0
start = timeit.default_timer()

cl_1centroid = []

for ic, c in enumerate(cl__seg_all__centroid[:]):
    if ic%1e4==0:
        print(ic/len(cl__seg_all__centroid))
    
    seg_c = seg_df_com[seg_df_com['Unnamed: 0']==c]
    
    # seg
    cb_x_str = seg_c['boundaryX_z0'].iloc[0]
    cb_y_str = seg_c['boundaryY_z0'].iloc[0]
    cb_x = np.fromstring(cb_x_str, sep=',')
    cb_y = np.fromstring(cb_y_str, sep=',')
    
    # bbox
    cb_x_max = np.max(cb_x)
    cb_x_min = np.min(cb_x)
    cb_y_max = np.max(cb_y)
    cb_y_min = np.min(cb_y)
    
    # seg polygon
    cb_arr = np.array([cb_x, cb_y]).T
    cb_tuple = list(map(tuple, cb_arr))
    polygon = Polygon(cb_tuple)
    
    # get the centroid in the bbox
    f1 = centroid_df_com.center_x >= cb_x_min
    f2 = centroid_df_com.center_x <= cb_x_max
    f3 = centroid_df_com.center_y >= cb_y_min
    f4 = centroid_df_com.center_y <= cb_y_max
    centroid_df_com_c = centroid_df_com[f1&f2&f3&f4]
    
    if len(centroid_df_com_c)>0:
        # check how many centroid within the polygon
        num_ = 0
        for j in range(len(centroid_df_com_c)):
                point = Point(centroid_df_com_c.center_x.iloc[j], centroid_df_com_c.center_y.iloc[j])
                if polygon.contains(point):
                    num_ = num_+1
        # keep the cell, if only 1
        if num_ == 1: #!!!
            cl_1centroid.append(c)
            
stop = timeit.default_timer()
print(f'Time: {stop - start}') # 291.3848544168286

print(f'#cells with 1 centroid at z=0 {len(cl_1centroid)}') # 58355

# sort cl_1centroid
cl_1centroid.sort()

### crop cells in `cl_1centroid`
cl_1centroid_TOCROP = cl_1centroid[(10000*K):(10000*K+10000)] # <<<
seg_df_com_TOCROP = seg_df_com[seg_df_com['Unnamed: 0'].isin(cl_1centroid_TOCROP)]
centroid_df_com_TOCROP = centroid_df_com[centroid_df_com.index.isin(cl_1centroid_TOCROP)]
xmin_ = centroid_df_com_TOCROP.center_x.min()-50
xmax_ = centroid_df_com_TOCROP.center_x.max()+50
ymin_ = centroid_df_com_TOCROP.center_y.min()-50
ymax_ = centroid_df_com_TOCROP.center_y.max()+50
expr_df_TOCROP = expr_df[(expr_df.global_x >= xmin_)
                        &(expr_df.global_y >= ymin_)
                        &(expr_df.global_x <= xmax_)
                        &(expr_df.global_y <= ymax_)]

start = timeit.default_timer()

crop_cell_df_dict = {}
print(f'total #cells to crop {len(cl_1centroid_TOCROP)}')

for ic, c in enumerate(cl_1centroid_TOCROP):
    if ic%1e3==0:
        print(ic/len(cl_1centroid_TOCROP))
    
    for z in range(Z):
        seg_c = seg_df_com_TOCROP[seg_df_com_TOCROP['Unnamed: 0']==c]
        cb_x_str = seg_c['boundaryX_z'+str(z)].iloc[0]
        cb_y_str = seg_c['boundaryY_z'+str(z)].iloc[0]
        cb_x = np.fromstring(cb_x_str, sep=',')
        cb_y = np.fromstring(cb_y_str, sep=',')

        cb_arr = np.array([cb_x, cb_y]).T
        cb_tuple = list(map(tuple, cb_arr))
        polygon = Polygon(cb_tuple)
        
        if z == 0:
            cb_x_max_z0 = np.amax(cb_x)
            cb_x_min_z0 = np.amin(cb_x)
            cb_y_max_z0 = np.amax(cb_y)
            cb_y_min_z0 = np.amin(cb_y)
            polygon_z0 = polygon
        
        cb_x_max = np.amax(cb_x)
        cb_x_min = np.amin(cb_x)
        cb_y_max = np.amax(cb_y)
        cb_y_min = np.amin(cb_y)

        expr_df_c = expr_df_TOCROP[(expr_df_TOCROP.global_z == z)
                                  &(expr_df_TOCROP.global_x >= cb_x_min)
                                  &(expr_df_TOCROP.global_y >= cb_y_min)
                                  &(expr_df_TOCROP.global_x <= cb_x_max)
                                  &(expr_df_TOCROP.global_y <= cb_y_max)]

        centroid_c = centroid_df_com_TOCROP[(centroid_df_com_TOCROP.center_x >= cb_x_min_z0)
                                           &(centroid_df_com_TOCROP.center_y >= cb_y_min_z0)
                                           &(centroid_df_com_TOCROP.center_x <= cb_x_max_z0)
                                           &(centroid_df_com_TOCROP.center_y <= cb_y_max_z0)]
        if len(centroid_c) == 1:
            center_xy = [centroid_c.center_x.iloc[0], centroid_c.center_y.iloc[0]]
        else: # >1
            for j in range(len(centroid_c)):
                point = Point(centroid_c.center_x.iloc[j], centroid_c.center_y.iloc[j])
                if polygon_z0.contains(point):
                    center_xy = [centroid_c.center_x.iloc[j], centroid_c.center_y.iloc[j]]
                    break

        label_c = np.full(len(expr_df_c), '0', dtype=object)
        for j in range(len(expr_df_c)):
            point = Point(expr_df_c.global_x.iloc[j], expr_df_c.global_y.iloc[j])
            if polygon.contains(point):
                label_c[j] = c
        expr_df_c['label'] = label_c
        expr_df_c = expr_df_c[expr_df_c.label!='0']
        expr_df_c['center_x'] = center_xy[0]
        expr_df_c['center_y'] = center_xy[1]        
        crop_cell_df_dict[c+'_'+str(z)] = expr_df_c
        
# check on cropped cell
# tmp = expr_df_c[expr_df_c.label!='0']
# plt.scatter(tmp.global_x, tmp.global_y)
# plt.plot(cb_x, cb_y)

stop = timeit.default_timer()
print(f'Time: {stop - start}') 

# concatenate all cells to one df
crop_cell_df = pd.concat(list(crop_cell_df_dict.values()))

# pkl
outfile = 'output_step1/crop_data_10K'+str(K)+'.pkl'
# save
pickle_dict = {}
pickle_dict['crop_cell_df'] = crop_cell_df
pickle_dict['cl_1centroid'] = cl_1centroid
pickle_dict['seg_df_com'] = seg_df_com
pickle_dict['centroid_df_com'] = centroid_df_com
with open(outfile, 'wb') as f:
    pickle.dump(pickle_dict, f)
    
# load
# with open(outfile, 'rb') as f:
#     data_dict = pickle.load(f)
# crop_cell_df = pickle_dict['crop_cell_df']
# plot_seg_df = pickle_dict['plot_seg_df']
# plot_xmin = pickle_dict['plot_xmin']
# plot_ymin = pickle_dict['plot_ymin']
# plot_d_x = pickle_dict['plot_d_x']
# plot_d_y = pickle_dict['plot_d_y']







