import os
import pickle
import timeit
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import torch
from torch import nn, optim
import math
from math import pi
from scipy import stats
from scipy.stats import beta
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
stats2 = importr('stats')
from sklearn.neighbors import KernelDensity
import ipdb # ipdb.set_trace()
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
pd.options.mode.chained_assignment = None  # default='warn'


class model_beta(torch.nn.Module):
    '''
    NHPP torch model, log likelihood of alternative models with beta kernels
    '''
    def __init__(self, init_A, init_B):
        super().__init__()
        self.A = torch.nn.Parameter(torch.rand(())+init_A)
        self.B = torch.nn.Parameter(torch.rand(())+init_B)

    def forward(self, ri, c0i, ni, a, b, ri_clamp_min, ri_clamp_max, L1_lam):
        '''
        location-wise log-likelihood (across locations of all cells)
        '''
        BB = math.gamma(a[0])*math.gamma(b[0]) / math.gamma(a[0]+b[0])
        ri_clamp = torch.clamp(ri, min=ri_clamp_min, max=ri_clamp_max)
        lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.exp(self.A)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + torch.exp(self.B)*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.exp(self.A)*a)/(a+b)+(torch.exp(self.B))/(2.0)) - L1_lam*torch.exp(self.A) # constrain A=exp(A)， B=exp(B)
        return lli


class model_null(torch.nn.Module):
    '''
    NHPP torch model, log likelihood null model
    '''
    def __init__(self, init_B):
        super().__init__()
        self.B = torch.nn.Parameter(torch.rand(())+init_B)

    def forward(self, ri, c0i, ni, ri_clamp_min, ri_clamp_max):
        '''
        subject-wise log-likelihood
        '''
        ri_clamp = torch.clamp(ri, min=ri_clamp_min, max=ri_clamp_max)
        lli = torch.log(c0i*2.0*math.pi)+torch.log(self.B*ri_clamp)-(c0i*2.0*math.pi/ni)*(self.B/2.0) 
        return lli


def loss_ll(pred):
    '''
    NHPP torch model, negative log likelihood as loss, for minimazing
    '''
    loss = - torch.sum(pred)
    return loss

class ELLA:
    '''
    Class of ELLA
    '''
    def __init__(self, dataset='untitled', beta_kernel_param_list=None, adam_learning_rate_max=1e-2, adam_learning_rate_min=1e-3, adam_learning_rate_adjust=1e7, adam_delta_loss_max=1e-2, adam_delta_loss_min=1e-5, adam_delta_loss_adjust=1e8, adam_niter_loss_unchange=20, max_iter=5000, min_iter=100, max_ntanbin=25, ri_clamp_min=0.01, ri_clamp_max=1.0, hpp_solution='analytical', lam_filter=0.0, L1_lam=0):
        '''
        Constructor
        Args:
            dateset: name of the data working on, default='untitled'
            beta_kernel_param_list: the list of beta kernels, will be specified later; default= the 22 kernels
            adam_learning_rate_max: max Adam initial learning rate; default=1e-2
            adam_learning_rate_min: min Adam initial learning rate; default=1e-3
            adam_learning_rate_adjust: initial Adam learning rate = maxll/adam_learning_rate_adjust, truncated at [adam_learning_rate_min, adam_learning_rate_max], maxll is the analytical hpp loglikelihood value; default=1e7
            adam_delta_loss_max: for Adam early stopping, max delta loss function; default=1e-2
            adam_delta_loss_min: for Adam early stopping, min delta loss function; default=1e-5
            adam_delta_loss_adjust: for Adam early stopping, delta loss = maxll/adam_delta_loss_adjust, truncated at [adam_delta_loss_min, adam_delta_loss_max], maxll is the analytical hpp loglikelihood value; default=1e8
            adam_niter_loss_unchange: for Adam early stopping, Adam stops if loss decrease < delta loss for adam_niter_loss_unchange iterations; default=20
            max_iter: Adam max number of iterations; default=5000
            min_iter: Adam min number of iterations; default=100
            max_ntanbin: for cells registration, number of bins in pi/2 (a quantrant); default=25
            ri_clamp_min: relative position truncated at min=ri_clamp_min; default=0.01
            ri_clamp_max: relative position truncated at max=ri_clamp_max; default=1.0
            hpp_solution: use 'analytical' or 'numerical' (Adam) to obtain hpp (null) model estimation; default='analytical'
            lam_filter: filter out estimated expression intensity lam with max(lam)-min(lam) <= lam_filter; default=0.0
            L1_lam: weight of L1 penalty on the scale parameter in nhpp model kernel fitting; default=0
        See also: https://jadexq.github.io/ELLA/advanced.html
        '''
        
        # name of the analysis
        self.dataset = dataset

        # basic constants
        self.epsilon = 1e-10 # a very small value, to prevent division by zero etc.
        self.nt: int # number of cell types
        self.nk: int # number of default kernels
        self.nc_all: int # total number of cells
        self.max_ntanbin = max_ntanbin
        self.ri_clamp_min = ri_clamp_min
        self.ri_clamp_max = ri_clamp_max
        self.hpp_solution = hpp_solution
        self.lam_filter = lam_filter
        self.sig_cutoff = 0.05 # significant level 
        self.n_overflow = 500 # if x larger than this exp(x) will be overflow

        # constant for computing max_ntanbin
        self.nc4ntanbin = 10 # number of cells (replacement allowed) for computing ntanbin
        self.min_bp = 5 # min boundary points in each tanbin
        self.high_res = 200 # x/yrange so as to be high resolution
        self.min_ntanbin_error = 3 # if ntanbin less than this, will print a message

        # list/dict/df
        self.ng_dict = {} # number of genes in each cell type
        self.nc_dict = {} # number of cells in each cell type
        self.type_list = [] # cell type
        self.cell_list_all = [] # list of all cells
        self.gene_list_dict = {} # gene list of each cell type
        self.cell_list_dict = {} # cell list of each cell type
        self.beta_kernel_param_list = [] # kernel param list
        self.cell_mask_df: pandas.DataFrame = None # df of cell masks
        self.data_df: pandas.DataFrame = None # df of gene expression data
        self.r_tl = {} # for nhpp fit, radius
        self.c0_tl = {} # for nhpp fit, read depth
        self.c0_tl_homo = {} # for nhpp fit, homo case calculation, no repeat wrt genes
        self.n_tl = {} # for nhpp fit, total umi count of a gene
        self.n_tl_homo = {} # for nhpp fit, homo case calculation, no repeat wrt genes
        self.A_est = {} # for nhpp fit, A (beta) est
        self.B_est = {} # for nhpp fit, B (alpha) est
        self.mll_est = {} # for nhpp fit, max log likelihood
        self.pv_raw_tl = {} # for nhpp result, raw pv
        self.ts_tl = {} # for nhpp result, likelihood ratio test statistic value
        self.pv_cauchy_tl = {} # for nhpp result, cauchy combined pv
        self.pv_fdr_tl = {} # for nhpp result, fdr-BY pv
        self.best_kernel_tl = {} # for nhpp result, best kernel
        self.weight_ml = {} # for nhpp results, weights calculated from max likelihood
        self.scores = {} # for nhpp results, scores
        self.weighted_lam_est = {} # for nhpp results, model averaging lam corresponding to all sig kernels
        self.ntanbin_dict = {} # ntanbin for each cell type, related to resolution
        self.kmeans_K_max = {} # kmeans clustering results
        self.kmeans_sig_lam_points_std_merged = None
        self.kmeans_exclude_nonsig = None
        self.kmeans_n_sig = None
        self.labels_dict = {} # pattern cluster labels

        # NHPP fit constants
        self.optimizer = 'adam' 
        self.learning_rate_max = adam_learning_rate_max
        self.learning_rate_min = adam_learning_rate_min
        self.learning_rate_adjust = adam_learning_rate_adjust
        self.max_iter = max_iter 
        self.min_iter = min_iter 
        self.delta_loss_max = adam_delta_loss_max
        self.delta_loss_min = adam_delta_loss_min
        self.delta_loss_adjust = adam_delta_loss_adjust
        self.niter_loss_unchange = adam_niter_loss_unchange
        self.L1_lam = L1_lam
        
        # NHPP intermediate results
        self.initial_A_dict = {} # Adam A (beta in manuscript) initial value
        self.initial_B_dict = {} # Adam B (alpha in manuscript) initial value
        self.loss_nhpp_dict = {} # save nhpp loss function values for plotting and covergence check
        self.loss_hpp_dict = {} # save hpp loss function values for plotting and covergence check
        self.early_stop_dict = {} # save if Adam stops early
        self.adaptive_learning_rate_dict = {} # save Adam adaptive learning rate
        self.adaptive_delta_loss_dict = {} # save Adam adaptive delta loss
        
        if beta_kernel_param_list is None:
            self.beta_kernel_param_list = [ # the default 22 kernels
                [1,     2.71],   
                [1.26,  3.34],   
                [2.05,  5.19],   
                [6.99,  14.98],  
                [19.41, 28.62],  
                [28.5,  28.5],   
                [28.62, 19.41],  
                [14.98, 6.99],   
                [5.19,  2.05],   
                [3.34,  1.26],   
                [2.71,  1],      
                [1,    2],       
                [1.13, 2.19],    
                [1.38, 2.52],    
                [1.88, 3.06],    
                [2.73, 3.60],    
                [3.5,  3.5],     
                [3.60, 2.73],    
                [3.06, 1.88],    
                [2.52, 1.38],    
                [2.19, 1.13],    
                [2,    1],       
            ]
        else:
            self.beta_kernel_param_list = beta_kernel_param_list # customized kernels
            

    def load_data(self, data_dict=None, data_path=None):
        '''
        Function of loading data
        Args:
            data_dict: input data dict
            data_path: path of input data
        Returns:
            None
        '''
        if data_dict==None and data_path==None:
            print('Please specify input data.')
        if data_dict!=None and data_path!=None:
            print('Please specify input data with EITHER data_dict or data_path.')
        if data_dict==None and data_path!=None:
            data_dict = pd.read_pickle(data_path)
                    
        self.type_list = data_dict['types'] # cell type list
        self.gene_list_dict = data_dict['genes'] # gene lists for each type
        self.cell_list_dict = data_dict['cells'] # cell list for each type
        self.cell_list_all = data_dict['cells_all'] # list of all cells
        self.cell_mask_df = data_dict['cell_seg'] # mask of cells
        self.data_df = data_dict['expr'] # gene expression data
        for t in self.type_list:
            self.ntanbin_dict[t] = self.max_ntanbin # ntanbin for each cell type, all set to the same value
        
        # update some global constants
        self.nt = len(self.type_list) # number of cell types
        self.nc_all = len(self.cell_list_all) # number of all cells
        ng_list = [] # number of genes
        for t in self.type_list:
            self.ng_dict[t] = len(self.gene_list_dict[t])
            ng_list.append(len(self.gene_list_dict[t]))
        nc_list = [] # number of cells
        for t in self.type_list:
            self.nc_dict[t] = len(self.cell_list_dict[t])
            nc_list.append(len(self.cell_list_dict[t]))
        self.nk = len(self.beta_kernel_param_list) # number of default kernels

        # print
        print(f"Number of cell types {self.nt}")
        print(f"Average number of genes {np.mean(ng_list)}")
        print(f"Average number of cells {np.mean(nc_list)}")
        print(f"Number of default kernels {self.nk}")

    
    def specify_ntanbin(self, input_ntanbin_dict=None):
        '''
        Function for specifying ntanbin OR calculate ntanbin across cell types
        Args:
            input_ntanbin_dict: give ntanbin manually (can be different across cell types), a dictionary, keys corresponding to cell type list
        Returns:
            None: (update self.ntanbin_dict)
        Note:
            Running this function can take some time and is not necessary. Usually, specify `max_ntanbin` should be sufficient.
            If run this function with input_ntanbin_dict=None, ntanbin will be calculated based on the resolution of cell segmentation boundary or mask.
        '''
        if input_ntanbin_dict is not None: # use customized ntanbin across cell types
            self.ntanbin_dict = input_ntanbin_dict 

        if input_ntanbin_dict is None: # compute ntanbin for each cell type:
            for t in self.type_list:
                # specify ntanbin_gen based on cell seg mask/boundary
                # random sample self.nc4ntanbin cells, allow replace
                cell_list_sampled = np.random.choice(self.cell_list_dict[t], self.nc4ntanbin, replace=True)
                cell_mask_df_sampled = self.cell_mask_df[self.cell_mask_df.cell.isin(cell_list_sampled)]
                # compute the #x and #y unique coords of these sampled cells
                nxu_sampled = []
                nyu_sampled = []
                for c in cell_list_sampled:
                    mask_c = cell_mask_df_sampled[cell_mask_df_sampled.cell==c]
                    nxu_sampled.append(mask_c.x.nunique())
                    nyu_sampled.append(mask_c.y.nunique())
                # specify ntanbin for pi/2 (a quantrant)
                # if resolution is super high
                if np.mean(nxu_sampled)>self.high_res and np.mean(nyu_sampled)>self.high_res:
                    ntanbin=self.max_ntanbin
                # if resolution is not super high
                else:
                    # require at least self.min_bp boundary points in each tanbin
                    theta = 2*np.arctan(self.min_bp/np.mean(nxu_sampled+nyu_sampled))
                    ntanbin_ = (pi/2)/theta
                    ntanbin = np.ceil(ntanbin_)
                    if ntanbin<self.min_ntanbin_error:
                        print(f'Cell type {t} failed, resolution not high enougth to support the analysis')
                # asign
                self.ntanbin_dict[t] = ntanbin


    def register_cells(self, nc_demo=None, outfile='output/df_registered.pkl'):
        '''
        Function of registering cells to unit circle
        Args:
            demo_nc:
                default - register all cells
                or register given number of cells for testing
        Returns:
            None
            (update global variable `df_registered`, save df_registered as .pkl)
        '''
        if nc_demo is None:
            nc_demo = self.nc_all

        dict_registered = {}

        df = self.data_df.copy() # cp original data
        df_gbC = df.groupby('cell', observed=False) # group by `cell`

        for ic, c in enumerate(tqdm(self.cell_list_all[:nc_demo], desc="Processing cells")):
            df_c = df_gbC.get_group(c).copy() # df for cell c
            t = df_c.type.iloc[0] # cell type for cell c
            mask_df_c = self.cell_mask_df[self.cell_mask_df.cell == c] # get the mask df for cell c
            center_c = [int(df_c.centerX.iloc[0]), int(df_c.centerY.iloc[0])] # nuclear center of cell c
            tanbin = np.linspace(0, pi/2, self.ntanbin_dict[t]+1)
            delta_tanbin = (2*math.pi)/(self.ntanbin_dict[t]*4)

            # add centered coord and ratio=y/x for df_c and mask_df_c
            df_c['x_c'] = df_c.x.copy() - center_c[0]
            df_c['y_c'] = df_c.y.copy() - center_c[1]
            df_c['d_c'] = (df_c.x_c.copy()**2+df_c.y_c.copy()**2)**0.5
            df_c['arctan'] = np.absolute(np.arctan(df_c.y_c / (df_c.x_c+self.epsilon)))
            mask_df_c['x_c'] = mask_df_c.x.copy() - center_c[0]
            mask_df_c['y_c'] = mask_df_c.y.copy() - center_c[1]
            mask_df_c['d_c'] = (mask_df_c.x_c.copy()**2+mask_df_c.y_c.copy()**2)**0.5
            mask_df_c['arctan'] = np.absolute(np.arctan(mask_df_c.y_c / (mask_df_c.x_c+self.epsilon)))

            # in each quatrant, find dismax_c for each tanbin interval using mask_df_c
            mask_df_c_q_dict = {}
            mask_df_c_q_dict['0'] = mask_df_c[(mask_df_c.x_c>=0) & (mask_df_c.y_c>=0)]
            mask_df_c_q_dict['1'] = mask_df_c[(mask_df_c.x_c<=0) & (mask_df_c.y_c>=0)]
            mask_df_c_q_dict['2'] = mask_df_c[(mask_df_c.x_c<=0) & (mask_df_c.y_c<=0)]
            mask_df_c_q_dict['3'] = mask_df_c[(mask_df_c.x_c>=0) & (mask_df_c.y_c<=0)]
            # compute the dismax_c
            dismax_c_mat = np.zeros((self.ntanbin_dict[t], 4))
            for q in range(4): # in each of the 4 quantrants
                mask_df_c_q = mask_df_c_q_dict[str(q)]
                mask_df_c_q['arctan_idx'] = (mask_df_c_q.arctan/delta_tanbin).astype(int) # arctan_idx from 0 to self.ntanbin_dict[t]-1
                dismax_c_mat[mask_df_c_q.groupby('arctan_idx').max()['d_c'].index.to_numpy(),q] = mask_df_c_q.groupby('arctan_idx').max()['d_c'].values # automatically sorted by arctan_idx from 0 to self.ntanbin_dict[t]-1

            # for df_c, for arctan in each interval, find max dis using dismax_c
            df_c_q_dict = {}
            df_c_q_dict['0'] = df_c[(df_c.x_c>=0) & (df_c.y_c>=0)]
            df_c_q_dict['1'] = df_c[(df_c.x_c<=0) & (df_c.y_c>=0)]
            df_c_q_dict['2'] = df_c[(df_c.x_c<=0) & (df_c.y_c<=0)]
            df_c_q_dict['3'] = df_c[(df_c.x_c>=0) & (df_c.y_c<=0)]
            d_c_maxc_dict = {}
            for q in range(4): # in each of the 4 quantrants
                df_c_q = df_c_q_dict[str(q)]
                d_c_maxc_q = np.zeros(len(df_c_q))
                df_c_q['arctan_idx'] = (df_c_q.arctan/delta_tanbin).astype(int) # arctan_idx from 0 to self.ntanbin_dict[t]-1
                for ai in range(self.ntanbin_dict[t]):
                    d_c_maxc_q[df_c_q.arctan_idx.values==ai] = dismax_c_mat[ai,q]
                d_c_maxc_dict[str(q)] = d_c_maxc_q
            d_c_maxc = np.zeros(len(df_c))
            d_c_maxc[(df_c.x_c>=0) & (df_c.y_c>=0)] = d_c_maxc_dict['0']
            d_c_maxc[(df_c.x_c<=0) & (df_c.y_c>=0)] = d_c_maxc_dict['1']
            d_c_maxc[(df_c.x_c<=0) & (df_c.y_c<=0)] = d_c_maxc_dict['2']
            d_c_maxc[(df_c.x_c>=0) & (df_c.y_c<=0)] = d_c_maxc_dict['3']
            df_c['d_c_maxc'] = d_c_maxc

            # scale centered x_c and y_c 
            d_c_s = np.zeros(len(df_c))
            x_c_s = np.zeros(len(df_c))
            y_c_s = np.zeros(len(df_c))
            d_c_s = df_c.d_c/(df_c.d_c_maxc+self.epsilon)
            x_c_s = df_c.x_c*(d_c_s/(df_c.d_c+self.epsilon))
            y_c_s = df_c.y_c*(d_c_s/(df_c.d_c+self.epsilon))
            df_c['x_c_s'] = x_c_s
            df_c['y_c_s'] = y_c_s
            df_c['d_c_s'] = d_c_s

            # append
            dict_registered[c] = df_c
            del df_c

        # concatenate to one df
        df_registered = pd.concat(list(dict_registered.values()))
        print(f'Number of cells registered {len(dict_registered)}')

        # update global data variable
        self.df_registered = df_registered

        # save registered df
        pickle_dict = {}
        pickle_dict['df_registered'] = df_registered

        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(outfile, 'wb') as f:
            pickle.dump(pickle_dict, f)


    def load_registered_cells(self, registered_dict=None, registered_path=None):
        '''
        Function of loading registered cells .pkl generated by `register_cells`
        Args:
            registered_path: path of the registered cells .pkl
            registered_dict: dict of registered cells
        Returns:
            None
            (update global data variable with `df_registered`)
        '''
        if registered_dict==None and registered_path==None:
            print('Please specify registered data.')
        if registered_dict!=None and registered_path!=None:
            print('Please specify registered data with EITHER registered_dict or registered_path.')
        if registered_dict==None and registered_path!=None:
            registered_dict = pd.read_pickle(registered_path)

        # registered cells df
        df_registered = registered_dict['df_registered']
        # update global data variable
        self.df_registered = df_registered
        

    def nhpp_prepare(self, ci_fix=None):
        '''
        Function of pareparing data for NHPP fit from self.df_registered
        Args:
            ci_fix = None, give ci (=1) for simulations if needed, othereise, will be calculated from sc_total/1000
        Returns:
            None (update self.r_tl, self.c0_tl, self.n_tl, etc.)
        '''
        df_gbT = self.df_registered.groupby('type', observed=False) # group by type

        for it, t in enumerate(self.type_list): # loop over all cell types
            df_t = df_gbT.get_group(t).copy() # df for cell type t
            df_t_gbG = df_t.groupby('gene', observed=False) # # groupby gene

            r_t = [] # relative position
            c0_t = [] # sc total counts of all genes
            c0_t_homo = []
            n_t = [] # the umi counts at r_ij
            n_t_homo = []

            for ig, g in enumerate(tqdm(self.gene_list_dict[t], desc=f'Preparing data for {t}')):
                n_g = []
                n_g_homo = []
                c0_g = []
                c0_g_homo = []
                r_g = []

                df_t_g = df_t_gbG.get_group(g) # df for gene g in cell type t
                df_t_g_gbC = df_t_g.groupby('cell', observed=False) # group by cell
                cell_list_t_g = df_t_g.cell.unique().tolist() # cell list for gene g in cell type t

                for c in cell_list_t_g:
                    n_c = []
                    c0_c = []
                    r_c = []

                    df_t_g_c = df_t_g_gbC.get_group(c)
                    ni = df_t_g_c.umi.sum()
                    c0i = df_t_g_c.sc_total.iloc[0]

                    if ci_fix is None:
                        # if real data
                        c0ik = int(c0i/1000)+1 # count per 1000, round up
                    if ci_fix is not None:
                    # if simulation, can fix ci(=1)
                        c0ik = int(ci_fix)

                    for j in range(len(df_t_g_c)):
                        umi = df_t_g_c.umi.iloc[j]
                        rj = max(df_t_g_c.d_c_s.iloc[j], self.epsilon)
                        r_c = r_c + [rj]*umi
                    n_c = [ni]*len(r_c)
                    c0_c = [c0ik]*len(r_c)
                    c0_g_homo.append(c0ik)
                    n_g_homo.append(len(r_c))

                    # concatenate all cells of gene-g
                    n_g = n_g + n_c
                    c0_g = c0_g + c0_c
                    r_g = r_g + r_c

                # append all genes of type t
                r_t.append(r_g)
                c0_t.append(c0_g)
                c0_t_homo.append(c0_g_homo)
                n_t.append(n_g)
                n_t_homo.append(n_g_homo)

            del df_t
            
            # add types
            self.r_tl[t] = r_t
            self.c0_tl[t] = c0_t
            self.c0_tl_homo[t] = c0_t_homo
            self.n_tl[t] = n_t
            self.n_tl_homo[t] = n_t_homo

        # save registered df
        pickle_dict = {}
        pickle_dict['r_tl'] = self.r_tl
        pickle_dict['c0_tl'] = self.c0_tl
        pickle_dict['c0_tl_homo'] = self.c0_tl_homo
        pickle_dict['n_tl'] = self.n_tl
        pickle_dict['n_tl_homo'] = self.n_tl_homo
        pickle_dict['type_list'] = self.type_list
        pickle_dict['gene_list_dict'] = self.gene_list_dict
        pickle_dict['cell_list_dict'] = self.cell_list_dict
        pickle_dict['cell_list_all'] = self.cell_list_all
        
        output_dir = 'output'
        outfile = 'output/df_nhpp_prepared.pkl'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(outfile, 'wb') as f:
            pickle.dump(pickle_dict, f)


    def load_nhpp_prepared(self, prepared_dict=None, prepared_path=None):
        '''
        Function of loading prepared data .pkl generated by `nhpp_prepare`
        Args:
            prepared_path: path of the prepared data .pkl
            prepared_dict: dict of prepared data
        Returns:
            None 
            (update global data variable with r_tl etc.)
        '''
        if prepared_dict==None and prepared_path==None:
            print('Please specify prepared data.')
        if prepared_dict!=None and prepared_path!=None:
            print('Please specify prepared data with EITHER prepared_dict or prepared_path.')
        if prepared_dict==None and prepared_path!=None:
            prepared_dict = pd.read_pickle(prepared_path)
                
        self.r_tl = data_dict['r_tl']
        self.c0_tl = data_dict['c0_tl']
        self.c0_tl_homo = data_dict['c0_tl_homo']
        self.n_tl = data_dict['n_tl']
        self.n_tl_homo = data_dict['n_tl_homo']

        self.type_list = data_dict['type_list'] # cell type list
        self.gene_list_dict = data_dict['gene_list_dict'] # gene lists for each type
        self.cell_list_dict = data_dict['cell_list_dict'] # cell list
        self.cell_list_all = data_dict['cell_list_all'] # list of all cells

        # update global constants
        self.nt = len(self.type_list) # number of cell types
        self.nc_all = len(self.cell_list_all) # number of all cells
        ng_list = [] # number of genes
        for t in self.type_list:
            self.ng_dict[t] = len(self.gene_list_dict[t])
            ng_list.append(len(self.gene_list_dict[t]))
        nc_list = []
        for t in self.type_list: # number of cells
            self.nc_dict[t] = len(self.cell_list_dict[t])
            nc_list.append(len(self.cell_list_dict[t]))
        self.nk = len(self.beta_kernel_param_list) # number of default kernels

        # print
        print(f"Number of cell types {self.nt}")
        print(f"Average number of genes {np.mean(ng_list)}")
        print(f"Average number of cells {np.mean(nc_list)}")
        print(f"Number of default kernels {self.nk}")

        
    def nhpp_fit(self, outfile = 'output/nhpp_fit_results.pkl', ng_demo=None, ig_start=0, save='less'):
        '''
        Function of NHPP fitting
        Args:
        outfile:
            save nhpp fit results as a .pkl
        demo_nc:
            default - run all genes
            or run a given number of genes for testing for each cell type
        ig_start:
            default=0, change this if run groups of genes separatly in parallel
        save:
            default='less', save less results (for example, don't save the loss at each iteration) to save space
        Returns:
            None (update self.A_est, self.C_est, self.mll_est)
        '''
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        else:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        for it, t in enumerate(self.type_list):
            A_est_t = np.zeros((self.ng_demo_dict[t], self.nk+1))
            B_est_t = np.zeros((self.ng_demo_dict[t], self.nk+1))
            mll_est_t = np.zeros((self.ng_demo_dict[t], self.nk+1))

            r_t = self.r_tl[t]
            c0_t = self.c0_tl[t]
            c0_t_homo = self.c0_tl_homo[t]
            n_t = self.n_tl[t]
            n_t_homo = self.n_tl_homo[t]

            initial_A_t = []
            initial_B_t = []
            loss_nhpp_t = []
            loss_hpp_t = []
            early_stop_t = []
            adaptive_learning_rate_t = []
            adaptive_delta_loss_t = []

            _gl = self.gene_list_dict[t][:self.ng_demo_dict[t]]
            for _ig, g in enumerate(tqdm(_gl, desc=f'NHPP fitting for {t}')):
                ig = int(_ig + ig_start)
                g = self.gene_list_dict[t][_ig]

                # get data for the ith gene
                r_g = r_t[ig]
                c0_g = c0_t[ig]
                c0_g_homo = np.array(c0_t_homo[ig])
                n_g = n_t[ig]
                n_g_homo = np.array(n_t_homo[ig])

                if (len(r_g)>0):
                    loss_nhpp_g = []
                    loss_hpp_g = []
                    early_stop_g = []

                    ### HPP analytical est and max loglikelihood value
                    # \sum_i=1^I Ji / 1/2 \sum_i=1^I c0i *(2pi)
                    hatB = np.sum(n_g_homo)/np.sum(c0_g_homo*math.pi) # B (alpha) est
                    maxll = np.sum(np.log(np.array(c0_g)*2.0*math.pi)+np.log(hatB*np.array(r_g))-(np.array(c0_g)*2.0*math.pi/np.array(n_g))*(hatB/2.0)) # HPP loglikelihood
                    
                    ### MLE of (N)HPP
                    # Adam initial values
                    init_A = -10
                    init_B_null = hatB
                    init_B_alter = np.log(np.maximum(hatB, 1e-7))
                    initial_A_t.append(init_A)
                    initial_B_t.append(init_B_null)
                    
                    # prepare data for torch
                    r = torch.reshape(torch.Tensor(r_g), (len(r_g),1))
                    c0 = torch.reshape(torch.Tensor(c0_g), (len(r_g),1))
                    n = torch.reshape(torch.Tensor(n_g), (len(r_g),1))
                    
                    # compute adaptive learning rate (LR) based on homo loglikelihood
                    LR = np.minimum(self.learning_rate_max, np.maximum(maxll/self.learning_rate_adjust, self.learning_rate_min))
                    # compute adaptive delta loss (DL) based on homo loglikelihood
                    DL = np.minimum(self.delta_loss_max, np.maximum(maxll/self.delta_loss_adjust, self.delta_loss_min))
                    
                    ## HPP fit (analytical or numerical)
                    if self.hpp_solution == 'analytical':
                        # save numerical HPP (null) solutions
                        B_est_t[ig, -1] = hatB
                        mll_est_t[ig, -1] = maxll
                    else:
                        # Adam analytical HPP (null) solutions
                        model = model_null(init_B_null)
                        # spesify optimizer
                        if (self.optimizer == 'adam'):
                            optimizer = optim.Adam(model.parameters(), lr=LR)
                        else:
                            print('should use Adam for max likelihood for HPP')
                        # training
                        loss_prev = math.inf
                        counter = 0
                        for step in range(0, self.max_iter):
                            optimizer.zero_grad()
                            predictions = model(r, c0, n, self.ri_clamp_min, self.ri_clamp_max)
                            loss = loss_ll(predictions)
                            loss_hpp_g.append(loss.detach().numpy())

                            # early stopping
                            es = self.max_iter
                            if step > self.min_iter:
                            #if step > 1:
                                if np.abs(loss_prev - loss.item()) < DL:
                                    counter = counter + 1
                                else:
                                    counter = 0
                                if counter > self.niter_loss_unchange:
                                    es = step
                                    break
                                loss_prev = loss.item()

                            # stop if loss become nan
                            if torch.isnan(loss):
                                es = step
                                break

                            loss.backward()
                            optimizer.step()
                            
                        # save analytical results
                        B_est_t[ig, -1] = model.B.data
                        mll_est_t[ig, -1] = - loss

                    ## NHPP fit
                    # Adam numerical solution, loop over all kernels
                    for k in range(self.nk):
                        # beta kernel shape parameters
                        aa = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][0]]*len(r_g)), (len(r_g),1))
                        bb = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][1]]*len(r_g)), (len(r_g),1))

                        # torch model
                        model = model_beta(init_A, init_B_alter)
                        # spesify optimizer
                        if (self.optimizer == 'adam'):
                            optimizer = optim.Adam(model.parameters(), lr=LR)
                        else:
                            print('should use adam for max likelihood')
                        # training
                        loss_prev = math.inf
                        counter = 0
                        loss_k = [] # save loss values across iterations for diagnostic plotting
                        for step in range(0, self.max_iter):
                            optimizer.zero_grad()
                            predictions = model(r, c0, n, aa, bb, self.ri_clamp_min, self.ri_clamp_max, self.L1_lam)
                            loss = loss_ll(predictions)
                            loss_k.append(loss.detach().numpy())

                            # early stopping
                            es = self.max_iter
                            if np.abs(loss_prev - loss.item()) < DL:
                                counter = counter + 1
                            else:
                                counter = 0
                            if counter > self.niter_loss_unchange:
                                es = step
                                break
                            loss_prev = loss.item()

                            # stop if loss become nan
                            if torch.isnan(loss):
                                es = step
                                break

                            loss.backward()
                            optimizer.step()

                        # save results
                        A_est_t[ig, k] = model.A.data
                        B_est_t[ig, k] = model.B.data
                        mll_est_t[ig, k] = - loss
                        early_stop_g.append(es)
                        loss_nhpp_g.append(loss_k)
                        
                early_stop_t.append(early_stop_g)
                loss_nhpp_t.append(loss_nhpp_g)
                loss_hpp_t.append(loss_hpp_g)
                adaptive_learning_rate_t.append(LR)
                adaptive_delta_loss_t.append(DL)
            
            # add to dict
            self.A_est[t] = A_est_t
            self.B_est[t] = B_est_t
            self.mll_est[t] = mll_est_t
            self.early_stop_dict[t] = early_stop_t
            self.loss_nhpp_dict[t] = loss_nhpp_t
            self.loss_hpp_dict[t] = loss_hpp_t
            self.initial_A_dict[t] = initial_A_t
            self.initial_B_dict[t] = initial_B_t
            self.adaptive_learning_rate_dict[t] = adaptive_learning_rate_t
            self.adaptive_delta_loss_dict[t] = adaptive_delta_loss_t

        # save A_est (beta in paper), B_est (alpha in paper), and mll_est
        pickle_dict = {}
        pickle_dict['A_est'] = self.A_est
        pickle_dict['B_est'] = self.B_est
        pickle_dict['mll_est'] = self.mll_est
        if save != 'less':
            pickle_dict['early_stop'] = self.early_stop_dict
            pickle_dict['loss_nhpp'] = self.loss_nhpp_dict
            pickle_dict['loss_hpp'] = self.loss_hpp_dict
            pickle_dict['initial_A'] = self.initial_A_dict
            pickle_dict['initial_B'] = self.initial_B_dict
            pickle_dict['adaptive_learning_rate'] = self.adaptive_learning_rate_dict

        output_dir = 'output'
        outfile = 'output/nhpp_fit_results.pkl'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(outfile, 'wb') as f:
            pickle.dump(pickle_dict, f)


    def load_nhpp_fit_results(self, res_dict=None, res_path=None):
        '''
        Function of loading nhpp fit results
        Args:
            res_dict: dict of NHPP results
            res_path: path of the dict
        Returns:
            None (update self.A_est, self.C_est, self.mll_est, self.best_kernel_tl)
        '''
        if res_dict==None and res_path==None:
            print('Please specify result data.')
        if res_dict!=None and res_path!=None:
            print('Please specify result data with EITHER res_dict or res_path.')
        if res_dict==None and res_path!=None:
            res_dict = pd.read_pickle(res_path)
        
        self.A_est = res_dict['A_est']
        self.B_est = res_dict['B_est']
        self.mll_est = res_dict['mll_est']
        try:
            self.early_stop_dict = res_dict['early_stop']
            self.loss_nhpp_dict = res_dict['loss_nhpp']
            self.loss_hpp_dict = res_dict['loss_hpp']
        except:
            pass


    def weighted_density_est(self, idx_selected_kernels=None, ng_demo=None):
        '''
        Function for computing modeling averaging weights, expressin intensities (lam est), and pattern scores
        Args:
            idx_selected_kernels: by default include all, or specify the idx of kernels that want to use
            ng_demo: number of genes included in each cell type, for test runs (should match with specifications in the previous nhpp_fit function)
        Returns:
            None (update self.weight_ml, self.scores, self.weighted_lam_est)
        '''
        if idx_selected_kernels is None:
            idx_selected_kernels=np.arange(self.nk).tolist()
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        else:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        weight_ml = {}
        scores = {}
        weighted_lam_est = {}

        for t in self.type_list:
            mll_t = self.mll_est[t][:,idx_selected_kernels]
            A_t_ = self.A_est[t][:,idx_selected_kernels]
            B_t_ = self.B_est[t][:,idx_selected_kernels]
            A_t = np.exp(A_t_)
            A_t[:,-1] = 0
            B_t = np.exp(B_t_)

            min_t = np.nanmin(mll_t) # handle nan
            mll_t = np.nan_to_num(mll_t, nan=min_t) # replace nan with minimum value
            A_t = np.nan_to_num(A_t, nan=0) # replace nan 0
            B_t = np.nan_to_num(B_t, nan=0) # replace nan 0

            # weight of kernel_included based on ml
            weight_ml_t = np.zeros(mll_t.shape)
            for ig in range(self.ng_demo_dict[t]):
                min_i = np.amin(mll_t[ig,:])
                mll_diff_i = mll_t[ig,:]-min_i
                # take care of numbers > n_overflow
                max_mll_diff_i = np.amax(mll_diff_i)
                if np.amax(max_mll_diff_i) > self.n_overflow:
                    d_ = max_mll_diff_i - self.n_overflow
                    mll_diff_i = mll_diff_i - d_
                    mll_diff_i = np.maximum(mll_diff_i, 0)
                ml_diff_i = np.exp(mll_diff_i)
                weight_ml_t[ig,:] = ml_diff_i/np.sum(ml_diff_i)

            # find the mode of weighted A*varphi+B
            scores_t = np.zeros(self.ng_demo_dict[t])
            weighted_lam_est_t = []
            x = np.linspace(0.001,0.999,100)
            for ig in range(self.ng_demo_dict[t]):
                weighted_lam_est_g = np.zeros(100)
                for ik, k in enumerate(idx_selected_kernels):
                    a, b = self.beta_kernel_param_list[k]
                    A_k = A_t[ig,ik]
                    B_k = B_t[ig,ik]
                    lam_k = np.maximum(0, A_k*beta.pdf(x, a, b)+B_k) # ensure non-negative
                    weighted_lam_est_g = weighted_lam_est_g + weight_ml_t[ig,ik]*lam_k
                idx = np.argmax(weighted_lam_est_g)
                scores_t[ig] = x[idx]
                weighted_lam_est_t.append(weighted_lam_est_g)

            self.weight_ml[t] = weight_ml_t
            self.scores[t] = scores_t
            self.weighted_lam_est[t] = weighted_lam_est_t

        # save scores, weights, and lam est in pkl
        output_dir = 'output'
        outfile = 'output/lam_est.pkl'
        pickle_dict = {}
        pickle_dict['scores_dict'] = self.scores
        pickle_dict['weights_dict'] = self.weight_ml
        pickle_dict['weighted_lam_dict'] = self.weighted_lam_est
      
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(outfile, 'wb') as f:
            pickle.dump(pickle_dict, f)


    def compute_pv(self, idx_selected_kernels=None, ng_demo=None):
        '''
        Function computing pv of idx_selected_kernels
        Args:
            idx_selected_kernels: by default include all
            ng_specified: by default include all genes
            ng_demo: number of genes included in each cell type, should match with specifications in the previous nhpp_fit function
        Returns:
            None (update self.pv_raw_tl, self.pv_cauchy_tl, self.pv_fdr_tl etc.)
        '''
        if idx_selected_kernels is None:
            idx_selected_kernels=np.arange(self.nk).tolist()
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        if ng_demo is not None:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        print(f'Number of kernels in use {len(idx_selected_kernels)}')

        for t in self.type_list:

            pv_t = np.full((self.ng_demo_dict[t], len(idx_selected_kernels)), 1.0) # to store pv
            ts_t = np.full((self.ng_demo_dict[t], len(idx_selected_kernels)), 0.0) # to store test statistic
            min_t = np.nanmin(self.mll_est[t])
            mll_est_t = np.nan_to_num(self.mll_est[t], nan=min_t) # replace nan with minimum value
            
            for ig in range(self.ng_demo_dict[t]):
                for ik in range(len(idx_selected_kernels)):
                    idx_k = idx_selected_kernels[ik]
                    T = -2*(mll_est_t[ig,-1] - mll_est_t[ig,idx_k])
                    cutoff0 = 0 
                    if T<=cutoff0 or np.isnan(T):
                        p = 1 - np.random.uniform(low=0.0, high=0.5, size=1)
                    else:
                        p = 1 - 0.5 - 0.5*stats.chi2.cdf(T, 1)
                    pv_t[ig,ik] = p
                    ts_t[ig,ik] = T

            # cauchy combination
            pv_cauchy_t = []
            w = np.full(len(idx_selected_kernels), 1/(len(idx_selected_kernels))) # equal weights
            for ig in range(self.ng_demo_dict[t]):
                pv_g = pv_t[ig,:]
                pv_g[pv_g>0.9999999] = 0.9999999
                pv_g[pv_g<0.0000001] = 0.0000001
                tt = np.sum(w * np.tan(pi*(0.5 - pv_g)))
                pc = 0.5 - np.arctan(tt)/pi
                pv_cauchy_t.append(pc)

            # FDR BY
            pv_fdr_t = np.array(stats2.p_adjust(FloatVector(pv_cauchy_t[:self.ng_dict[t]]), method = 'BY'))
            
            # update FDR pv based on lam est
            lam_t = self.weighted_lam_est[t]
            pv_fdr_t[(np.max(lam_t, axis=1)-np.min(lam_t, axis=1)) <= self.lam_filter] = 1

            # kernel(s) idx with max likelihood value
            best_kernel_t = []
            for ig in range(self.ng_demo_dict[t]):
                bk = np.where(mll_est_t[ig,:] == np.amin(mll_est_t[ig,:]))[0][0]
                best_kernel_t.append(bk)

            # save
            print(t +' #sig='+str(np.sum(pv_fdr_t<=self.sig_cutoff)))
            self.pv_raw_tl[t] = pv_t
            self.ts_tl[t] = ts_t
            self.pv_cauchy_tl[t] = pv_cauchy_t
            self.pv_fdr_tl[t] = pv_fdr_t
            self.best_kernel_tl[t] = best_kernel_t

        # save pv_fdr_tl as pkl
        output_dir = 'output'
        outfile = 'output/pv_est.pkl'
        pickle_dict = {}
        pickle_dict['pv_fdr_dict'] = self.pv_fdr_tl

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(outfile, 'wb') as f:
            pickle.dump(pickle_dict, f)
            

    def pattern_clustering(self, K_max=10, exclude_nonsig=True):
        '''
        Function doing k-means clustering of lam est
        Args:
            K_max: max number of clustering being examined; default=10
            exclude_nonsig: exclude the lam of nonsignificant genes; default=True
        '''
        # extract 21 lam values
        x_points_ = np.arange(0,100,5)
        x_points = np.append(x_points_, 99)
        
        sig_lam_points_std_dict = {}
        sig_lam_dict = {}
        sig_gl_dict = {}
        n_sig = []

        for t in self.type_list:
            if exclude_nonsig==True:
                pv_t = self.pv_fdr_tl[t]
            else:
                pv_t = np.full(len(self.pv_fdr_tl[t]), 0.0)
                
            lam_t = np.array(self.weighted_lam_est[t])
            gl_t = np.array(self.gene_list_dict[t])

            sig_lam_t = lam_t[pv_t<=self.sig_cutoff,:]
            sig_gl_t = gl_t[pv_t<=self.sig_cutoff]

            lam_points_t = sig_lam_t[:,x_points]
            lam_points_std_t = np.zeros(lam_points_t.shape)
            for j in range(lam_points_t.shape[0]):
                lam_j = lam_points_t[j,:]
                lam_points_std_t[j,:] = (lam_j-np.min(lam_j))/(np.max(lam_j)-np.min(lam_j)) # min-max std

            sig_lam_points_std_dict[t] = lam_points_std_t
            sig_lam_dict[t] = sig_lam_t
            sig_gl_dict[t] = sig_gl_t
            n_sig.append(len(sig_gl_t))        
        
        # concatenate sig std lam (21 values) across cell types
        sig_lam_points_std_merged = np.concatenate(list(sig_lam_points_std_dict.values()), axis=0)
        # concatenate sig lam
        sig_lam_merged = np.concatenate(list(sig_lam_dict.values()), axis=0)
        
        # compute the 1st derivative of lam std
        sig_lam_points_std_merged_dev = sig_lam_points_std_merged[:,1:] - sig_lam_points_std_merged[:,:-1]
        # min-max std
        min_ = np.min(sig_lam_points_std_merged_dev, axis=1)
        max_ = np.max(sig_lam_points_std_merged_dev, axis=1)
        min_mat = np.tile(min_, (sig_lam_points_std_merged_dev.shape[1],1)).transpose()
        max_mat = np.tile(max_, (sig_lam_points_std_merged_dev.shape[1],1)).transpose()
        sig_lam_points_std_merged_dev_std = (sig_lam_points_std_merged_dev-min_mat)/(max_mat-min_mat)
        
        # kmeans clustering
        print(f'The max number of clusters K={K_max}')
        random.seed(2024)
        kmeans_res = {}
        X = np.concatenate((sig_lam_points_std_merged_dev_std, sig_lam_points_std_merged), axis=1)

        output_K = {}
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        for k in range(1, K_max+1):
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=5)
            pred_y = kmeans.fit_predict(X)
            output_K['pred'+str(k)] = pred_y
            output_K['center'+str(k)] = kmeans.cluster_centers_
            # Elbow
            distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_,
                                                'euclidean'), axis=1)) / X.shape[0])
            inertias.append(kmeans.inertia_)
            mapping1[k] = sum(np.min(cdist(X, kmeans.cluster_centers_,
                                           'euclidean'), axis=1)) / X.shape[0]
            mapping2[k] = kmeans.inertia_

        # viz distortion and inertias
        nr = 1
        nc = 2
        ss_nc = 3
        ss_nr = 2
        fig = plt.figure(figsize=(nc*ss_nc, nr*ss_nr))
        gs = fig.add_gridspec(nr,nc,
                              width_ratios=[1]*nc,
                              height_ratios=[1]*nr)
        gs.update(wspace=0.3, hspace=0.3)

        # distortion
        ax = plt.subplot(gs[0,0])
        x = np.arange(K_max)+1
        ax.plot(x, distortions, 'o-', color='black')
        ax.set_title('Distortion')
        __ = ax.set_xticks(x,x)

        # inertia
        ax = plt.subplot(gs[0,1])
        x = np.arange(K_max)+1
        ax.plot(x, inertias, 'o-', color='black')
        ax.set_title('Inertia')
        __ = ax.set_xticks(x,x)
        
        self.kmeans_K_max = output_K
        self.kmeans_sig_lam_points_std_merged = sig_lam_points_std_merged
        self.kmeans_exclude_nonsig = exclude_nonsig
        self.kmeans_n_sig = n_sig
        
        
    def pattern_labeling(self, K):
        '''
        Function doing k-means clustering of lam est and assign cluster labels
        Args:
            K: number of clusters, manually input
        '''
        K_opt = K
        # re-orgnize cluster labels based on "peak" from 0 to 1
        cluster_label_old = self.kmeans_K_max['pred'+str(K_opt)]
        center_argmax = np.zeros(K_opt)
        for k in range(K_opt):
            center_argmax[k] = np.median(np.argmax(self.kmeans_sig_lam_points_std_merged[cluster_label_old==k,:], axis=1))

        # build a dict of key=old label val=new label
        order = center_argmax.argsort()
        ranks = order.argsort()
        update_label_dict = dict(zip(np.arange(K_opt), ranks))

        # update the labels
        cluster_label = np.zeros_like(cluster_label_old)
        for k in update_label_dict.keys():
            cluster_label[cluster_label_old==k] = update_label_dict.get(k)
        
        unique_values, counts = np.unique(cluster_label, return_counts=True)
        for value, count in zip(unique_values, counts):
            print(f"Pattern {value+1}: {count} genes")
            
        # create dictionary of clustering labels
        num_sig = []
        for t in self.type_list: # note, types should in the same order    
            if self.kmeans_exclude_nonsig==True:
                pv_t = self.pv_fdr_tl[t]
            else:
                pv_t = np.full(len(self.pv_fdr_tl[t]), 0.0)
            
            num_sig.append(np.sum(pv_t<=self.sig_cutoff))

        label_dict = {}
        for it, t in enumerate(self.type_list): # note, types should in the same order
            idx1 = int(np.sum(self.kmeans_n_sig[:it]))
            idx2 = idx1 + self.kmeans_n_sig[it]
            if self.kmeans_exclude_nonsig==True:
                pv_t = self.pv_fdr_tl[t]
            else:
                pv_t = np.full(len(self.pv_fdr_tl[t]), 0.0)
            label_t = np.full(len(pv_t), -1)
            cluster_label_t = cluster_label[idx1:idx2]
            label_t[pv_t<=self.sig_cutoff] = cluster_label_t
            label_dict[t] = label_t
            
        self.labels_dict = label_dict
        
        # plots
        red = '#d96256'
        lightorange = '#fabc2e'
        lightgreen = '#93c572'
        lightblue = '#5d8aa8'
        darkblue = '#284d88'
        purple = '#856088'
        pink = '#F25278'
        lightgreen2 = '#32CD32'
        lightblue2 = '#189AB4'
        darkorange2 = '#FA8128'
        colors = [red, lightorange, lightgreen, lightblue, darkblue, purple, pink, lightgreen2, lightblue2, darkorange2]
        
        # barplot number(%) of genes
        labels_all = np.concatenate(list(self.labels_dict.values()))
        labels_cluster, labels_num = np.unique(labels_all, return_counts=True)

        bar_width = 0.5
        bar_positions = [0, 1]
        nr = 1
        nc = 1
        ss_nr = 0.5
        ss_nc = 3
        fig = plt.figure(figsize=(nc*ss_nc, nr*ss_nr), dpi=200)
        gs = fig.add_gridspec(nr, nc,
                              width_ratios=[1]*nc,
                              height_ratios=[1]*nr)
        gs.update(wspace=0.0, hspace=0.0)
        ax = plt.subplot(gs[0, 0])

        # sig genes
        labels_num_sig = labels_num[labels_cluster!=-1]
        left = 0
        for i in range(K):
            ax.barh(bar_positions[0], labels_num_sig[i], bar_width, left=left, color=colors[i])
            left += labels_num_sig[i]
            ax.text(left - labels_num_sig[i]/2, bar_positions[0], f'{labels_num_sig[i]}\n({labels_num_sig[i]/np.sum(labels_num_sig)*100:.0f}%)', fontsize=5, ha='center', va='center', color='black')

        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(f'Number of genes')
        ax.set_yticks([])
        # plt.savefig(f'output/fig_labels_all_bar.pdf', dpi=300, bbox_inches='tight')
        
        # lam est plot
        x = np.linspace(0, 1, 100)

        nr = 1
        nc = 1
        ss_nc = 3
        ss_nr = 1.5
        fig = plt.figure(figsize=(nc*ss_nc, nr*ss_nr), dpi=200)
        gs = fig.add_gridspec(nr,nc,
                              width_ratios=[1]*nc,
                              height_ratios=[1]*nr)
        gs.update(wspace=0.0, hspace=0.0)
        ax = plt.subplot(gs[0, 0])

        for t in self.type_list:
            label_t = self.labels_dict[t]
            lam_t = self.weighted_lam_est[t]
            gene_t = self.gene_list_dict[t]
            for j in range(len(gene_t)):
                l = label_t[j]
                if l>-1:
                    color = colors[l]
                    alpha = 0.5
                    zorder = 1
                    lam_j = lam_t[j]
                    lam_std_j = (lam_j-np.min(lam_j))/(np.max(lam_j)-np.min(lam_j)) # min-max std
                    ax.plot(x, lam_std_j, alpha=alpha, lw=1.2, color=color, zorder=zorder)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.04, 1.04)
        __ = ax.set_xticks([0,0.5,1], [0,0.5,1])
        __ = ax.set_yticks([0,0.5,1], [0,0.5,1])
        ax.set_xlabel('Relative postion')
        ax.set_ylabel('Intensity')
        #plt.savefig(f'output/fig_lam_all.pdf', dpi=300, bbox_inches='tight')
        
        # pattern scores event plot
        scores_all = np.concatenate(list(self.scores.values()))
        labels_all = np.concatenate(list(self.labels_dict.values()))
        scores_all_sig = scores_all[labels_all>-1]
        labels_all_sig = labels_all[labels_all>-1]

        nr = 1
        nc = 1
        ss_nc = 3
        ss_nr = 1
        fig = plt.figure(figsize=(nc*ss_nc, nr*ss_nr), dpi=200)
        gs = fig.add_gridspec(nr,nc,
                              width_ratios=[1]*nc,
                              height_ratios=[1]*nr)
        gs.update(wspace=0.0, hspace=0.0)
        ax = plt.subplot(gs[0, 0])

        alpha=0.8
        ll=1
        lw=0.8

        for k in range(K):
            scores_k = scores_all_sig[labels_all_sig==k]
            ax.eventplot([scores_k + np.random.uniform(0, 0.05, len(scores_k))], 
                         orientation='horizontal', 
                         colors=[colors[k]],
                         alpha=alpha,
                         linelengths=ll, lw=lw, lineoffsets=k)

        ax.set_xlim(-0.02, 1.02)
        __ = ax.set_yticks(np.arange(K), np.arange(K)+1)
        __ = ax.set_xticks([0,0.5,1], [0,0.5,1])
        ax.set_xlabel('Relative position')
        ax.set_ylabel('Clusters')
        #plt.savefig(f'output/fig_score_event.pdf', dpi=300, bbox_inches='tight')

