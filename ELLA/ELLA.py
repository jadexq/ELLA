import pickle
import timeit
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

# need to taken care of this later!!!
pd.options.mode.chained_assignment = None  # default='warn'

class model_beta(torch.nn.Module):
    '''
    NHPP torch model, log likelihood of alternative models with beta kernels
    normalized kernels
    '''
    def __init__(self, init_A, init_B):
        super().__init__()
        self.A = torch.nn.Parameter(torch.rand(())+init_A)
        self.B = torch.nn.Parameter(torch.rand(())+init_B)

    def forward(self, ri, c0i, ni, a, b, ri_clamp_min, ri_clamp_max):
        '''
        subject-wise log-likelihood
        '''
        BB = math.gamma(a[0])*math.gamma(b[0]) / math.gamma(a[0]+b[0])

        #ri_clamp = torch.clamp(ri, min=0.01, max=0.99)
        #ri_clamp = torch.clamp(ri, min=0.01, max=1.0)
        ri_clamp = torch.clamp(ri, min=ri_clamp_min, max=ri_clamp_max)

        # constrain A=exp(A)， B=exp(B)
        lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.exp(self.A)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + torch.exp(self.B)*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.exp(self.A)*a)/(a+b)+(torch.exp(self.B))/(2.0))

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
        #ri_clamp = torch.clamp(ri, min=0.01, max=0.99)
        #ri_clamp = torch.clamp(ri, min=0.01, max=1.0)
        ri_clamp = torch.clamp(ri, min=ri_clamp_min, max=ri_clamp_max)

        #lli = torch.log(c0i) + torch.log(torch.clamp(self.B, min=0)*ri) - (c0i/ni)*(torch.clamp(self.B, min=0)/2.0) # clamp(B)
        #lli = torch.log(c0i) + torch.log(torch.exp(self.B)*ri) - (c0i/ni)*(torch.exp(self.B)/2.0) # exp(B)
        #lli = torch.log(c0i) + torch.log(self.B*self.B*ri) - (c0i/ni)*(self.B*self.B/2.0) # B^2
        #lli = torch.log(c0i) + torch.log(F.relu(self.B)*ri) - (c0i/ni)*(F.relu(self.B)/2.0) # relu B
        lli = torch.log(c0i*2.0*math.pi)+torch.log(self.B*ri_clamp)-(c0i*2.0*math.pi/ni)*(self.B/2.0) # B
        
        return lli


class history_model_beta(torch.nn.Module):
    '''
    NHPP torch model, log likelihood of alternative models with beta kernels
    '''
    def __init__(self, init_A, init_B):
        super().__init__()
        self.A = torch.nn.Parameter(torch.rand(())+init_A)
        self.B = torch.nn.Parameter(torch.rand(())+init_B)

    def forward(self, ri, c0i, ni, a, b):
        '''
        subject-wise log-likelihood
        '''
        BB = math.gamma(a[0])*math.gamma(b[0]) / math.gamma(a[0]+b[0])

        ri_clamp = torch.clamp(ri, min=0.01, max=0.99)

        # constrain A=exp(A)，clamp(B)
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.exp(self.A)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + torch.clamp(self.B, min=0)*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.exp(self.A)*a)/(a+b)+(torch.clamp(self.B, min=0))/(2.0))

        # constrain A=exp(A)
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.exp(self.A)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.exp(self.A)*a)/(a+b)+(self.B)/(2.0))

        # constrain A=exp(A), kernel max=1
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.exp(self.A)*(1/maxk)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.exp(self.A)*(1/maxk)*a)/(a+b)+(self.B)/(2.0))

        # constrain A=A^2
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((self.A*self.A/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*ri_clamp) - (c0i*2.0*math.pi/ni)*((self.A*self.A*a)/(a+b)+(self.B)/(2.0))

        # constrain A=clamp(A)
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.clamp(self.A, min=0)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.clamp(self.A, min=0)*a)/(a+b)+(self.B)/(2.0))

        # constrain A=exp(A)
        # lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.exp(self.A)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.exp(self.A)*a)/(a+b)+(self.B)/(2.0))

        # constrain A=exp(A), B=exp(B)
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.exp(self.A)/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + torch.exp(self.B)*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.exp(self.A)*a)/(a+b)+(torch.exp(self.B))/(2.0))

        # constrain A=A^2, B=B^2
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((self.A*self.A/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*self.B*ri_clamp) - (c0i*2.0*math.pi/ni)*((self.A*self.A*a)/(a+b)+(self.B*self.B)/(2.0))

        # constrain A=A^2, clamp lam
#         lli = torch.log(c0i*2.0*math.pi) + torch.log(torch.clamp((self.A*self.A/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*ri_clamp, min=1e-8)) - (c0i*2.0*math.pi/ni)*torch.clamp(((self.A*self.A*a)/(a+b)+(self.B)/(2.0)), min=0) # A, B, add contrain to make Lambda^*_i(1)>=0, lambda*>0, clamp ri to avoid na A_est/loss

        # constrain A>0, B>0
#         lli = torch.log(c0i*2.0*math.pi) + torch.log((torch.clamp(self.A, min=0)/BB)*torch.pow(ri_clamp, a)*torch.pow(1-ri_clamp, b-1) + torch.clamp(self.B, min=0)*ri_clamp) - (c0i*2.0*math.pi/ni)*((torch.clamp(self.A, min=0)*a)/(a+b)+torch.clamp(self.B, min=0)/(2.0))

#         lli = torch.log(c0i*2.0*math.pi) + torch.log(torch.clamp((self.A/BB)*torch.pow(ri_clamp,a)*torch.pow(1-ri_clamp,b-1) + self.B*ri_clamp, min=1e-8)) - (c0i*2.0*math.pi/ni)*torch.clamp(((self.A*a)/(a+b)+(self.B)/(2.0)), min=0) # A, B, add contrain to make Lambda^*_i(1)>=0, lambda*>0, clamp ri to avoid na A_est/loss

#         lli = torch.log(c0i*2.0*math.pi) + torch.log(torch.clamp((self.A/BB)*torch.pow(ri,a)*torch.pow(1-ri,b-1) + self.B*ri, min=1e-8)) - (c0i*2.0*math.pi/ni)*torch.clamp(((self.A*a)/(a+b)+(self.B)/(2.0)), min=0) # A, B, add contrain to make Lambda^*_i(1)>=0, lambda*>0

        #lli = torch.log(c0i) + torch.log((self.A/BB)*torch.pow(ri,a)*torch.pow(1-ri,b-1) + self.B*ri) - (c0i/ni)*torch.clamp(((self.A*a)/(a+b)+(self.B)/(2.0)), min=0) # A, B, add contrain to make Lambda^*_i(1)>=0
        #lli = torch.log(c0i) + torch.log((self.A/BB)*torch.pow(ri,a)*torch.pow(1-ri,b-1) + self.B*ri) - (c0i*self.A*a)/(ni*(a+b)) - (c0i*self.B)/(2.0*ni) # A, B
        #lli = torch.log(c0i) + torch.log((self.A*self.A/BB)*torch.pow(ri,a)*torch.pow(1-ri,b-1) + self.B*self.B*ri) - (c0i*self.A*self.A*a)/(ni*(a+b)) - (c0i*self.B*self.B)/(2.0*ni) # A^2, B^2
        #lli = torch.log(c0i) + torch.log((self.A*self.A/BB)*torch.pow(ri,a)*torch.pow(1-ri,b-1) + self.B*ri) - (c0i*self.A*self.A*a)/(ni*(a+b)) - (c0i*self.B)/(2.0*ni) # A^2, B
        #lli = torch.log(c0i) + torch.log((F.relu(self.A)/BB)*torch.pow(ri,a)*torch.pow(1-ri,b-1) + F.relu(self.B)*ri) - (c0i*F.relu(self.A)*a)/(ni*(a+b)) - (c0i*F.relu(self.B))/(2.0*ni) # relu A, B

        return lli


def loss_ll(pred):
    '''
    NHPP torch model, negative log likelihood as loss, for minimazing
    '''
    loss = - torch.sum(pred)
    return loss

class ELLA:
    '''
    Class of EG analysis
    '''
    def __init__(self, dataset: str, beta_kernel_param_list=None, adam_learning_rate_max=None, adam_learning_rate_min=None, adam_learning_rate_adjust=None, adam_delta_loss_max=None, adam_delta_loss_min=None, adam_delta_loss_adjust=None, adam_niter_loss_unchange=None, max_iter=None, min_iter=None, max_ntanbin=None, ri_clamp_min=None, ri_clamp_max=None, max_workers=None):
        '''
        Constructor
        Args:
            dateset, name of the dataset: seqscope, stereoseq, merfish, seqfish
                or in general seq or fish
            adam_learning_rate_max: initial LR of Adam, also the max, default=0.01
            adam_learning_rate_min: min LR of Adam, default=0.001
            adam_learning_rate_adjust: LR = maxll/adam_learning_rate_adjust, default 1e7
        '''
        # dataset name
        self.dataset = dataset

        # tech
        self.tech_list_dict = {}
        self.tech_list_dict['seq'] = ['seq', 'seqscope', 'stereoseq', 'merfish_seq', 'seqfish_seq']
        self.tech_list_dict['fish'] = ['fish', 'merfish', 'seqfish']

        # paths
        self.data_path = 'input/'+dataset+'_data_dict.pkl'

        # constants
        self.epsilon = 1e-10 # a very small value, to prevent division by zero etc.
        # self.c0 = 1e-8 # term added to the denominator to improve numerical stability
        # self.c1 = 0.001 # a small constant, min radius
        self.nt: int # number of cell types
        self.nk: int # number of default kernels
        self.nc_all: int # total number of cells

        # ntanbin related const
        self.max_ntanbin = 25 # default and max ntanbin for pi/2 (each quantrant)
        if max_ntanbin is not None:
            self.max_ntanbin = max_ntanbin
        self.ri_clamp_min = 1e-10
        if ri_clamp_min is not None:
            self.ri_clamp_min = ri_clamp_min
        self.ri_clamp_max = 1.0
        if ri_clamp_min is not None:
            self.ri_clamp_max = ri_clamp_max
        self.nc4ntanbin = 50 # number of cells (replacement allowed) for computing ntanbin
        self.min_bp = 5 # min boundary points in each tanbin
        self.high_res = 200 # x/yrange so as to be high resolution
        self.min_ntanbin_error = 3 # if ntanbin less than this, will print a message

        # list/dict/df
        self.ng_dict = {} # number of genes in each cell type
        self.ng_demo_dict = {} # number of genes in each cell type, given a small number for testing
        self.nc_dict = {} # number of cells in each cell type
        self.type_list = [] # cell type
        self.cell_list_all = [] # list of all cells
        self.gene_list_dict = {} # gene list of each cell type
        self.gene_list_ncavl_dict = {} # number of cells avl corresponding to gene list
        self.cell_list_dict = {} # cell list of each cell type
        self.beta_kernel_param_list = [] # kernel param list
        self.cell_mask_df: pandas.DataFrame = None # df of cell masks
        self.data_df: pandas.DataFrame = None # df of gene expression data
        self.r_tl = {} # for nhpp fit, radius
        self.c0_tl = {} # for nhpp fit, read depth
        self.c0_tl_homo = {} # for nhpp fit, homo case calculation, no repeat wrt genes
        self.n_tl = {} # for nhpp fit, total umi count of a gene
        self.n_tl_homo = {} # for nhpp fit, homo case calculation, no repeat wrt genes
        self.A_est = {} # for nhpp fit, A est
        self.B_est = {} # for nhpp fit, B est
        self.mll_est = {} # for nhpp fit, max log likelihood
        self.pv_raw_tl = {} # for nhpp result, raw pv
        self.ts_tl = {} # for nhpp result, likelihood ratio test statistic value
        self.pv_cauchy_tl = {} # for nhpp result, cauchy combined pv
        self.pv_fdr_tl = {} # for nhpp result, fdr-BY pv
        self.best_kernel_tl = {} # for nhpp result, best kernel
        self.weight_ml = {} # for nhpp results, weights calculated from max likelihood
        self.scores = {} # for nhpp results, scores
        self.weighted_lam_est = {} # for nhpp results, model averaging lam corresponding to all sig kernels
        self.labels = {} # for nhpp results, labels derived from scores
        self.ntanbin_dict = {} # ntanbin for each cell type, related to resolution
        if self.dataset in self.tech_list_dict['fish']:
            self.nc_ratio_dict = {} # fish data only, avg nc ratio of each cell type
            self.nuclear_mask_df: pandas.DataFrame = None # fish data only, df of nuclear masks

        # colors
        self.red = '#c0362c'
        self.lightgreen = '#93c572'
        self.darkgreen = '#4c9141'
        self.lightblue = '#5d8aa8'
        self.darkblue = '#2e589c'
        self.white = '#fafafa'
        self.lightgray = '#d3d3d3'
        self.darkgray ='#545454'
        self.lightorange = '#fabc2e'
        self.darkorange = '#fb9912'
        self.yellow = '#e4d00a'
        self.color_array = np.zeros([10,4])
        self.color_array[:,0] = 147/255
        self.color_array[:,1] = 197/255
        self.color_array[:,2] = 114/255
        self.color_array[:,-1] = np.concatenate((np.linspace(0.0,0.5, 4),
                                                 np.linspace(0.5,1.0, 6)))
        self.green_alpha_object = colors.LinearSegmentedColormap.from_list(name='green_alpha', colors=self.color_array)

        # NHPP fit constants
        self.optimizer = 'adam' # 'adam or sgd'
        self.learning_rate_initial = 0.01 # initial learning rate also the max LR, default 0.01
        self.max_workers = max_workers
        if adam_learning_rate_max is not None:
            self.learning_rate_max = adam_learning_rate_max
        self.learning_rate_min = 0.001 # min learning rate, default=1e-3
        if adam_learning_rate_min is not None:
            self.learning_rate_min = adam_learning_rate_min
        self.learning_rate_adjust = 1e7 # adaptive LR = maxll/learning_rate_adjust, default=1e7
        if adam_learning_rate_adjust is not None:
            self.learning_rate_adjust = adam_learning_rate_adjust
        self.max_iter = 5000
        self.min_iter = 100
        if max_iter is not None:
            self.max_iter = max_iter # max Adam iterations
        if min_iter is not None:
            self.min_iter = min_iter # min Adam iterations
        self.delta_loss_max = 0.001 # early stopping criterion 1 max, will be adjust by homo maxll
        self.delta_loss_min = 0.0001 # early stopping criterion 1 min
        if adam_delta_loss_max is not None:
             self.delta_loss_max = adam_delta_loss_max
        if adam_delta_loss_min is not None:
             self.delta_loss_min = adam_delta_loss_min
        self.niter_loss_unchange = 20 # early stopping criterion 2
        self.delta_loss_adjust = 1e8 # delta_loss = maxll/delta_loss_adjust
        if adam_delta_loss_adjust is not None:
            self.delta_loss_adjust = adam_delta_loss_adjust
        if adam_niter_loss_unchange is not None:
            self.niter_loss_unchange = adam_niter_loss_unchange
        # NHPP intermediate results
        self.initial_A_dict = {} # Adam A initial value
        self.initial_B_dict = {} # Adam B initial value
        self.loss_nhpp_dict = {} # save nhpp loss function values for plotting
        self.loss_hpp_dict = {} # save hpp loss function values for plotting
        self.early_stop_dict = {} # save if Adam early stop step
        self.adaptive_learning_rate_dict = {} # save Adam adaptive learning rate
        self.adaptive_delta_loss_dict = {} # save Adam adaptive delta loss

        # results related constants
        self.sig_cutoff = 0.05
        self.n_overflow = 500 # if larger than this exp will be overflow
        
        if beta_kernel_param_list is None:
            # default kernels if not specified
            self.beta_kernel_param_list = [
                [1,     2.71],   # 0      2.71 --- kurt -0.1
                [1.26,  3.34],   # 0.1    2.27
                [2.05,  5.19],   # 0.2    2.50
                [6.99,  14.98],  # 0.3    4.02 
                [19.41, 28.62],  # 0.4    5.61
                [28.5,  28.5],   # 0.5    6
                [28.62, 19.41],  # 0.6    5.61
                [14.98, 6.99],   # 0.7    4.02 
                [5.19,  2.05],   # 0.8    2.50
                [3.34,  1.26],   # 0.9    2.27
                [2.71,  1],      # 1.0    2.71
                [1,    2],       # 0      2    --- kurt -0.6
                [1.13, 2.19],    # 0.1    1.74
                [1.38, 2.52],    # 0.2    1.71   
                [1.88, 3.06],    # 0.3    1.81
                [2.73, 3.60],    # 0.4    1.96
                [3.5,  3.5],     # 0.5    2.04
                [3.60, 2.73],    # 0.6    1.96
                [3.06, 1.88],    # 0.7    1.81    
                [2.52, 1.38],    # 0.8    1.71   
                [2.19, 1.13],    # 0.9    1.74
                [2,    1],       # 1.0    2
                # [1, 4],     # --- k=9
                # [1, 2],
                # [2, 5],
                # [2, 3],
                # [2, 2],
                # [3, 2],
                # [5, 2],
                # [2, 1],
                # [4, 1],
                # [1, 4],     # --- s=4
                # [4, 13],
                # [8, 15],
                # [13, 13],
                # [15, 8],
                # [13, 4],
                # [4, 1],
                # [1, 2],     # --- s=2
                # [1.6, 3.5],
                # [2.5, 4],
                # [3.5, 3.5],
                # [4, 2.5],
                # [3.5, 1.6],
                # [2, 1],
            ]
        if beta_kernel_param_list is not None:
            self.beta_kernel_param_list = beta_kernel_param_list

    def load_data(self, data_dict=None, data_path=None):
        '''
        Function of loading data
        Args:
            data_dict: default load .pkl from self.data_path, if given, use the given data_dict
            data_path: default 'input/'+dataset+'_data_dict.pkl', if given, load from given path
            beta_kernel_param_list: default 8 kernels if not specified
        Returns:
            None
        '''
        if data_dict is None:
            if data_path is None:
                # load prepared data from default path
                with open(self.data_path, 'rb') as f:
                    data_dict = pickle.load(f)
            if data_path is not None:
                # load prepared data from given path
                with open(data_path, 'rb') as f:
                    data_dict = pickle.load(f)

        # cell type list
        self.type_list = data_dict['types']
        # topN genes lists for each type
        self.gene_list_dict = data_dict['genes']
        # cell list
        self.cell_list_dict = data_dict['cells']
        # list of all cells
        self.cell_list_all = data_dict['cells_all']
        # mask of cells
        self.cell_mask_df = data_dict['cell_seg']
        # gene expression data
        self.data_df = data_dict['expr']
        # ntanbin for each cell type -- set to default value, can specify later
        # self.ntanbin_dict = data_dict['ntanbin_dict']
        for t in self.type_list:
            self.ntanbin_dict[t] = self.max_ntanbin
        # fish data only
        if self.dataset in self.tech_list_dict['fish']:
            self.nc_ratio_dict = data_dict['nc_ratio_dict']
            self.nuclear_mask_df = data_dict['nuclear_mask_df']

        # update global constants
        # number of cell types
        self.nt = len(self.type_list)
        # number of all cells
        self.nc_all = len(self.cell_list_all)
        # number of genes
        ng_list = []
        for t in self.type_list:
            self.ng_dict[t] = len(self.gene_list_dict[t])
            ng_list.append(len(self.gene_list_dict[t]))
        nc_list = []
        # number of cells
        for t in self.type_list:
            self.nc_dict[t] = len(self.cell_list_dict[t])
            nc_list.append(len(self.cell_list_dict[t]))
        # number of default kernels
        self.nk = len(self.beta_kernel_param_list)

        # print
        print(f"Number of cell types {self.nt}")
        print(f"Average number of genes {np.mean(ng_list)}")
        print(f"Average number of cells {np.mean(nc_list)}")
        print(f"Number of default kernels {self.nk}")

    def specify_ntanbin_old(self, input_ntanbin_dict=None):
        '''
        Function of specifying ntanbin
        Args:
            ntanbin: give ntanbin manually, a dictionary, keys corresponding to cell type list
        Returns:
            None: (update self.ntanbin_dict)
        '''
        if input_ntanbin_dict is not None:
            # use the manually entered
            self.ntanbin_dict = input_ntanbin_dict

        if input_ntanbin_dict is None:
            # compute ntanbin for each cell type:
            for t in self.type_list:
                # specify ntanbin_gen based on cell_mask_df
                # random sample self.nc4ntanbin cells, allow replace
                cell_list_sampled = np.random.choice(self.cell_list_dict[t], self.nc4ntanbin, replace=True)
                cell_mask_df_sampled = self.cell_mask_df[self.cell_mask_df.cell.isin(cell_list_sampled)]
                # compute the xrange and y range of these sampled cells
                xrange_sampled = []
                yrange_sampled = []
                for c in cell_list_sampled:
                    mask_c = cell_mask_df_sampled[cell_mask_df_sampled.cell==c]
                    xmax = mask_c.x.max()
                    xmin = mask_c.x.min()
                    ymax = mask_c.y.max()
                    ymin = mask_c.y.min()
                    xrange_sampled.append(xmax-xmin)
                    yrange_sampled.append(ymax-ymin)

                # specify ntanbin for pi/2 (a quantrant)
                # if resolution is super high
                if np.mean(xrange_sampled)>self.high_res and np.mean(yrange_sampled)>self.high_res:
                    ntanbin=self.max_ntanbin
                # if resolution is not super high
                else:
                    # require at least self.min_bp boundary points in each tanbin
                    theta = 2*np.arctan(self.min_bp/np.mean(xrange_sampled+yrange_sampled))
                    ntanbin_ = (pi/2)/theta
                    ntanbin = np.ceil(ntanbin_)
                    if ntanbin<self.min_ntanbin_error:
                        print(f'Cell type {t} failed, resolution not high enougth to support the analysis')
                # asign
                self.ntanbin_dict[t] = ntanbin


    def specify_ntanbin(self, input_ntanbin_dict=None):
        '''
        Function of specifying ntanbin
        Args:
            ntanbin: give ntanbin manually, a dictionary, keys corresponding to cell type list
        Returns:
            None: (update self.ntanbin_dict)

        Note: updated this function bc
            1. ntanbin should be calculated based on mask
            2. should use the resolution of mask (num of unique x/y coords) to compute
        '''
        if input_ntanbin_dict is not None:
            # use the manually entered
            self.ntanbin_dict = input_ntanbin_dict

        if input_ntanbin_dict is None:
            # compute ntanbin for each cell type:
            for t in self.type_list:
                # specify ntanbin_gen based on cell_mask_df
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


    def plot_number_of_cells(self):
        '''
        Function of plotting number of cells in each cell type
        Args:
            None
        Returns:
            None
        '''
        df1 = self.data_df.groupby('cell').first().copy()
        df2 = df1.type.value_counts().sort_index()
        fig, ax = plt.subplots(1,1,figsize=(3, 3))
        df2.plot(ax = ax, kind='bar', stacked=False, color=self.lightblue)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Number of cells of each type')
        try:
            fig.savefig('output/plot_number_of_cells.png')
        except:
            pass

    def register_cells_slow(self, nc_demo=None):
        '''
        Function of registering cells to unit circle
        Args:
            demo_nc:
                default - register all cells
                or register given number of cells for testing
        Returns:
            None
            (update global variable `df_registered`)
            (save df_registered as .pkl)
        '''
        if nc_demo is None:
            nc_demo = self.nc_all

        start = timeit.default_timer()

        dict_registered = {}

        df = self.data_df.copy() # cp original data
        df_gbC = df.groupby('cell') # group by `cell`

        for ic, c in enumerate(self.cell_list_all[:nc_demo]): # loop over all cells
            if ic % int(len(self.cell_list_all)/10+1) == 0:
                print(f'{np.round(ic/len(self.cell_list_all)*100,1)} % cell {c}')

            df_c = df_gbC.get_group(c).copy() # df for cell c
            t = df_c.type.iloc[0] # cell type for cell c
            mask_df_c = self.cell_mask_df[self.cell_mask_df.cell == c].copy() # get the mask df for cell c
            center_c = [int(df_c.centerX.iloc[0]), int(df_c.centerY.iloc[0])] # nuclear center of cell c
            if self.dataset in self.tech_list_dict['fish']: # for fish data only
                nmask_df_c = self.nuclear_mask_df[self.nuclear_mask_df.cell == c].copy() # get the nuclear mask df for cell c
            #tanbin = np.round(np.linspace(0, 3.1416/2, self.ntanbin_dict[t]), 5)
            tanbin = np.linspace(0, pi/2, self.ntanbin_dict[t])

            # add centered coord and ratio=y/x for df_c and mask_df_c
            df_c['x_c'] = df_c.x.copy() - center_c[0]
            df_c['y_c'] = df_c.y.copy() - center_c[1]
            df_c['d_c'] = (df_c.x_c.copy()**2+df_c.y_c.copy()**2)**0.5
            df_c['arctan'] = np.absolute(np.arctan(df_c.y_c / (df_c.x_c+self.epsilon)))
            mask_df_c['x_c'] = mask_df_c.x.copy() - center_c[0]
            mask_df_c['y_c'] = mask_df_c.y.copy() - center_c[1]
            mask_df_c['d_c'] = (mask_df_c.x_c.copy()**2+mask_df_c.y_c.copy()**2)**0.5
            mask_df_c['arctan'] = np.absolute(np.arctan(mask_df_c.y_c / (mask_df_c.x_c+self.epsilon)))
            if self.dataset in self.tech_list_dict['fish']: # for fish data only
                nmask_df_c['x_c'] = nmask_df_c.x.copy() - center_c[0]
                nmask_df_c['y_c'] = nmask_df_c.y.copy() - center_c[1]
                nmask_df_c['d_c'] = (nmask_df_c.x_c.copy()**2+nmask_df_c.y_c.copy()**2)**0.5
                nmask_df_c['arctan'] = np.absolute(np.arctan(nmask_df_c.y_c / (nmask_df_c.x_c+self.epsilon)))

            # in each quatrant, find dismax_c for each tanbin interval using mask_df_c
            dismax_c = np.zeros((self.ntanbin_dict[t]-1, 4))
            # df in quatrants
            mask_df_c_quatrant1 = mask_df_c[(mask_df_c.x_c>=0) & (mask_df_c.y_c>=0)].copy()
            mask_df_c_quatrant2 = mask_df_c[(mask_df_c.x_c<=0) & (mask_df_c.y_c>=0)].copy()
            mask_df_c_quatrant3 = mask_df_c[(mask_df_c.x_c<=0) & (mask_df_c.y_c<=0)].copy()
            mask_df_c_quatrant4 = mask_df_c[(mask_df_c.x_c>=0) & (mask_df_c.y_c<=0)].copy()
            if self.dataset in self.tech_list_dict['fish']: # for fish data only
                dismax_n = np.zeros((self.ntanbin_dict[t]-1, 4))
                nmask_df_c_quatrant1 = nmask_df_c[(nmask_df_c.x_c>=0) & (nmask_df_c.y_c>=0)].copy()
                nmask_df_c_quatrant2 = nmask_df_c[(nmask_df_c.x_c<=0) & (nmask_df_c.y_c>=0)].copy()
                nmask_df_c_quatrant3 = nmask_df_c[(nmask_df_c.x_c<=0) & (nmask_df_c.y_c<=0)].copy()
                nmask_df_c_quatrant4 = nmask_df_c[(nmask_df_c.x_c>=0) & (nmask_df_c.y_c<=0)].copy()
            # find dismax
            for j in range(self.ntanbin_dict[t]-1):
                itv1 = tanbin[j]
                itv2 = tanbin[j+1]
                # quatrant1
                mask_df_c_quatrant1_itv = mask_df_c_quatrant1[(mask_df_c_quatrant1.arctan>=itv1) &
                                                              (mask_df_c_quatrant1.arctan<itv2)]
                dismax_c[j,0] = mask_df_c_quatrant1_itv.d_c.max()
                # quatrant2
                mask_df_c_quatrant2_itv = mask_df_c_quatrant2[(mask_df_c_quatrant2.arctan>=itv1) &
                                                              (mask_df_c_quatrant2.arctan<itv2)]
                dismax_c[j,1] = mask_df_c_quatrant2_itv.d_c.max()
                # quatrant3
                mask_df_c_quatrant3_itv = mask_df_c_quatrant3[(mask_df_c_quatrant3.arctan>=itv1) &
                                                              (mask_df_c_quatrant3.arctan<itv2)]
                dismax_c[j,2] = mask_df_c_quatrant3_itv.d_c.max()
                # quatrant4
                mask_df_c_quatrant4_itv = mask_df_c_quatrant4[(mask_df_c_quatrant4.arctan>=itv1) &
                                                              (mask_df_c_quatrant4.arctan<itv2)]
                dismax_c[j,3] = mask_df_c_quatrant4_itv.d_c.max()

                if self.dataset in self.tech_list_dict['fish']: # for fish data only
                    # quatrant1
                    nmask_df_c_quatrant1_itv = nmask_df_c_quatrant1[(nmask_df_c_quatrant1.arctan>=itv1) &
                                                                  (nmask_df_c_quatrant1.arctan<itv2)]
                    dismax_n[j,0] = nmask_df_c_quatrant1_itv.d_c.max()
                    # quatrant2
                    nmask_df_c_quatrant2_itv = nmask_df_c_quatrant2[(nmask_df_c_quatrant2.arctan>=itv1) &
                                                                  (nmask_df_c_quatrant2.arctan<itv2)]
                    dismax_n[j,1] = nmask_df_c_quatrant2_itv.d_c.max()
                    # quatrant3
                    nmask_df_c_quatrant3_itv = nmask_df_c_quatrant3[(nmask_df_c_quatrant3.arctan>=itv1) &
                                                                  (nmask_df_c_quatrant3.arctan<itv2)]
                    dismax_n[j,2] = nmask_df_c_quatrant3_itv.d_c.max()
                    # quatrant4
                    nmask_df_c_quatrant4_itv = nmask_df_c_quatrant4[(nmask_df_c_quatrant4.arctan>=itv1) &
                                                                  (nmask_df_c_quatrant4.arctan<itv2)]
                    dismax_n[j,3] = nmask_df_c_quatrant4_itv.d_c.max()

            #ipdb.set_trace()
            # for df_c, for arctan in each interval, find max dis using dismax_c
            d_c_maxc = np.zeros(len(df_c))
            # quatrant1
            f1 = df_c.x_c.values>=0
            f2 = df_c.y_c.values>=0
            arctan_quatrant = df_c.arctan[f1 & f2]
            d_c_maxc_quatrant = []
            for k in arctan_quatrant:
                it = np.where(k < tanbin)[0][0]-1
                d_c_maxc_quatrant.append(dismax_c[it,0])
            d_c_maxc[f1&f2] = d_c_maxc_quatrant
            # quatrant2
            f1 = df_c.x_c.values<=0
            f2 = df_c.y_c.values>=0
            arctan_quatrant = df_c.arctan[f1 & f2]
            d_c_maxc_quatrant = []
            for k in arctan_quatrant:
                it = np.where(k < tanbin)[0][0]-1
                d_c_maxc_quatrant.append(dismax_c[it,1])
            d_c_maxc[f1&f2] = d_c_maxc_quatrant
            # quatrant3
            f1 = df_c.x_c.values<=0
            f2 = df_c.y_c.values<=0
            arctan_quatrant = df_c.arctan[f1 & f2]
            d_c_maxc_quatrant = []
            for k in arctan_quatrant:
                it = np.where(k < tanbin)[0][0]-1
                d_c_maxc_quatrant.append(dismax_c[it,2])
            d_c_maxc[f1&f2] = d_c_maxc_quatrant
            # quatrant4
            f1 = df_c.x_c.values>=0
            f2 = df_c.y_c.values<=0
            arctan_quatrant = df_c.arctan[f1 & f2]
            d_c_maxc_quatrant = []
            for k in arctan_quatrant:
                it = np.where(k < tanbin)[0][0]-1
                d_c_maxc_quatrant.append(dismax_c[it,3])
            d_c_maxc[f1&f2] = d_c_maxc_quatrant
            # append to df_c
            df_c['d_c_maxc'] = d_c_maxc

            if self.dataset in self.tech_list_dict['fish']: # for fish data only
                d_c_maxn = np.zeros(len(df_c))
                # quatrant1
                f1 = df_c.x_c>=0
                f2 = df_c.y_c>=0
                arctan_quatrant = df_c.arctan[f1 & f2]
                d_c_maxn_quatrant = []
                for k in arctan_quatrant:
                    it = np.where(k < tanbin)[0][0]-1
                    d_c_maxn_quatrant.append(dismax_n[it,0])
                d_c_maxn[f1&f2] = d_c_maxn_quatrant
                # quatrant2
                f1 = df_c.x_c<=0
                f2 = df_c.y_c>=0
                arctan_quatrant = df_c.arctan[f1 & f2]
                d_c_maxn_quatrant = []
                for k in arctan_quatrant:
                    it = np.where(k < tanbin)[0][0]-1
                    d_c_maxn_quatrant.append(dismax_n[it,1])
                d_c_maxn[f1&f2] = d_c_maxn_quatrant
                # quatrant3
                f1 = df_c.x_c<=0
                f2 = df_c.y_c<=0
                arctan_quatrant = df_c.arctan[f1 & f2]
                d_c_maxn_quatrant = []
                for k in arctan_quatrant:
                    it = np.where(k < tanbin)[0][0]-1
                    d_c_maxn_quatrant.append(dismax_n[it,2])
                d_c_maxn[f1&f2] = d_c_maxn_quatrant
                # quatrant4
                f1 = df_c.x_c>=0
                f2 = df_c.y_c<=0
                arctan_quatrant = df_c.arctan[f1 & f2]
                d_c_maxn_quatrant = []
                for k in arctan_quatrant:
                    it = np.where(k < tanbin)[0][0]-1
                    d_c_maxn_quatrant.append(dismax_n[it,3])
                d_c_maxn[f1&f2] = d_c_maxn_quatrant
                # append to df_c
                df_c['d_c_maxn'] = d_c_maxn

            # scale centered x_c and y_c - seq version
            if self.dataset in self.tech_list_dict['seq']: # for seq data only
                d_c_s = np.zeros(len(df_c))
                x_c_s = np.zeros(len(df_c))
                y_c_s = np.zeros(len(df_c))
                d_c_s = df_c.d_c.copy()/df_c.d_c_maxc.copy()
                x_c_s = df_c.x_c.copy()*(d_c_s/df_c.d_c.copy())
                y_c_s = df_c.y_c.copy()*(d_c_s/df_c.d_c.copy())
                df_c['x_c_s'] = x_c_s
                df_c['y_c_s'] = y_c_s
                df_c['d_c_s'] = d_c_s

            # scale centered x_c and y_c - fish version
            if self.dataset in self.tech_list_dict['fish']: # for fish data only
                t = df_c.type.iloc[0]
                nc_cutoff = self.nc_ratio_dict[t] # nuclear cyto ratio
                d_c_s = np.zeros(len(df_c))
                x_c_s = np.zeros(len(df_c))
                y_c_s = np.zeros(len(df_c))
                f1 = df_c.d_c < d_c_maxn # nuclear
                f2 = df_c.d_c >= d_c_maxn # cyto
                # in nuclear region, divided by maxn (x/y to [-1,1], d to [0,1])
                x_c_f1 = df_c.x_c[f1].copy()
                y_c_f1 = df_c.y_c[f1].copy()
                d_c_f1 = df_c.d_c[f1].copy()
                mn_f1 = df_c.d_c_maxn[f1].copy()
                # compute
                d_c_s_f1 = d_c_f1/mn_f1 # this <=1 is guaranteed
                x_c_s_f1 = d_c_s_f1*x_c_f1/d_c_f1
                y_c_s_f1 = d_c_s_f1*y_c_f1/d_c_f1
                # assign [-1,1] to [-nc_cutoff,nc_cutoff]
                x_c_s[f1] = x_c_s_f1/(1.0/nc_cutoff)
                y_c_s[f1] = y_c_s_f1/(1.0/nc_cutoff)
                d_c_s[f1] = d_c_s_f1/(1.0/nc_cutoff)
                # in cyto region, divided by maxcn (x/y to [-2,-1], [1,2], d to [1,2])
                x_c_f2 = df_c.x_c[f2].copy()
                y_c_f2 = df_c.y_c[f2].copy()
                d_c_f2 = df_c.d_c[f2].copy()
                mn_f2 = df_c.d_c_maxn[f2].copy()
                mc_f2 = df_c.d_c_maxc[f2].copy()
                # compute
                d_ = (d_c_f2-mn_f2)/(mc_f2-mn_f2) # this >=1 is guaranteed
                d_ = d_/(1.0/(1.0-nc_cutoff))
                d_c_s_f2 = np.minimum(nc_cutoff+d_, 1) # remove >1 values (can be regarded as numerical error)
                x_c_s_f2 = (nc_cutoff+d_)*x_c_f2/d_c_f2
                y_c_s_f2 = (nc_cutoff+d_)*y_c_f2/d_c_f2
                # assign
                x_c_s[f2] = x_c_s_f2
                y_c_s[f2] = y_c_s_f2
                d_c_s[f2] = d_c_s_f2
                # add to c_df
                df_c['x_c_s'] = x_c_s
                df_c['y_c_s'] = y_c_s
                df_c['d_c_s'] = d_c_s

            # append
            dict_registered[c] = df_c

        # concatenate to one df
        df_registered = pd.concat(list(dict_registered.values()))

        stop = timeit.default_timer()

        print(f'Time: {stop - start}')
        print(f'Number of cells registered {len(dict_registered)}')

        # update global data variable
        self.df_registered = df_registered

        # save registered df
        outfile = 'output/df_registered.pkl'
        pickle_dict = {}
        pickle_dict['df_registered'] = df_registered
        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass


    def register_cells(self, nc_demo=None, outfile='output/df_registered.pkl'):
        '''
        Function of registering cells to unit circle
        Args:
            demo_nc:
                default - register all cells
                or register given number of cells for testing
        Returns:
            None
            (update global variable `df_registered`)
            (save df_registered as .pkl)
        '''
        if nc_demo is None:
            nc_demo = self.nc_all

        start = timeit.default_timer()

        dict_registered = {}

        df = self.data_df.copy() # cp original data
        df_gbC = df.groupby('cell') # group by `cell`

        for ic, c in enumerate(self.cell_list_all[:nc_demo]): # loop over all cells
            if ic % int(len(self.cell_list_all)/10+1) == 0:
                print(f'{np.round(ic/len(self.cell_list_all)*100,1)} % cell {c}')

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

            # scale centered x_c and y_c - seq version
            #if self.dataset in self.tech_list_dict['seq']: # for seq data only
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

        # concatenate to one df
        df_registered = pd.concat(list(dict_registered.values()))

        stop = timeit.default_timer()

        print(f'Time: {stop - start}')
        print(f'Number of cells registered {len(dict_registered)}')

        # update global data variable
        self.df_registered = df_registered

        # save registered df
        pickle_dict = {}
        pickle_dict['df_registered'] = df_registered
        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass


    def load_registered_cells(self, registered_path=None, registered_dict=None):
        '''
        Function of loading registered cells
        Args:
            registered_dict: read from pkl by default
        Returns:
            None
            (update global data variable with `df_registered`)
        '''
        if registered_dict is None:
            if registered_path is None:
                registered_cells_path = 'output/df_registered_saved.pkl'
            else:
                registered_cells_path = registered_path
            # load registered cells
            with open(registered_cells_path, 'rb') as f:
                registered_dict = pickle.load(f)

        # registered cells df
        df_registered = registered_dict['df_registered']
        # update global data variable
        self.df_registered = df_registered

    def plot_registered_cells(self, c='2102_2069', g='Alb'):
        '''
        Function of plotting one cell for one gene in self.df_registered
        Args:
            c: cell name
            g: gene name
        Returns:
            None
        '''
        df_c = self.df_registered[self.df_registered.cell==c].copy()
        df_c_g = df_c[df_c.gene==g]
        mask_c = self.cell_mask_df[self.cell_mask_df.cell==c]

        fig, ax = plt.subplots(1,2,figsize=(6, 3))

        try:
            plt.register_cmap(cmap=self.green_alpha_object)
        except:
            pass

        # original cell
        ax[0].scatter(mask_c.y, mask_c.x, color='blue', alpha=0.1) # mask_df
        if self.dataset in self.tech_list_dict['fish']: # for fish data only
            nmask_c = self.nuclear_mask_df[self.nuclear_mask_df.cell==c]
            ax[0].scatter(nmask_c.y, nmask_c.x, color='orange', alpha=0.1) # nuclear mask

        ax[0].scatter(df_c.y,
                      df_c.x,
                      c = self.darkgray,
                      alpha = 0.5)

        im = ax[0].scatter(df_c_g.y,
                           df_c_g.x,
                           c = df_c_g.umi,
                           vmin = 0,
                           cmap = 'green_alpha')
        #plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
        ax[0].scatter(df_c_g.centerY.iloc[0], df_c_g.centerX.iloc[0], c=self.red, marker='+', s=100)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_facecolor('black')
        ax[0].set_title('Original cell')

        # registered cell
        ax[1].scatter(df_c.y_c_s,
                      df_c.x_c_s,
                      c = self.darkgray,
                      alpha = 0.5)
        circle_c = plt.Circle((0,0), 1, color=self.darkblue, fill=False)
        ax[1].add_patch(circle_c)
        if self.dataset in self.tech_list_dict['fish']: # for fish data only
            t = df_c.type.iloc[0]
            nc_cutoff = self.nc_ratio_dict[t]
            circle_n = plt.Circle((0,0), nc_cutoff, color=self.darkblue, fill=False)
            ax[1].add_patch(circle_n)
        im = ax[1].scatter(df_c_g.y_c_s,
                           df_c_g.x_c_s,
                           c = df_c_g.umi,
                           vmin = 0,
                           cmap = 'green_alpha')
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].scatter(0, 0, c=self.red, marker='+', s=100)
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_facecolor('black')
        ax[1].set_title('Registered cell')

        plt.suptitle('cell '+ c +' gene '+g)

        plt.tight_layout()
        try:
            fig.savefig('output/plot_registered_cells.png')
        except:
            pass
        #plt.close()

    def nhpp_prepare(self, ci_fix=None):
        '''
        Function of pareparing data for NHPP fit from self.df_registered
        Args:
            ci_fix = None, give ci (=1) for simulations if needed, othereise, will be calculated from sc_total/1000
        Returns:
            None (update self.r_tl, self.c0_tl, self.n_tl, etc.)
        '''
        start = timeit.default_timer()

        df_gbT = self.df_registered.groupby('type') # group by type

        for it, t in enumerate(self.type_list): # loop over all cell types
            print(f'Prepare data of cell type {t}')

            df_t = df_gbT.get_group(t).copy() # df for cell type t
            df_t_gbG = df_t.groupby('gene') # # groupby gene

            r_t = []
            c0_t = []
            c0_t_homo = []
            n_t = []
            n_t_homo = []
            gene_list_ncavl_t = np.zeros(self.ng_dict[t])

            for ig, g in enumerate(self.gene_list_dict[t]):
                n_g = []
                n_g_homo = []
                c0_g = []
                c0_g_homo = []
                r_g = []

                df_t_g = df_t_gbG.get_group(g) # df for gene g in cell type t
                df_t_g_gbC = df_t_g.groupby('cell') # group by cell
                cell_list_t_g = df_t_g.cell.unique().tolist() # cell list for gene g in cell type t
                gene_list_ncavl_t[ig] = len(cell_list_t_g) # number of cells avl for gene g in cell type t

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
                        # rj = max(df_t_g_c.d_c_s.iloc[j], self.c1) # rj shouldn't be too close to 0
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

            # add types
            self.r_tl[t] = r_t
            self.c0_tl[t] = c0_t
            self.c0_tl_homo[t] = c0_t_homo
            self.n_tl[t] = n_t
            self.n_tl_homo[t] = n_t_homo
            self.gene_list_ncavl_dict[t] = gene_list_ncavl_t

        stop = timeit.default_timer()
        print('Time: ', stop - start)

        # save registered df
        outfile = 'output/df_nhpp_prepared.pkl'
        pickle_dict = {}
        pickle_dict['r_tl'] = self.r_tl
        pickle_dict['c0_tl'] = self.c0_tl
        pickle_dict['c0_tl_homo'] = self.c0_tl_homo
        pickle_dict['n_tl'] = self.n_tl
        pickle_dict['n_tl_homo'] = self.n_tl_homo
        pickle_dict['gene_list_ncavl_dict'] = self.gene_list_ncavl_dict
        pickle_dict['type_list'] = self.type_list
        pickle_dict['gene_list_dict'] = self.gene_list_dict
        pickle_dict['cell_list_dict'] = self.cell_list_dict
        pickle_dict['cell_list_all'] = self.cell_list_all

        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass

    def load_nhpp_prepared(self, data_path=None, data_dict=None, beta_kernel_param_list=None):
        '''
        Function of load prepared data for nhpp fit, use for simulations only.
        Args:
            data_dict: if given, give a dict save data for nhpp fit; default load from df_nhpp_prepared_saved.pkl
            beta_kernel_param_list: list of beta kernel hyperparams, use default values if not provided here
        Returns:
            None
        '''
        if data_dict is None:
            # load nhpp prepared data
            if data_path is None:
                nhpp_prepared_cells_path = 'output/df_nhpp_prepared_saved.pkl'
            else:
                nhpp_prepared_cells_path = data_path
            with open(nhpp_prepared_cells_path, 'rb') as f:
                data_dict = pickle.load(f)
        if beta_kernel_param_list is not None:
            # given beta kernel hyperparam list
            self.beta_kernel_param_list = beta_kernel_param_list

        self.r_tl = data_dict['r_tl']
        self.c0_tl = data_dict['c0_tl']
        self.c0_tl_homo = data_dict['c0_tl_homo']
        self.n_tl = data_dict['n_tl']
        self.n_tl_homo = data_dict['n_tl_homo']
        #self.beta_kernel_param_list = data_dict['beta_kernel_param_list']
        try:
            self.gene_list_ncavl_dict = data_dict['gene_list_ncavl_dict']
        except:
            pass

        if data_dict is not None:
            # cell type list
            self.type_list = data_dict['type_list']
            # topN genes lists for each type
            self.gene_list_dict = data_dict['gene_list_dict']
            # cell list
            self.cell_list_dict = data_dict['cell_list_dict']
            # list of all cells
            self.cell_list_all = data_dict['cell_list_all']

            # update global constants
            # number of cell types
            self.nt = len(self.type_list)
            # number of all cells
            self.nc_all = len(self.cell_list_all)
            # number of genes
            ng_list = []
            for t in self.type_list:
                self.ng_dict[t] = len(self.gene_list_dict[t])
                ng_list.append(len(self.gene_list_dict[t]))
            nc_list = []
            # number of cells
            for t in self.type_list:
                self.nc_dict[t] = len(self.cell_list_dict[t])
                nc_list.append(len(self.cell_list_dict[t]))
            # number of default kernels
            self.nk = len(self.beta_kernel_param_list)

        # print
        print(f"Number of cell types {self.nt}")
        print(f"Average number of genes {np.mean(ng_list)}")
        print(f"Average number of cells {np.mean(nc_list)}")
        print(f"Number of default kernels {self.nk}")


    def plot_nc_avl(self, t=None):
        '''
        Function of plotting number of cells avl corresponding to the gene list of cell type t
        Args:
            t: cell type, default, the fist type in self.type_list
        Returns:
            None
        '''
        if t is None:
            t = self.type_list[0]

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        n, bins, patches = ax[0].hist(self.gene_list_ncavl_dict[t], 50)
        ax[1].plot(np.arange(self.ng_dict[t]), self.gene_list_ncavl_dict[t])
        plt.suptitle('Number of cells avl for each gene in type '+str(t))
        plt.tight_layout()
        try:
            fig.savefig('output/plot_nc_avl.png')
        except:
            pass

    def plot_ncounts_avl(self, t=None):
        '''
        For each cell type, for all genes, plot avg number of counts per cell.
        Args:
            None
        Returns:
            None (a plot)
        '''
        fig = plt.figure(figsize=(2*len(self.type_list), 3), dpi=200)
        gs = fig.add_gridspec(1, len(self.type_list),
                              width_ratios=[1]*len(self.type_list),
                              height_ratios=[1])

        for it, t in enumerate(self.type_list):
            ng_t = self.ng_dict[t]
            r_t = self.r_tl[t]
            ncavl_t = self.gene_list_ncavl_dict[t]

            # compute the total counts in all cells
            avg_counts_t = []
            for ig in range(ng_t):
                r_t_g = r_t[ig]
                ncavl_t_g = ncavl_t[ig]
                avg_counts_t.append(len(r_t_g)/ncavl_t_g)

            # plot
            ax = plt.subplot(gs[0, it])
            x = np.arange(len(avg_counts_t))
            y = avg_counts_t
            ax.plot(x,y)
            ax.set_title(t)

        plt.tight_layout()
        try:
            fig.savefig('output/plot_ncounts_avl.png')
        except:
            pass


    def nhpp_adaptive_learning_rate_range(self):
        '''
        Function of computing adaptive initial learning rate of Adam
        Args: none
        Returns: update self.use_adaptive_learning_rate and self.adaptive_learning_rate_dict
        '''
        # compute the likelihood value of homo/null model of the first and last gene in each type
        for it, t in enumerate(self.type_list):
            print(f'type {t}')

            r_t = self.r_tl[t]
            c0_t = self.c0_tl[t]
            c0_t_homo = self.c0_tl_homo[t]
            n_t = self.n_tl[t]
            n_t_homo = self.n_tl_homo[t]

            # homo likelihood of the first gene (genes are sorted and ranked from high expressed to low expressed)
            ig = 0
            g = self.gene_list_dict[t][ig]

            r_g = r_t[ig]
            c0_g = c0_t[ig]
            c0_g_homo = np.array(c0_t_homo[ig])
            n_g = n_t[ig]
            n_g_homo = np.array(n_t_homo[ig])

            hatB = 2*np.sum(n_g_homo)/np.sum(c0_g_homo)
            maxll_first = -np.sum(n_g_homo) + np.sum(n_g_homo)*np.log(hatB) + np.sum(n_g_homo*np.log(c0_g_homo)) + np.sum(np.log(r_g))
            LR_max = np.minimum(self.learning_rate_max, np.maximum(maxll_first/self.learning_rate_adjust, self.learning_rate_min))

            # homo likelihood of the last gene
            ig = self.ng_dict[t]-1
            g = self.gene_list_dict[t][ig]

            r_g = r_t[ig]
            c0_g = c0_t[ig]
            c0_g_homo = np.array(c0_t_homo[ig])
            n_g = n_t[ig]
            n_g_homo = np.array(n_t_homo[ig])

            hatB = 2*np.sum(n_g_homo)/np.sum(c0_g_homo)
            maxll_last = -np.sum(n_g_homo) + np.sum(n_g_homo)*np.log(hatB) + np.sum(n_g_homo*np.log(c0_g_homo)) + np.sum(np.log(r_g))
            LR_min = np.minimum(self.learning_rate_max, np.maximum(maxll_last/self.learning_rate_adjust, self.learning_rate_min))

            print(f'adjust {self.learning_rate_adjust} max {self.learning_rate_max} min {self.learning_rate_min}')
            print(f'delta loss max {self.delta_loss_max} max {self.delta_loss_min} adjust {self.delta_loss_adjust} niter loss unchange {self.niter_loss_unchange}')
            print(f'type {t} maxll 1st {maxll_first:.2f} maxll last {maxll_last:.2f} LR max {LR_max:.4f} LR min {LR_min:.4f}')

    def nhpp_fit_bkp(self, ng_demo=None):
        '''
        Function of NHPP fitting
        Args:
        demo_nc:
            default - run all genes
            or run a given number of genes for testing for all types
        Returns:
            None (update self.A_est, self.C_est, self.mll_est)
        '''
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        else:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo


        start = timeit.default_timer()

        for it, t in enumerate(self.type_list):
            print(f'NHPP fit of cell type {t}')

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
            loss_t = []
            early_stop_t = []
            adaptive_learning_rate_t = []
            adaptive_delta_loss_t = []

            for ig, g in enumerate(self.gene_list_dict[t][:self.ng_demo_dict[t]]):
                g = self.gene_list_dict[t][ig]

                if ig % int(self.ng_demo_dict[t]/10+1) == 0:
                    print(f'{    np.round(ig/self.ng_demo_dict[t]*100,1)} NHPP fit of the {ig+1}th gene {g}')

                # get data for the ith gene
                r_g = r_t[ig]
                c0_g = c0_t[ig]
                c0_g_homo = np.array(c0_t_homo[ig])
                n_g = n_t[ig]
                n_g_homo = np.array(n_t_homo[ig])

                if (len(r_g)>0):

                    ### homo (A=0) PP B est and max loglikelihood value
                    # \sum_i=1^I Ji / 1/2 \sum_i=1^I c0i *(2pi)
                    hatB = 2*np.sum(n_g_homo)/np.sum(c0_g_homo*2.0*math.pi)
                    # -sum_i=1^I Ji + \sum_i=1^I Ji log(2\sum_i Ji / \sum_i c0i*(2pi)) + sum_i Ji log c0i *(2pi)
                    # + sum_i=1^I sum_j=1^Ji log rij
                    maxll = -np.sum(n_g_homo) + np.sum(n_g_homo)*np.log(hatB) + np.sum(n_g_homo*np.log(c0_g_homo*2.0*math.pi)) + np.sum(np.log(r_g))
                    # save
                    B_est_t[ig,-1] = hatB
                    mll_est_t[ig,-1] = maxll

                    ### nonhomo PP fit with kernels
                    # Adam initial values
                    init_A = 0 # initial value of A = 0
                    init_B = hatB # initial value of B = hatB (B est under homo case)
                    initial_A_t.append(init_A)
                    initial_B_t.append(init_B)

                    # prepare data for torch
                    r = torch.reshape(torch.Tensor(r_g), (np.sum(n_g_homo),1))
                    c0 = torch.reshape(torch.Tensor(c0_g), (np.sum(n_g_homo),1))
                    n = torch.reshape(torch.Tensor(n_g), (np.sum(n_g_homo),1))

                    ### loop over all kernels
                    early_stop_g = []
                    loss_g = []

                    # compute adaptive learning rate (LR) based on homo loglikelihood
                    LR = np.minimum(self.learning_rate_max, np.maximum(mll_est_t[ig,-1]/self.learning_rate_adjust, self.learning_rate_min))
                    # compute adaptive delta loss (DL) based on homo loglikelihood
                    DL = np.minimum(self.delta_loss_max, np.maximum(mll_est_t[ig,-1]/self.delta_loss_adjust, self.delta_loss_min))

                    varphi_max = [3.988, 1.998, 2.457, 1.778, 1.500, 1.778, 2.457, 3.988]
                    for k in range(self.nk):
                        # beta pdf shape parameters
                        aa = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][0]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))
                        bb = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][1]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))
                        maxk = torch.reshape(torch.Tensor([varphi_max[k]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))

                        # torch model
                        model = model_beta(init_A, init_B)
                        # spesify optimizer
                        if (self.optimizer == 'adam'):
                            optimizer = optim.Adam(model.parameters(), lr=LR)
                        else:
                            print('should use adam for max likelihood')
                        # training
                        loss_prev = math.inf
                        counter = 0
                        loss_k = [] # save loss values for diagnostic plotting
                        for step in range(0, self.max_iter):
                            optimizer.zero_grad()
                            predictions = model(r, c0, n, aa, bb, maxk)
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
                        A_est_t[ig,k] = model.A.data
                        B_est_t[ig,k] = model.B.data
                        mll_est_t[ig,k] = - loss
                        early_stop_g.append(es)
                        loss_g.append(loss_k)

                early_stop_t.append(early_stop_g)
                loss_t.append(loss_g)
                adaptive_learning_rate_t.append(LR)
                adaptive_delta_loss_t.append(DL)

            # add to dict
            self.A_est[t] = A_est_t
            self.B_est[t] = B_est_t
            self.mll_est[t] = mll_est_t
            self.early_stop_dict[t] = early_stop_t
            self.loss_dict[t] = loss_t
            self.initial_A_dict[t] = initial_A_t
            self.initial_B_dict[t] = initial_B_t
            self.adaptive_learning_rate_dict[t] = adaptive_learning_rate_t
            self.adaptive_delta_loss_dict[t] = adaptive_delta_loss_t

        stop = timeit.default_timer()
        print('Time: ', stop - start)

        # save A_est, C_est, and mll_est
        outfile = 'output/nhpp_fit_results.pkl'
        pickle_dict = {}
        pickle_dict['A_est'] = self.A_est
        pickle_dict['B_est'] = self.B_est
        pickle_dict['mll_est'] = self.mll_est
        pickle_dict['early_stop'] = self.early_stop_dict
        pickle_dict['loss'] = self.loss_dict
        pickle_dict['initial_A'] = self.initial_A_dict
        pickle_dict['initial_B'] = self.initial_B_dict
        pickle_dict['adaptive_learning_rate'] = self.adaptive_learning_rate_dict
        pickle_dict['loss'] = self.loss_dict

        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass
        
    def nhpp_fit(self, outfile = 'output/nhpp_fit_results.pkl', ng_demo=None):
        '''
        Function of NHPP fitting
        Args:
        outfile:
            save nhpp fit results as a .pkl
        demo_nc:
            default - run all genes
            or run a given number of genes for testing for all types
        Returns:
            None (update self.A_est, self.C_est, self.mll_est)
        '''
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        else:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo


        start = timeit.default_timer()

        for it, t in enumerate(self.type_list):
            print(f'NHPP fit of cell type {t}')

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

            for ig, g in tqdm(enumerate(self.gene_list_dict[t][:self.ng_demo_dict[t]]), total=self.ng_demo_dict[t]):
                g = self.gene_list_dict[t][ig]

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

                    ### homo (A=0) PP B est and max loglikelihood value
                    # \sum_i=1^I Ji / 1/2 \sum_i=1^I c0i *(2pi)
                    hatB = 2*np.sum(n_g_homo)/np.sum(c0_g_homo*2.0*math.pi)
                    # -sum_i=1^I Ji + \sum_i=1^I Ji log(2\sum_i Ji / \sum_i c0i*(2pi)) + sum_i Ji log c0i *(2pi)
                    # + sum_i=1^I sum_j=1^Ji log rij
                    maxll = -np.sum(n_g_homo) + np.sum(n_g_homo)*np.log(hatB) + np.sum(n_g_homo*np.log(c0_g_homo*2.0*math.pi)) + np.sum(np.log(r_g))

                    # save - not in use, use Adam est instead
                    # B_est_t[ig,-1] = hatB
                    # mll_est_t[ig,-1] = maxll
                    
                    ### numerical MLE
                    # Adam initial values
                    #init_A = 0 # initial value of A = 0
                    init_A = -10
                    #init_B = hatB # initial value of B = hatB (B est under homo case)
                    init_B_null = hatB
                    init_B_alter = np.log(np.maximum(hatB, 1e-7))
                    #init_B = np.sqrt(hatB)

                    ### nonhomo PP fit with kernels
                    initial_A_t.append(init_A)
                    initial_B_t.append(init_B_null)

                    # prepare data for torch
                    r = torch.reshape(torch.Tensor(r_g), (np.sum(n_g_homo),1))
                    c0 = torch.reshape(torch.Tensor(c0_g), (np.sum(n_g_homo),1))
                    n = torch.reshape(torch.Tensor(n_g), (np.sum(n_g_homo),1))
                    
                    # compute adaptive learning rate (LR) based on homo loglikelihood
                    LR = np.minimum(self.learning_rate_max, np.maximum(maxll/self.learning_rate_adjust, self.learning_rate_min))
                    # compute adaptive delta loss (DL) based on homo loglikelihood
                    DL = np.minimum(self.delta_loss_max, np.maximum(maxll/self.delta_loss_adjust, self.delta_loss_min))
                    
                    ### PP fit (not analytical est)
                    model = model_null(init_B_null)
                    # spesify optimizer
                    if (self.optimizer == 'adam'):
                        #optimizer = optim.Adam(model.parameters(), lr=0.1*LR)
                        optimizer = optim.Adam(model.parameters(), lr=LR)
                    else:
                        print('should use adam for max likelihood for HPP')
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
                    # save results
                    B_est_t[ig, -1] = model.B.data
                    mll_est_t[ig, -1] = - loss
            
                    ### nonhomo PP fit
                    # loop over all kernels
                    # varphi_max = [3.988, 1.998, 2.457, 1.778, 1.500, 1.778, 2.457, 3.988]
                    for k in range(self.nk):
                        # beta pdf shape parameters
                        aa = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][0]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))
                        bb = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][1]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))
                        # maxk = torch.reshape(torch.Tensor([varphi_max[k]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))

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
                        loss_k = [] # save loss values for diagnostic plotting
                        for step in range(0, self.max_iter):
                            optimizer.zero_grad()
                            predictions = model(r, c0, n, aa, bb, self.ri_clamp_min, self.ri_clamp_max)
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

        stop = timeit.default_timer()
        print('Time: ', stop - start)

        # save A_est, C_est, and mll_est
        # outfile = 'output/nhpp_fit_results.pkl'
        pickle_dict = {}
        pickle_dict['A_est'] = self.A_est
        pickle_dict['B_est'] = self.B_est
        pickle_dict['mll_est'] = self.mll_est
        pickle_dict['early_stop'] = self.early_stop_dict
        pickle_dict['loss_nhpp'] = self.loss_nhpp_dict
        pickle_dict['loss_hpp'] = self.loss_hpp_dict
        pickle_dict['initial_A'] = self.initial_A_dict
        pickle_dict['initial_B'] = self.initial_B_dict
        pickle_dict['adaptive_learning_rate'] = self.adaptive_learning_rate_dict

        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass


    def nhpp_fit_parallel_g(self, ig):
        g = self.gene_list_dict[self.T][ig]

        # get data for the ith gene
        r_g = self.r_t[ig]
        c0_g = self.c0_t[ig]
        c0_g_homo = np.array(self.c0_t_homo[ig])
        n_g = self.n_t[ig]
        n_g_homo = np.array(self.n_t_homo[ig])

        res = {}

        if (len(r_g)>0):

            A_est_t = np.zeros(self.nk+1)
            B_est_t = np.zeros(self.nk+1)
            mll_est_t = np.zeros(self.nk+1)
            early_stop_g = np.zeros(self.nk)
            loss_nhpp_g = []
            loss_hpp_g = []

            ### homo (A=0) PP B est and max loglikelihood value
            # \sum_i=1^I Ji / 1/2 \sum_i=1^I c0i *(2pi)
            hatB = 2*np.sum(n_g_homo)/np.sum(c0_g_homo*2.0*math.pi)
            # -sum_i=1^I Ji + \sum_i=1^I Ji log(2\sum_i Ji / \sum_i c0i*(2pi)) + sum_i Ji log c0i *(2pi)
            # + sum_i=1^I sum_j=1^Ji log rij
            maxll = -np.sum(n_g_homo) + np.sum(n_g_homo)*np.log(hatB) + np.sum(n_g_homo*np.log(c0_g_homo*2.0*math.pi)) + np.sum(np.log(r_g))

#             ### save, if use the analytical est of B
#             B_est_t[-1] = hatB
#             mll_est_t[-1] = maxll

            ### Numerical MLE
            # Adam initial values
            #init_A = 0 # initial value of A = 0
            init_A = -10
            #init_B = hatB # initial value of B = hatB (B est under homo case)
            init_B_null = hatB
            init_B_alter = np.log(np.maximum(hatB, 1e-7))
            #init_B = np.sqrt(hatB)

            # prepare data for torch
            r = torch.reshape(torch.Tensor(r_g), (np.sum(n_g_homo),1))
            c0 = torch.reshape(torch.Tensor(c0_g), (np.sum(n_g_homo),1))
            n = torch.reshape(torch.Tensor(n_g), (np.sum(n_g_homo),1))

            # compute adaptive learning rate (LR) based on homo loglikelihood
            LR = np.minimum(self.learning_rate_max, np.maximum(maxll/self.learning_rate_adjust, self.learning_rate_min))
            # compute adaptive delta loss (DL) based on homo loglikelihood
            DL = np.minimum(self.delta_loss_max, np.maximum(maxll/self.delta_loss_adjust, self.delta_loss_min))

            ### PP fit (not analytical est)
            model = model_null(init_B_null)
            # spesify optimizer
            if (self.optimizer == 'adam'):
                #optimizer = optim.Adam(model.parameters(), lr=0.1*LR)
                optimizer = optim.Adam(model.parameters(), lr=LR)
            else:
                print('should use adam for max likelihood for HPP')
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
            # save results
            B_est_t[-1] = model.B.data
            mll_est_t[-1] = - loss

            ### nonhomo PP fit
            # loop over all kernels
            # varphi_max = [3.988, 1.998, 2.457, 1.778, 1.500, 1.778, 2.457, 3.988]
            # varphi_max = [1]*self.nk
            for k in range(self.nk):
                # beta pdf shape parameters
                a = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][0]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))
                b = torch.reshape(torch.Tensor([self.beta_kernel_param_list[k][1]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))
                # maxk = torch.reshape(torch.Tensor([varphi_max[k]]*np.sum(n_g_homo)), (np.sum(n_g_homo),1))
                # torch model
                model = model_beta(init_A, init_B_alter)
                # spesify optimizer
                if (self.optimizer == 'adam'):
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                else:
                    print('should use adam for max likelihood for NHPP')
                # training
                loss_prev = math.inf
                counter = 0
                loss_k = [] # save loss values for diagnostic plotting
                for step in range(0, self.max_iter):
                    optimizer.zero_grad()
                    predictions = model(r, c0, n, a, b, self.ri_clamp_min, self.ri_clamp_max)
                    loss = loss_ll(predictions)
                    loss_k.append(loss.detach().numpy())

                    # early stopping
                    es = self.max_iter
                    if step > self.min_iter:
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
                A_est_t[k] = model.A.data
                B_est_t[k] = model.B.data
                mll_est_t[k] = - loss
                early_stop_g[k] = es
                loss_nhpp_g.append(loss_k)

            res['A_est_t'] = A_est_t
            res['B_est_t'] = B_est_t
            res['mll_est_t'] = mll_est_t
            res['early_stop_g'] = early_stop_g
            res['loss_nhpp_g'] = loss_nhpp_g
            res['loss_hpp_g'] = loss_hpp_g
            res['init_A'] = init_A
            res['init_B'] = init_B_null
            res['LR'] = LR
            res['DL'] = DL

        return res


    def nhpp_fit_parallel(self, outfile = 'output/nhpp_fit_results.pkl', ng_demo=None):
        '''
        Function of NHPP fitting
        Args:
        demo_nc:
            default - run all genes
            or run a given number of genes for testing for all types
        Returns:
            None (update self.A_est, self.C_est, self.mll_est)
        '''
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        else:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo


        start = timeit.default_timer()

        for it, t in enumerate(self.type_list):
            print(f'NHPP fit of cell type {t}')

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

            # tmp self vars
            self.nums = range(self.ng_demo_dict[t])
            self.T = t
            self.r_t = r_t
            self.c0_t = c0_t
            self.c0_t_homo = c0_t_homo
            self.n_t = n_t
            self.n_t_homo = n_t_homo
            #  If max_workers is None or not given, it will default to the number of processors on the machine.
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                res = executor.map(self.nhpp_fit_parallel_g, self.nums)

            for ig, r in enumerate(res):
                A_est_t[ig,:] = r['A_est_t']
                B_est_t[ig,:] = r['B_est_t']
                mll_est_t[ig,:] = r['mll_est_t']
                early_stop_t.append(r['early_stop_g'])
                loss_nhpp_t.append(r['loss_nhpp_g'])
                loss_hpp_t.append(r['loss_hpp_g'])
                initial_A_t.append(r['init_A'])
                initial_B_t.append(r['init_B'])
                adaptive_learning_rate_t.append(r['LR'])
                adaptive_delta_loss_t.append(r['DL'])

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

        stop = timeit.default_timer()
        print('Time: ', stop - start)

        # save A_est, C_est, and mll_est
        pickle_dict = {}
        pickle_dict['A_est'] = self.A_est
        pickle_dict['B_est'] = self.B_est
        pickle_dict['mll_est'] = self.mll_est
        pickle_dict['early_stop'] = self.early_stop_dict
        pickle_dict['loss_nhpp'] = self.loss_nhpp_dict
        pickle_dict['loss_hpp'] = self.loss_hpp_dict
        pickle_dict['initial_A'] = self.initial_A_dict
        pickle_dict['initial_B'] = self.initial_B_dict
        pickle_dict['adaptive_learning_rate'] = self.adaptive_learning_rate_dict

        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass


    def load_nhpp_fit_results(self, res_dict=None, path=None):
        '''
        Function of loading nhpp fit results
        Args:
            None
        Returns:
            None (update self.A_est, self.C_est, self.mll_est, self.best_kernel_tl)
        '''
        
        if path is not None:        
            with open(path, 'rb') as f:
                pickle_dict = pickle.load(f)
            self.A_est = pickle_dict['A_est']
            self.B_est = pickle_dict['B_est']
            self.mll_est = pickle_dict['mll_est']
            self.early_stop_dict = pickle_dict['early_stop']
            self.loss_nhpp_dict = pickle_dict['loss_nhpp']
            self.loss_hpp_dict = pickle_dict['loss_hpp']
        elif res_dict is not None:
            self.A_est = res_dict['A_est']
            self.B_est = res_dict['B_est']
            self.mll_est = res_dict['mll_est']
            self.early_stop_dict = res_dict['early_stop']
            self.loss_nhpp_dict = res_dict['loss_nhpp']
            self.loss_hpp_dict = res_dict['loss_hpp']
        else:
            print('Please specify res_dict or path.')
            


    def plot_kernels(self, idx_selected_kernels=None):
        '''
        Function plotting kernels
        Arg:
            idx_selected_kernels, by default include all
        Returns:
            None
        '''
        if idx_selected_kernels is None:
            idx_selected_kernels=np.arange(self.nk).tolist()

        print(f'Number of kernels selected {len(idx_selected_kernels)}')

        fig, ax = plt.subplots(1, len(idx_selected_kernels), figsize=(len(idx_selected_kernels)*1, 1.5)) # default/max number of kernels=9
        for ik in range(self.nk):
            if ik in idx_selected_kernels:
                a, b = self.beta_kernel_param_list[ik]
                x = np.linspace(0.001, 0.999, 100)
                ax[ik].plot(x, beta.pdf(x, a, b), color=self.lightblue, linewidth=1, alpha=0.6)
                ax[ik].set_title('['+str(ik+1)+'] a='+str(a)+' b='+str(b))
            ax[ik].set_xticks([])
            ax[ik].set_yticks([])
        plt.suptitle('NHPP kernels, mode=(a-1)/(a+b-2)')
        plt.tight_layout()
        try:
            fig.savefig('output/plot_kernels.png')
        except:
            pass

    def compute_pv_chi2_1(self, idx_selected_kernels=None, ng_specified=None, ng_demo=None):
        '''
        Function computing pv of idx_selected_kernels
        Args:
            idx_selected_kernels: by default include all
            ng_specified: by default include all genes
            ng_demo: number of genes included in each cell type, should match with specifications in the previous nhpp_fit function
        Returns:
            None (update self.pv_raw_tl, self.pv_cauchy_tl, self.pv_fdr_tl)
        '''
        if idx_selected_kernels is None:
            idx_selected_kernels=np.arange(self.nk).tolist()
        if ng_specified is None:
            ng_specified_dict = self.ng_dict
        if ng_specified is not None:
            ng_specified_dict = {}
            for t in self.type_list:
                ng_specified_dict[t] = ng_specified
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        if ng_demo is not None:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        print(f'Number of kernels selected {len(idx_selected_kernels)}')

        for t in self.type_list:

            pv_t = np.full((self.ng_demo_dict[t], len(idx_selected_kernels)), 1.0) # to store pv
            ts_t = np.full((self.ng_demo_dict[t], len(idx_selected_kernels)), 0.0) # to store test statistic
            min_t = np.nanmin(self.mll_est[t])
            mll_est_t = np.nan_to_num(self.mll_est[t], nan=min_t) # replace nan with minimum value

            for ig in range(self.ng_demo_dict[t]):
                for ik in range(len(idx_selected_kernels)):
                    idx_k = idx_selected_kernels[ik]
                    T = -2*(mll_est_t[ig,-1] - mll_est_t[ig,idx_k])
                    p = 1 - stats.chi2.cdf(T,1)
                    pv_t[ig,ik] = p
                    ts_t[ig,ik] = T

            # cauchy combination
            pv_cauchy_t = []
            w = np.full(len(idx_selected_kernels), 1/(len(idx_selected_kernels))) # equal weights
            for ig in range(self.ng_demo_dict[t]):
                pv_g = pv_t[ig,:]
                #pv_g[pv_g>0.99] = 0.99 # spark/x did this <<<
                pv_g[pv_g>0.9999999] = 0.9999999
                pv_g[pv_g<0.0000001] = 0.0000001
                tt = np.sum(w * np.tan(pi*(0.5 - pv_g)))
                pc = 0.5 - np.arctan(tt)/pi
                pv_cauchy_t.append(pc)

            # FDR BY
            pv_fdr_t = np.array(stats2.p_adjust(FloatVector(pv_cauchy_t[:ng_specified_dict[t]]), method = 'BY'))

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
        outfile = 'output/pv_fdr.pkl'
        pickle_dict = {}
        pickle_dict['pv_fdr_dict'] = self.pv_fdr_tl

        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass

    def compute_pv(self, idx_selected_kernels=None, ng_specified=None, ng_demo=None):
        '''
        Function computing pv of idx_selected_kernels
        Args:
            idx_selected_kernels: by default include all
            ng_specified: by default include all genes
            ng_demo: number of genes included in each cell type, should match with specifications in the previous nhpp_fit function
        Returns:
            None (update self.pv_raw_tl, self.pv_cauchy_tl, self.pv_fdr_tl)
        '''
        if idx_selected_kernels is None:
            idx_selected_kernels=np.arange(self.nk).tolist()
        if ng_specified is None:
            ng_specified_dict = self.ng_dict
        if ng_specified is not None:
            ng_specified_dict = {}
            for t in self.type_list:
                ng_specified_dict[t] = ng_specified
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        if ng_demo is not None:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        print(f'Number of kernels selected {len(idx_selected_kernels)}')

        for t in self.type_list:

            pv_t = np.full((self.ng_demo_dict[t], len(idx_selected_kernels)), 1.0) # to store pv
            ts_t = np.full((self.ng_demo_dict[t], len(idx_selected_kernels)), 0.0) # to store test statistic
            min_t = np.nanmin(self.mll_est[t])
            mll_est_t = np.nan_to_num(self.mll_est[t], nan=min_t) # replace nan with minimum value
            
            for ig in range(self.ng_demo_dict[t]):
                for ik in range(len(idx_selected_kernels)):
                    idx_k = idx_selected_kernels[ik]
                    T = -2*(mll_est_t[ig,-1] - mll_est_t[ig,idx_k])
                    #p = 1 - stats.chi2.cdf(T,1)
                    cutoff0 = 2e-3 # <<<!!!
                    if T<=cutoff0:
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
                #pv_g[pv_g>0.99] = 0.99 # spark/x did this <<<
                pv_g[pv_g>0.9999999] = 0.9999999
                pv_g[pv_g<0.0000001] = 0.0000001
                tt = np.sum(w * np.tan(pi*(0.5 - pv_g)))
                pc = 0.5 - np.arctan(tt)/pi
                pv_cauchy_t.append(pc)

            # FDR BY
            pv_fdr_t = np.array(stats2.p_adjust(FloatVector(pv_cauchy_t[:ng_specified_dict[t]]), method = 'BY'))

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
        outfile = 'output/pv_fdr.pkl'
        pickle_dict = {}
        pickle_dict['pv_fdr_dict'] = self.pv_fdr_tl

        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass


    def weighted_density_est(self, idx_selected_kernels=None, ng_demo=None):
        '''
        Function computing weights, lam est, and scores of idx_selected_kernels
        (Weight all lam ests of selected kernels with likelihoods)
        Args:
            idx_selected_kernels: by default include all
            ng_demo:  number of genes included in each cell type, should match with specifications in the previous nhpp_fit function
        Returns:
            None (update GRESULT)
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

            # handle nan
            min_t = np.nanmin(mll_t)
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
                    #lam_k = A_k*beta.pdf(x, a, b)+B_k
                    lam_k = np.maximum(0, A_k*beta.pdf(x, a, b)+B_k) # make this non-negative!!!
                    weighted_lam_est_g = weighted_lam_est_g + weight_ml_t[ig,ik]*lam_k
                idx = np.argmax(weighted_lam_est_g)
                scores_t[ig] = x[idx]
                weighted_lam_est_t.append(weighted_lam_est_g)

            self.weight_ml[t] = weight_ml_t
            self.scores[t] = scores_t
            self.weighted_lam_est[t] = weighted_lam_est_t

        # save scores as pkl
        outfile = 'output/scores.pkl'
        pickle_dict = {}
        pickle_dict['scores_dict'] = self.scores
        pickle_dict['weights_dict'] = self.weight_ml
        pickle_dict['weighted_lam_dict'] = self.weighted_lam_est

        try:
            with open(outfile, 'wb') as f:
                pickle.dump(pickle_dict, f)
        except:
            pass

    def plot_number_of_sig_genes(self, selected_type_list=None, ng_demo=None):
        '''
        Function plotting number of significant genes in each cell type
        Args:
            selected_type_list: cell type list to plot, by default include all
            ng_demo:  number of genes included in each cell type, should match with specifications in the previous nhpp_fit function
        Returns:
            None
        '''
        if selected_type_list is None:
            selected_type_list=self.type_list
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        else:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        sig_idx = {}
        sig_gene = {}
        sig_gene_common = self.gene_list_dict[selected_type_list[0]]
        for t in selected_type_list:
            pv_fdr = self.pv_fdr_tl[t]
            sig_idx_t = pv_fdr <= self.sig_cutoff
            sig_idx[t] = sig_idx_t
            sig_gene[t] = (np.array(self.gene_list_dict[t][:self.ng_demo_dict[t]])[sig_idx_t]).tolist()
            sig_gene_common = list(set(sig_gene_common).intersection(sig_gene[t]))

        n_sig_common = len(sig_gene_common)
        n_sig_distinct = []
        for t in selected_type_list:
            n_sig_t = np.sum(sig_idx[t]) - n_sig_common
            n_sig_distinct.append(n_sig_t)

        df = pd.DataFrame({
            'common': [n_sig_common]*len(selected_type_list),
            'distinct': n_sig_distinct},
            index = selected_type_list)

        fig, ax = plt.subplots(1,1,figsize=(3, 3))
        df.plot(ax = ax, kind='bar', stacked=True, color=[self.darkblue, self.lightblue])
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Number of sig genes at FDR '+str(self.sig_cutoff))
        try:
            fig.savefig('fig/'+self.dataset+'_plot_number_of_sig_genes.png')
        except:
            pass

    def plot_scores_of_sig_genes(self, selected_type_list=None, ng_demo=None):
        '''
        Function plotting scores of significant genes in each cell type
        Args:
            cell type list to plot, by default include all
        Returns:
            None
        '''
        if selected_type_list is None:
                selected_type_list=self.type_list
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        else:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        if len(selected_type_list)>1:
            fig, ax = plt.subplots(1, len(selected_type_list), figsize=(2*len(selected_type_list), 3))
            for it, t in enumerate(selected_type_list):
                pv_fdr_t = self.pv_fdr_tl[t]
                sig_idx_t = pv_fdr_t<=self.sig_cutoff
                n, bins, patches = ax[it].hist(self.scores[t][sig_idx_t], 50, color=self.lightblue)
                n_sig_t = np.sum(pv_fdr_t<=self.sig_cutoff)
                ax[it].set_ylabel(str(t)+' #sig='+str(n_sig_t))
        if len(selected_type_list)==1:
            fig, ax = plt.subplots(1, len(selected_type_list), figsize=(2*len(selected_type_list), 3))
            for it, t in enumerate(selected_type_list):
                pv_fdr_t = self.pv_fdr_tl[t]
                sig_idx_t = pv_fdr_t<=self.sig_cutoff
                n, bins, patches = ax.hist(self.scores[t][sig_idx_t], 50, color=self.lightblue)
                n_sig_t = np.sum(pv_fdr_t<=self.sig_cutoff)
                ax.set_ylabel(str(t)+' #sig='+str(n_sig_t))
        plt.suptitle('Scores of significant genes')
        plt.tight_layout()
        try:
            fig.savefig('output/plot_scores_of_sig_genes.png')
        except:
            pass

    def plot_est_density_of_sig_genes(self, selected_type_list=None, ng_demo=None):
        '''
        Function plotting est density of significant genes in each cell type
        Args:
            selected_type_list: cell type list to plot, by default include all
            ng_demo:  number of genes included in each cell type, should match with specifications in the previous nhpp_fit function
        Returns:
            None
        '''
        if selected_type_list is None:
            selected_type_list=self.type_list
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        if ng_demo is not None:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        x = np.linspace(0.001,0.999,100)
        sig_cutoff = 0.05
        color_cutoff = [0.0, 0.2, 0.4, 0.6, 0.8] # <<<<<
        cols = [self.red, self.darkorange, self.lightorange, self.lightgreen, self.lightblue]

        if len(selected_type_list)>1:
            fig, ax = plt.subplots(2, len(selected_type_list), figsize=(3*len(selected_type_list), 6))
            for it, t in enumerate(selected_type_list):
                pv_fdr_t = self.pv_fdr_tl[t]
                sig_idx_t = pv_fdr_t<=self.sig_cutoff
                weighted_lam_est_t = self.weighted_lam_est[t]
                scores_t = self.scores[t]

                # plot each sig gene
                for ig in range(self.ng_demo_dict[t]):
                    if pv_fdr_t[ig]<=self.sig_cutoff:
                        score_g = scores_t[ig]
                        idx_g = np.where(score_g >= color_cutoff)[0][-1]
                        color_g = cols[idx_g]
                        lw_t_g = weighted_lam_est_t[ig]
                        y = lw_t_g 
                        # lw_t_g_std = (lw_t_g-np.mean(lw_t_g))/np.std(lw_t_g) # std by mean and sd
                        # area = np.trapz(y, dx=1/len(y)) # std by area under curve
                        # y_std = y/area
                        y_std = (y-np.min(y))/(np.max(y)-np.min(y)+1e-8) # std by min-max
                        
                        ax[0,it].plot(x, lw_t_g, color=color_g, linewidth=0.5, alpha=0.8)
                        #ax[1,it].plot(x, lw_t_g_std, color=color_g, linewidth=0.5, alpha=0.8)
                        ax[1,it].plot(x, y_std, color=color_g, linewidth=0.5, alpha=0.8)
                n_sig_t = np.sum(pv_fdr_t<=self.sig_cutoff)
                ax[0,it].set_ylabel(str(t)+' #sig='+str(n_sig_t))
                ax[1,it].set_ylabel('Normalized')
        if len(selected_type_list)==1:
            fig, ax = plt.subplots(2, len(selected_type_list), figsize=(3*len(selected_type_list), 6), dpi=200)
            for it, t in enumerate(selected_type_list):
                pv_fdr_t = self.pv_fdr_tl[t]
                sig_idx_t = pv_fdr_t<=self.sig_cutoff
                weighted_lam_est_t = self.weighted_lam_est[t]
                scores_t = self.scores[t]

                # plot each sig gene
                for ig in range(self.ng_demo_dict[t]):
                    if pv_fdr_t[ig]<=self.sig_cutoff:
                        score_g = scores_t[ig]
                        idx_g = np.where(score_g >= color_cutoff)[0][-1]
                        color_g = cols[idx_g]
                        lw_t_g = weighted_lam_est_t[ig]
                        y = lw_t_g 
                        # lw_t_g_std = (lw_t_g-np.mean(lw_t_g))/np.std(lw_t_g) # std by mean and sd
                        # area = np.trapz(y, dx=1/len(y)) # std by area under curve
                        # y_std = y/area
                        y_std = (y-np.min(y))/(np.max(y)-np.min(y)+1e-8) # std by min-max
                        ax[0].plot(x, lw_t_g, color=color_g, linewidth=0.5, alpha=0.8)
                        #ax[1].plot(x, lw_t_g_std, color=color_g, linewidth=0.5, alpha=0.8)
                        ax[1].plot(x, y_std, color=color_g, linewidth=0.5, alpha=0.8)
                n_sig_t = np.sum(pv_fdr_t<=self.sig_cutoff)
                ax[0].set_ylabel(str(t)+' #sig='+str(n_sig_t))
                ax[1].set_ylabel('Normalized')
        plt.suptitle('NHPP averaged densities of significant genes\n')
        plt.tight_layout()
        try:
            fig.savefig('output/plot_est_density_of_sig_genes.png')
        except:
            pass

    def plot_labels_of_sig_genes(self, selected_type_list=None, ng_demo=None):
        '''
        Function plotting labels of significant genes in each cell type
        Scores are discretized to 5 labels
        Plot the proportion of each label in each cell type
        Args:
            cell type list to plot, by default include all
        Returns:
            None (update self.labels)
        '''
        if selected_type_list is None:
            selected_type_list=self.type_list
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        if ng_demo is not None:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        # discretize sig gene scores to 5 labels
        for it, t in enumerate(selected_type_list):
            label_t = (self.scores[t]*5).astype(int)
            sig_idx_t = self.pv_fdr_tl[t]<=self.sig_cutoff
            label_t[~sig_idx_t] = -1
            self.labels[t] = label_t

        # proportion of each label in each cell type
        count_label = {}
        for it, t in enumerate(selected_type_list):
            count_label_t = []
            for k in range(5):
                count_label_t.append(np.sum(self.labels[t]==k))
            count_label[t] = count_label_t

        # pie plot
        cols = [self.red, self.darkorange, self.lightorange, self.lightgreen, self.lightblue]
        labs = np.arange(1, 6).tolist()

        if len(selected_type_list)>1:
            fig, ax = plt.subplots(1, len(selected_type_list), figsize=(2*len(selected_type_list), 3))
            for it, t in enumerate(selected_type_list):
                patches, autotexts = ax[it].pie(count_label[t], labels=labs, shadow=False, startangle=90, colors=cols)
                ax[it].set_title(t)
        if len(selected_type_list)==1:
            fig, ax = plt.subplots(1, len(selected_type_list), figsize=(2*len(selected_type_list), 3))
            for it, t in enumerate(selected_type_list):
                patches, autotexts = ax.pie(count_label[t], labels=labs, shadow=False, startangle=90, colors=cols)
                ax.set_title(t)
        plt.suptitle('Labels of significant genes')
        plt.tight_layout()
        try:
            fig.savefig('output/plot_labels_of_sig_genes.png')
        except:
            pass

    def plot_labels_in_common(self, selected_type_list=None, ng_demo=None):
        '''
        Function plotting labels in common of significant genes in each cell type
        Args:
            cell type list to plot, by default include all
        Returns:
            None
        '''
        if selected_type_list is None:
            selected_type_list=self.type_list
        if ng_demo is None:
            self.ng_demo_dict = self.ng_dict
        if ng_demo is not None:
            self.ng_demo_dict = {}
            for t in self.type_list:
                self.ng_demo_dict[t] = ng_demo

        fig, ax = plt.subplots(1, 5, figsize=(2*5, 3)) # 5 labels

        # calculating numbers
        n_gene_distinct_ll = {}
        n_gene_common_ll = {}
        for l in range(5): #!!! 5 labels
            gene_common_l = self.gene_list_dict[selected_type_list[0]]
            n_gene_l = []
            for t in selected_type_list:
                # sig gene with label l
                gene_l_t = np.array(self.gene_list_dict[t])[self.labels[t] == l] # nonsig label=-1
                gene_common_l = list(set(gene_common_l).intersection(gene_l_t))
                n_gene_l.append(len(gene_l_t))
            n_gene_common_l = len(gene_common_l)
            n_gene_distinct_l = np.array(n_gene_l) - n_gene_common_l
            n_gene_common_ll[l] = [n_gene_common_l]*len(selected_type_list)
            n_gene_distinct_ll[l] = n_gene_distinct_l

        # plot
        for l in range(5): #!!! 5 labels
            df = pd.DataFrame({'#common': n_gene_common_ll[l],
                               '#distinct': n_gene_distinct_ll[l]},
                               index = selected_type_list)
            df.plot(ax = ax[l], kind='bar', stacked=True, color=[self.darkblue, self.lightblue])
            ax[l].set_ylabel('label='+str(l+1))
            ax[l].set_xticks(ax[l].get_xticks(), ax[l].get_xticklabels(), rotation=45, ha='right')
            # adjustment
            if l!=4:
                ax[l].get_legend().remove()
        plt.suptitle('Labels of significant genes')
        plt.tight_layout()
        try:
            fig.savefig('output/plot_labels_in_common.png')
        except:
            pass

    def plot_results_one_gene(self, t=None, g=None, c=None, bw = 0.1, details=False):
        '''
        Function of plotting the results of one gene and one cell in one cell type
        Args:
            c: cell name, default the fisrt
            g: gene name, default the fisrt
            t: cell type, default the fisrt
            bw: kernel bandwidth of naive kernel density estimation method
            default value as listed above
            details: whether plot cellwise densities and binned densities or not, default False
        Returns:
            None
        '''
        if t is None:
            t = self.type_list[0]
            #print(f't={t}')
        if g is None:
            g = self.gene_list_dict[t][0]
            #print(f'g={g}')
        if c is None:
            c = self.cell_list_dict[t][0]
            #print(f'c={c}')

        df_c = self.df_registered[self.df_registered.cell==c].copy()
        df_c_g = df_c[df_c.gene==g]

        fig = plt.figure(figsize=(7, 6))
        gs = GridSpec(nrows=2, ncols=2)

        try:
            plt.register_cmap(cmap=self.green_alpha_object)
        except:
            pass

        ### plot original cell
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.scatter(df_c.y,
                    df_c.x,
                    c = self.darkgray,
                    alpha = 0.5)
        im = ax0.scatter(df_c_g.y,
                         df_c_g.x,
                         c = df_c_g.umi,
                         vmin = 0,
                         cmap = 'green_alpha')
        plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        ax0.scatter(df_c_g.centerY.iloc[0], df_c_g.centerX.iloc[0], c=self.red, marker='P', s=100)
        ax0.set_aspect('equal', adjustable='box')
        ax0.set_facecolor('black')
        ax0.set_title('Original cell')

        ### plot registered cell
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.scatter(df_c.y_c_s,
                    df_c.x_c_s,
                    c = self.darkgray,
                    alpha = 0.5)
        circle_c = plt.Circle((0,0), 1, color=self.darkblue, fill=False)
        ax1.add_patch(circle_c)
        if self.dataset in self.tech_list_dict['fish']: # for fish data only
            t = df_c.type.iloc[0]
            nc_cutoff = self.nc_ratio_dict[t]
            circle_n = plt.Circle((0,0), nc_cutoff, color=self.darkblue, fill=False)
            ax1.add_patch(circle_n)
        im = ax1.scatter(df_c_g.y_c_s,
                         df_c_g.x_c_s,
                         c = df_c_g.umi,
                         vmin = 0,
                         cmap = 'green_alpha')
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.scatter(0, 0, c=self.red, marker='P', s=100)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_facecolor('black')
        ax1.set_title('Registered cell')

        idx_g = np.where(np.array(self.gene_list_dict[t])==g)[0][0] # the idx of specified gene
        ax2 = fig.add_subplot(gs[:, 1])
        score_t = self.scores[t]
        score_t_g = score_t[idx_g]
        y_relative_g = [] # for collecting the kernel density for all cells
        sc_gene_g = [] # for collecting the sc total counts of this gene
        df = self.df_registered.copy()
        df_t_g = df[(df.type==t) & (df.gene==g)]
        df_t_g_gbC = df_t_g.groupby('cell')
        cell_list_t_g = df_t_g.cell.unique().tolist()
        if not details:
            # gene g counts per cell
            for ic, cc in enumerate(cell_list_t_g):
                df_t_g_c = df_t_g_gbC.get_group(cc)
                sc_gene_g.append(df_t_g_c.umi.sum())
        if details:
            ### plot naive kernel density estimation (normalize area under curve=1)
            # density of all cells
            for ic, cc in enumerate(cell_list_t_g):
                df_t_g_c = df_t_g_gbC.get_group(cc)
                total_c = int(df_t_g_c.sc_total.iloc[0]/1000)+1 # count per k
                sc_gene_g.append(df_t_g_c.umi.sum())
                d_ = df_t_g_c.d_c_s.values
                umi_ = df_t_g_c.umi.values/(d_+self.c0) # normalize by area
                umi_ = np.round(umi_).astype(int)
                x_ = []
                for ix in range(len(df_t_g_c)):
                    x_ = x_ + [d_[ix]]*umi_[ix]
                X = np.array(x_)[:, np.newaxis]
                X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
                kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X)
                log_dens = kde.score_samples(X_plot)
                dens = np.exp(log_dens)/total_c
                y_relative_g.append(dens)
                # plot for individual cells
                area = np.trapz(dens, dx=1/len(dens))
                ax2.plot(X_plot[:, 0], dens/area, color=self.lightgreen, alpha=0.1)
            # mean density
            y_relative_g2 = np.array(y_relative_g)
            y_mean_g = np.mean(y_relative_g2, axis=0)
            area = np.trapz(y_mean_g, dx=1/len(y_mean_g))
            ax2.plot(X_plot[:, 0], y_mean_g/area, color=self.darkgreen, linewidth=2, alpha=1)

            ### plot bin 10 density (piecewise constant method) (normalize area under curve=1)
            # bin10 mean density
            y_relative_g = [] # for collecting the EG for all cells
            sc_gene_g = [] # for collecting the sc total counts of this gene
            for ic, cc in enumerate(cell_list_t_g):
                df_t_g_c = df_t_g_gbC.get_group(cc)
                total_c = df_t_g_c.sc_total.iloc[0]
                df_ = df_t_g_c[['umi', 'd_c_s']].copy()
                df_['umi2'] = df_.umi.copy()/df_.d_c_s.copy()
                df_['d2'] = np.floor(df_.d_c_s.copy()*10)
                df2_ = df_.groupby('d2')[['umi2', 'umi']].sum()
                x = np.linspace(0, 1, 11)
                y = np.zeros(10)
                idx = df2_.index.to_numpy().astype(int)
                if max(idx)<10:
                    y[idx] = df2_.umi2.values
                if max(idx)==10:
                    idx2 = idx[:-1]
                    y[idx2] = (df2_.umi2.values)[:-1]
                    y[-1] = y[-1] + (df2_.umi2.values)[-1]
                y = np.append(y, y[-1])
                # std y
                y_relative = y/total_c
                y_relative = y_relative/(np.sum(y_relative)+self.epsilon)
                y_relative_g.append(y_relative)
                sc_gene_g.append(df2_.umi.sum())
                # plot for individual cells
                area = np.trapz(y_relative[:10], dx=0.1)
                ax2.step(x, y_relative/area, where='post', color=self.lightblue, linewidth=1, alpha=0.1)
            # mean bin10 density
            y_relative_g2 = np.array(y_relative_g)
            y_mean_g = np.mean(y_relative_g2, axis=0)
            y_mean_g = y_mean_g/np.sum(y_mean_g)
            area = np.trapz(y_mean_g[:10], dx=0.1)
            ax2.step(x, y_mean_g/area, where='post', color=self.darkblue, linewidth=2, alpha=1)

        ### plot NHPP density (normalize area under curve=1)
        # weignted intensity
        weighted_lam_est_t = self.weighted_lam_est[t]
        weighted_lam_est_t_g = weighted_lam_est_t[idx_g]
        x = np.linspace(0.001,0.999,100)
        y = weighted_lam_est_t_g
        area = np.trapz(y, dx=1/len(y))
        ax2.plot(x, y/area, color=self.darkorange, linewidth=2, alpha=1)
        ax2.set_title('NHPP density est')
        # add pv and score
        pv_g = np.round(self.pv_fdr_tl[t][idx_g], 4)
        score_g = np.round(self.scores[t][idx_g], 2)
        nc_g = len(cell_list_t_g) # cells available of this gene
        scm_g = np.round(np.mean(sc_gene_g),2) # average counts per cell of this gene
        ax2.text(0.03, ax2.get_ylim()[1]*0.85,
                'score '+str(score_g)+'\n'+
                'pv (fdr) '+str(pv_g)+'\n'+
                'number of cells '+str(nc_g)+'\n'+
                'average counts per cell '+str(scm_g),
                fontsize = 10)

        plt.suptitle('cell '+c+' gene '+g+ ' type '+t)
        plt.tight_layout()
        # plt.close()
        try:
            fig.savefig('output/plot_results_one_gene.png')
        except:
            pass
