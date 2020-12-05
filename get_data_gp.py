import os
import glob
import pickle
import numpy as np
import torch
import math
from easydict import EasyDict as edict
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import Matern
from model import *
from utils.gp_helper import cal_kern_spec_mix_nomu_sep, cal_marg_likelihood, standardize, cal_marg_likelihood_single_L
from utils.nmll_opt import nmll_opt, nmll_opt_gp
import pdb
from tqdm import tqdm


is_interval = False
torch.set_printoptions(precision=4,linewidth=150)
np.set_printoptions(precision=4,linewidth=150)
save_dir = './data/synthetic/gp_synthetic_2_15_dim_30p_5k/'

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

if not os.path.exists(save_dir):
  os.mkdir(save_dir)
  print('made directory {}'.format(save_dir))

def is_pos_def(X):
  eig_values,_ = torch.eig(X,eigenvectors=False)
  if_psd = torch.all(eig_values[:,0]>0)
  return if_psd

def gen_data(min_dim = 2,
             max_dim = 15,
             min_num_mix = 1,
             max_num_mix = 10,
             intensity_pt = 30,
             num_gps=5000, 
             seed=111,
             base_lengthscale = 0.1,
             max_lengthscale=1.5,
             min_lengthscale=0.05,
             p_large = 1/4.,
             epsilon = 1e-2,
             is_opt=False,
             opt_settings=None):
  """
    Generate synthetic Gaussian process data
  """
  npr = np.random.RandomState(seed)
  D = npr.randint(min_dim, high=max_dim+1, size=num_gps)
  N = npr.poisson(intensity_pt, num_gps)
  #M = npr.randint(min_num_mix, high=max_num_mix+1, size=num_gps)
  M = max_num_mix
  data = []
  ii = 0
  for ii in tqdm(range(num_gps)):
    data_dict = {}
    # generate two seperate datasets, X and X_2
    X = (npr.rand(N[ii], D[ii])-0.5)*2
    X_2 = (npr.rand(N[ii], D[ii])-0.5)*2
    X, X_2, _, _ = standardize(X, X_2)
    X = X * base_lengthscale
    X_2 = X_2 * base_lengthscale

    weights = (npr.dirichlet(np.ones(M), D[ii])).T
    length_normal = np.exp(npr.rand(M, D[ii]) * (math.log(max_lengthscale)-math.log(min_lengthscale)) + math.log(min_lengthscale))
    var_normal = 1/length_normal**2/(2*math.pi**2)
    is_length_large = npr.multinomial(1,[1-p_large,p_large],(D[ii]))
    length_large = np.ones((M, D[ii]))*1000
    var_small = 1/length_large**2/(2*math.pi**2)
    var = is_length_large[:,0]*var_normal+is_length_large[:,1]*var_small

    var_torch = torch.from_numpy(var).float().to(device)
    weights_torch = torch.from_numpy(weights).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    kern_matrix = cal_kern_spec_mix_nomu_sep(X_torch, X_torch, var_torch, weights_torch)
    X_2_torch = torch.from_numpy(X_2).float().to(device)
    kern_matrix_X2 = cal_kern_spec_mix_nomu_sep(X_2_torch, X_2_torch, var_torch, weights_torch)
    try:
      L = np.linalg.cholesky(kern_matrix.cpu().numpy()+epsilon*np.eye(N[ii]))
      L_2 = np.linalg.cholesky(kern_matrix_X2.cpu().numpy()+epsilon*np.eye(N[ii]))
    except Exception:
      continue
  
    f = np.matmul(L, npr.randn(N[ii], 1))
    f_2 = np.matmul(L_2, npr.randn(N[ii], 1))
    f, mean_f, std_f = standardize(f)
    f_2, mean_f_2, std_f_2 = standardize(f_2)


    L_torch = torch.from_numpy(L).float().to(device)
    L_2_torch = torch.from_numpy(L_2).float().to(device)
    nmll = -cal_marg_likelihood_single_L(torch.from_numpy(f).float().to(device), L_torch)
    nmll_test = -cal_marg_likelihood_single_L(torch.from_numpy(f_2).float().to(device), L_2_torch)
    data_dict['nmll'] = nmll.cpu().numpy()
    data_dict['nmll_test'] = nmll_test.cpu().numpy()
    data_dict['X'] = X # N X D
    data_dict['X_2'] = X_2 # N X D
    data_dict['f'] = f.squeeze(-1) # N
    data_dict['f_2'] = f_2.squeeze(-1)# N
    sm_params_orig = edict()
    sm_params_orig.var = var_torch
    sm_params_orig.weights = weights_torch
    data_dict['sm_params_orig'] = sm_params_orig
    data_dict['nmll_opt_sm'] = data_dict['nmll']
    data_dict['nmll_opt_sm_test'] = data_dict['nmll_test']

    if is_opt:
      # perform marginal likelihood optimization 
      model_params = edict()
      model_params.input_dim = X.shape[1]
      model_params.num_mix = 10
      model_params.is_dim_diff = True
      model_params.is_no_mu = True
      sm_params, _ = nmll_opt(data_dict, model_params, opt_settings)
      data_dict['sm_params'] = sm_params

    data += [data_dict]

  return data

def dump_data(data_list, tag='train'):
  count = 0
  print('Dump {} data!'.format(tag))
  for data in data_list:

    pickle.dump(
      data,
      open(
        os.path.join(save_dir, 'synthetic_{}_{:07d}.p'.format(tag, count)),
        'wb'))

    count += 1

  print('100.0 %%')


if __name__ == '__main__':
  settings = edict()
  settings.epsilon = 1.0e-2
  settings.lr = 1.0e-1
  settings.training_iter = 200
  settings.is_print = False
  settings.device = device
  settings.opt_is_lbfgs = False

  train_dataset = gen_data(seed=123, num_gps=5000, is_opt=False, opt_settings=settings)
  dev_dataset = gen_data(seed=456, num_gps=5000, is_opt=False, opt_settings=settings)
  test_dataset = gen_data(seed=789, num_gps=5000, is_opt=False, opt_settings=settings)

  dump_data(train_dataset, 'train')
  dump_data(dev_dataset, 'dev')
  dump_data(test_dataset, 'test')
