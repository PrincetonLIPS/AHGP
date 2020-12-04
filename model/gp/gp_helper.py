import torch
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_gaussian_mixture_1d(var, weights, mu=None):
  """
  Visualize 1D Gaussian mixture
  """
  if mu is None:
    mu = np.zeros_like(var)
  x = np.linspace(start = -10, stop = 10, num = 2000)
  y_cum = np.zeros_like(x)
  for ii in range(var.shape[0]):
    y = norm(0,np.sqrt(var[ii].item())).pdf(x)
    y_cum = y * weights[ii].item() + y_cum
  plt.plot(x, y_cum)


def standardize(data_train, *args):
  """
  Standardize a dataset to have zero mean and unit standard deviation.
  :param data_train: 2-D Numpy array. Training data.
  :param data_test: 2-D Numpy array. Test data.
  :return: (train_set, test_set, mean, std), The standardized dataset and
    their mean and standard deviation before processing.
  """
  std = np.std(data_train, 0, keepdims=True)
  std[std == 0] = 1
  mean = np.mean(data_train, 0, keepdims=True)
  data_train_standardized = (data_train - mean) / std
  output = [data_train_standardized]
  for d in args:
    dd = (d - mean) / std
    output.append(dd)
  output.append(mean)
  output.append(std)
  return output

def GP_noise(y1, K11, K12, K22, epsilon_noise, device):
  """
  Calculate the posterior mean and covariance matrix for y2 based on the noisy observations y1 and the given kernel matrix
  """
  # Kernel of the noisy observations
  K11 = K11 + epsilon_noise * torch.eye(K11.shape[0]).to(device)
  solved, _ = torch.solve(K12, K11)
  # Compute posterior mean
  mu_2 = torch.matmul(solved.T, y1)
  var_2 = K22 - torch.matmul(solved.T, K12)
  return mu_2, var_2  # mean, covariance

def cal_marg_likelihood_single(K, f, epsilon, device):
  N = f.shape[0]
  L = torch.cholesky(K+epsilon*torch.eye(N).to(device))
  singular_values = L.diagonal(offset=0)
  logdet = torch.sum(torch.log(singular_values)*2)
  data_fit = -(f.transpose(-1,-2)).matmul(torch.inverse(K+epsilon*torch.eye(N).to(device))).matmul(f).squeeze(-1)
  AvgMLL = (0.5*data_fit - 0.5*logdet)/N - 0.5*math.log(2*math.pi)
  return AvgMLL

def cal_marg_likelihood_single_L(f, L):
  N = f.shape[0]
  singular_values = L.diagonal(offset=0)
  logdet = torch.sum(torch.log(singular_values)*2)
  L_inv = torch.inverse(L)
  f_bar = L_inv.matmul(f)
  data_fit = -(f_bar.transpose(-1,-2).matmul(f_bar)).squeeze(-1)
  AvgMLL = (0.5*data_fit - 0.5*logdet)/N - 0.5*math.log(2*math.pi)
  return AvgMLL

def cal_marg_likelihood(K, f, epsilon, kernel_mask, diagonal_mask, N, device):
  # K: B X N X N
  # f: B X N X 1 (filled with zeros)
  diag_size = f.shape[1]
  K = (K + epsilon*torch.eye(diag_size).to(device).unsqueeze(0))*kernel_mask # fill the rest with zeros
  K = K+torch.eye(diag_size).to(device).unsqueeze(0)*(1-kernel_mask) # add ones to the diagonal
  L = torch.cholesky(K)
  singular_values = L.diagonal(offset=0, dim1=1, dim2=2)
  logdet = torch.sum(torch.log(singular_values)*2*(1-diagonal_mask),1)
  data_fit = -(f.transpose(-1,-2)).matmul(torch.inverse(K)).matmul(f).squeeze(1).squeeze(1)
  AvgMLL = (0.5*data_fit - 0.5*logdet)/N - 0.5*math.log(2*math.pi)
  return AvgMLL

def cal_kern_per(X1,X2,period,lengthscale):
  #lengthscale: (B or None) X D
  #period:(B or None) X D
  #X1: (B or None) X N1 X D, X2: (B or None) X N2 X D
  period = period.unsqueeze(-2) # (B or None) X 1 X D
  X1 = X1.div(period).unsqueeze(-2) #shape --> (B or None) X N X 1 X D
  X2 = X2.div(period).unsqueeze(-3) #shape --> (B or None) x 1 x N x D
  X_diff = torch.abs(X1 - X2) #shape --> B x N x N x D
  lengthscale = (lengthscale**2).unsqueeze(-2).unsqueeze(-2) # B X 1 X 1 X D
  K = (-2*(torch.sin(math.pi*X_diff)**2)/lengthscale).exp_() # B X N X N X D
  K = torch.prod(K,-1) # B X N X N
  return K

def cal_kern_rbf(X1,X2,lengthscale):
  #lengthscale: B or None X D
  #X1: B or None X N1 X D, X2: B or None X N2 X D
  lengthscale = lengthscale.unsqueeze(-2)#B X 1 X D
  X1 = X1.div(lengthscale)
  X2 = X2.div(lengthscale)
  X1_norm = torch.sum(X1 ** 2, dim = -1).unsqueeze(-1)#B X N1 X 1
  X2_norm = torch.sum(X2 ** 2, dim = -1).unsqueeze(-2)#B X 1 X N2
  Distance_squared = (X1_norm + X2_norm - 2 * torch.matmul(X1, X2.transpose(-1,-2))).clamp_min_(0)
  K = torch.exp(-Distance_squared) #shape: B X N1 X N2
  return K

def cal_kern_matern(X1,X2,lengthscale,nu=0.5):
  #lengthscale: B X D
  #X1: B X N1 X D, X2: B X N2 X D
  lengthscale = lengthscale.unsqueeze(-2)#B X 1 X D
  X1 = X1.div(lengthscale)
  X2 = X2.div(lengthscale)
  X1_norm = torch.sum(X1 ** 2, dim = -1).unsqueeze(-1)#B X N1 X 1
  X2_norm = torch.sum(X2 ** 2, dim = -1).unsqueeze(-2)#B X 1 X N2
  Distance_squared = (X1_norm + X2_norm - 2 * torch.matmul(X1, X2.transpose(-1,-2))).clamp_min_(1e-30)
  Distance = torch.sqrt(Distance_squared)
  exp_component = torch.exp(-math.sqrt(nu * 2) * Distance)
  if nu == 0.5:
    constant_component = 1
  elif nu == 1.5:
    constant_component = (math.sqrt(3) * Distance).add(1)
  elif nu == 2.5:
    constant_component = (math.sqrt(5) * Distance).add(1).add(5.0 / 3.0 * (Distance) ** 2)
  K = torch.mul(constant_component,exp_component) #shape: B X N1 X N2
  return K

def cal_kern_spec_mix_sep(X1, X2, mu, var, weights):
  #X1: shape B X N1 X D, X2: B X N2 X D
  #mu: B X M X (D or 1)
  #var: B X M X (D or 1)
  #weights: B X M X (D or 1)
  X1 = X1.unsqueeze(-2) #shape --> (B or None) X N X 1 X D
  X2 = X2.unsqueeze(-3) #shape --> B x 1 x N x D
  X_diff = (X1 - X2).unsqueeze(-4) #shape --> B x 1 x N x N x D
  X_diff_squared = X_diff**2
  var = var.unsqueeze(-2).unsqueeze(-2) # shape --> B x M x 1 x 1 x (D or 1)
  mu = mu.unsqueeze(-2).unsqueeze(-2) # shape --> B x M x 1 x 1 x (D or 1)
  kern_all = (weights.unsqueeze(-2).unsqueeze(-2))*torch.exp(-2*(math.pi**2)*X_diff_squared*var)*torch.cos(2*math.pi*X_diff*mu) # shape --> B x M x N x N x D
  kern_all = torch.sum(kern_all,-4) #sum up the average of the mixture of kernels, shape --> B x N x N x D
  kern = torch.prod(kern_all,-1) #shape --> B x N x N
  return kern

def cal_kern_spec_mix_nomu_sep(X1, X2, var, weights):
  X1 = X1.unsqueeze(-2) #shape --> (B or None) X N X 1 X D
  X2 = X2.unsqueeze(-3) #shape --> B x 1 x N x D
  X_diff = (X1 - X2).unsqueeze(-4) #shape --> B x 1 x N x N x D
  X_diff_squared = X_diff**2
  var = var.unsqueeze(-2).unsqueeze(-2) # shape --> B x M x 1 x 1 x (D or 1)
  kern_all = (weights.unsqueeze(-2).unsqueeze(-2))*torch.exp(-2*(math.pi**2)*X_diff_squared*var) # shape --> B x M x N x N x D
  kern_all = torch.sum(kern_all,-4) #sum up the average of the mixture of kernels, shape --> B x N x N x D
  kern = torch.prod(kern_all,-1) #shape --> B x N x N
  return kern

def cal_kern_spec_mix(X1, X2, mu, var, weights):
  #X1: shape B X N1 X D, X2: B X N2 X D
  #mu: B X M X (D or 1)
  #var: B X M X (D or 1)
  #weights: B X M
  X1 = X1.unsqueeze(-2) #shape --> B X N1 X 1 X D
  X2 = X2.unsqueeze(-3) #shape --> B x 1 x N2 x D
  X_diff = (X1 - X2).unsqueeze(-4) #shape --> B x 1 x N1 x N2 x D
  X_diff_squared = X_diff**2
  var = var.unsqueeze(-2).unsqueeze(-2) # shape --> B x M x 1 x 1 x (D or 1)
  mu = mu.unsqueeze(-2).unsqueeze(-2) # shape --> B x M x 1 x 1 x (D or 1)
  log_exp_component = -2*(math.pi**2)*X_diff_squared*var
  exp_component = torch.exp(torch.sum(log_exp_component,-1)) #shape --> B x M x N1 x N2
  cos_component = torch.prod(torch.cos(2*math.pi*X_diff*mu),-1)# product of all D dimensions
  weights = weights.unsqueeze(-1).unsqueeze(-1) # shape --> B x M x 1 x 1
  kern_all = weights*exp_component*cos_component # shape --> B x M x N1 x N2
  kern = torch.sum(kern_all,-3) #sum up the average of the mixture of kernels
  return kern

def cal_kern_spec_mix_nomu(X1, X2, var, weights):
  #X1: shape B X N1 X D, X2: B X N2 X D
  #var: B X M X (D or 1)
  #weights: B X M
  X1 = X1.unsqueeze(-2) #shape --> B X N1 X 1 X D
  X2 = X2.unsqueeze(-3) #shape --> B x 1 x N2 x D
  X_diff = (X1 - X2).unsqueeze(-4) #shape --> B x 1 x N1 x N2 x D
  X_diff_squared = X_diff**2
  var = var.unsqueeze(-2).unsqueeze(-2) # shape --> B x M x 1 x 1 x (D or 1)
  log_exp_component = -2*(math.pi**2)*X_diff_squared*var
  exp_component = torch.exp(torch.sum(log_exp_component,-1)) #shape --> B x M x N1 x N2
  weights = weights.unsqueeze(-1).unsqueeze(-1) # shape --> B x M x 1 x 1
  kern_all = weights*exp_component # shape --> B x M x N1 x N2
  kern = torch.sum(kern_all,-3) #sum up the average of the mixture of kernels
  return kern