import math
import numpy as np
from typing import Tuple
import time
import torch
from scipy import special
from model.gp_sm_mll import GPMLL_SM
from easydict import EasyDict as edict
from utils.gp_helper import cal_kern_spec_mix_sep, cal_kern_spec_mix_nomu_sep, GP_noise, standardize
from emukit.core.interfaces.models import IModel
from emukit.quadrature.interfaces import IBaseGaussianProcess
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel
from emukit.quadrature.kernels.bounds import BoxBounds
from emukit.quadrature.kernels.integration_measures import IntegrationMeasure
from emukit.model_wrappers import GPyModelWrapper, RBFGPy
import pdb
from GPy.models import GPRegression, SparseGPRegression
import GPy

class BO_GP_Model(IModel):
  """
      This is a Gaussian Process Model that can be used by emukit.
  """
  def __init__(self, X, Y, config, ai_model=None):
    self.time_count = 0
    self.data_dim = config.data_dim
    self.npr = np.random.RandomState(123)
    self.ai = config.is_ai
    self.ai_model = ai_model
    self.is_GPY = config.is_GPY
    self.is_sparseGP = config.is_sparseGP
    self.is_no_mu = config.is_no_mu
    self.is_dim_diff = config.is_dim_diff
    self.X_train = X
    self.Y_train = Y
    self.kernel = None
    self.noise_level = config.epsilon
    self.sm_params = edict()
    self.device = torch.device('cpu')
    self.X_test_torch = None
    self.X_train_torch = torch.from_numpy(X).float().to(self.device)
    self.Y_train_torch = torch.from_numpy(Y).float().to(self.device)
    self.num_mix = config.num_mix
    if config.is_GPY:
      kernel = GPy.kern.RBF(input_dim=config.data_dim, variance=self.npr.rand(1)*0.5, lengthscale=self.npr.rand(config.data_dim), ARD=True)
      # kernel.variance.fix()
      for jj in range(self.num_mix-1):
        rbf_new = GPy.kern.RBF(input_dim=config.data_dim, variance=self.npr.rand(1)*0.5, lengthscale=self.npr.rand(config.data_dim), ARD=True)
        # rbf_new.variance.fix()
        kernel = kernel + rbf_new
      if self.is_sparseGP:
        num_inducing_pts = config.num_inducing_pts
        z = (np.random.rand(num_inducing_pts,config.data_dim)-0.5)*config.interval_std
        self.model_gpy = GPy.models.SparseGPRegression(X, Y, kernel, Z=z)
      else:
        self.model_gpy = GPy.models.GPRegression(X, Y, kernel)

      self.model_gpy.Gaussian_noise.variance = config.epsilon
      self.model_gpy.Gaussian_noise.variance.fix()
    else:
      self.nmll_vec = []
      if not self.ai:
        self.opt_iter = config.opt_iter
        self.opt_lr = config.opt_lr
        self.opt_is_print = config.opt_is_print
        model_params = edict()
        model_params.input_dim = self.X.shape[1]
        model_params.num_mix = self.num_mix
        model_params.is_dim_diff = self.is_dim_diff
        model_params.is_no_mu = self.is_no_mu
        model_params.warm_start = False
        self.gp_sm_model = GPMLL_SM(model_params)
        self.settings = edict()
        self.settings.epsilon = self.noise_level
        self.settings.lr = self.opt_lr
        self.settings.training_iter = self.opt_iter
        self.settings.is_print = self.opt_is_print
        self.settings.device = self.device
        self.settings.opt_method = config.opt_method
  def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict mean and variance values for given points
    :param X: array of shape (n_points x n_inputs) of points to run prediction for
    :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
    """
    # !!!
    self.X_test_torch = torch.from_numpy(X).float().to(self.device)
    K11 = cal_kern_spec_mix_nomu_sep(self.X_train_torch, self.X_train_torch, self.sm_params.var, self.sm_params.weights)
    K12 = cal_kern_spec_mix_nomu_sep(self.X_train_torch, self.X_test_torch, self.sm_params.var, self.sm_params.weights)
    K22 = cal_kern_spec_mix_nomu_sep(self.X_test_torch, self.X_test_torch, self.sm_params.var, self.sm_params.weights)
    mu, var = GP_noise(self.Y_train_torch , K11, K12, K22, self.noise_level, self.device)
    mu = mu.cpu().numpy()
    var = np.diag(var.cpu().numpy())[:,None]
    return mu, var

  def set_kernel(self):
    K11 = cal_kern_spec_mix_nomu_sep(self.X_train_torch, self.X_train_torch, self.sm_params.var, self.sm_params.weights)
    self.kernel = K11.detach().cpu().numpy()

  def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Sets training data in model
    :param X: new points
    :param Y: function values at new points X
    """
    self.X_train = X
    self.Y_train = Y
    self.X_train_torch = torch.from_numpy(X).float().to(self.device)
    self.Y_train_torch = torch.from_numpy(Y).float().to(self.device)
    K11 = cal_kern_spec_mix_nomu_sep(self.X_train_torch, self.X_train_torch, self.sm_params.var, self.sm_params.weights)
    self.kernel = K11.detach().cpu().numpy()


  def optimize(self) -> None:
    """
    Optimize hyper-parameters of model
    """
    if self.is_GPY:
      self.model_gpy.randomize()
      self.model_gpy.optimize_restarts(1, robust=True)
      nmll = self.model_gpy.log_likelihood()
      self.nmll_vec.append(nmll)
      length = self.model_gpy.kern['.*lengthscale'].values()
      length = length.reshape(-1,self.data_dim)
      weights = self.model_gpy.kern['.*variance'].values()
      self.sm_params.weights = torch.from_numpy(weights).float().unsqueeze(-1)
      self.sm_params.var = torch.from_numpy(1/4/math.pi**2/length**2).float()
    else:
      if self.ai:
        num_data = self.X_train_torch.shape[0]
        data_dim = self.X_train_torch.shape[1]
        X_data =self.X_train_torch.unsqueeze(0) # 1 X N X D
        F = self.Y_train_torch.unsqueeze(0).squeeze(-1) # 1 X N
        node_mask = torch.ones(1, num_data) # 1 X N
        diagonal_mask = torch.zeros(1, num_data) # 1 X N
        dim_mask = torch.ones(1, data_dim) # 1 X D
        kernel_mask = torch.ones(1, num_data, num_data) # 1 X N X N
        N = torch.ones(1) * num_data # 1
        #Timer starts
        time_start = time.time()
        if self.is_no_mu:
          var, weights, nmll = self.ai_model(X_data,X_data,F,F,node_mask,dim_mask,kernel_mask,diagonal_mask,N,kernel_mask,diagonal_mask,N,self.device)
        #Timer ends
        time_end = time.time()
        self.nmll_vec.append(nmll.item())
        self.time_count = self.time_count + time_end - time_start
        self.sm_params.var = var.detach().squeeze(0) # M X D
        self.sm_params.weights = weights.detach().squeeze(0)
      else:
        if self.settings.opt_method == 'Adam':
          optimizer = torch.optim.Adam([
          {'params': self.gp_sm_model.parameters()},  # Includes GaussianLikelihood parameters
          ], lr=self.settings.lr) 
        elif self.settings.opt_method == 'LBFGS':
          optimizer = torch.optim.LBFGS([
            {'params': self.gp_sm_model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=self.settings.lr, max_iter=10, tolerance_grad=2.0e-4)
        else:
          raise ValueError("No opt method of given name!")
        loss = self.gp_sm_model(self.X_train_torch, self.Y_train_torch, self.settings.epsilon, self.settings.device)
        self.gp_sm_model.train()

        #Timer starts
        time_start = time.time()
        for j in range(self.settings.training_iter):
          def closure():
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            loss = self.gp_sm_model(self.X_train_torch, self.Y_train_torch, self.settings.epsilon, self.settings.device)
            # Calc loss and backprop gradients
            loss.backward()
            if self.settings.is_print:
              grad_norm = 0
              for p in self.gp_sm_model.parameters():
                if p.requires_grad:
                  param_norm = p.grad.data.norm()
                  grad_norm += param_norm.item() ** 2
              grad_norm = grad_norm ** (1./2)
              print('Iter %d/%d - Loss: %.3f - grad_norm: %.3e' % (j + 1, self.settings.training_iter, loss.item(), grad_norm))
            return loss
          loss = optimizer.step(closure)
          
        nmll = loss.detach().cpu().numpy()
        self.nmll_vec.append(nmll.item())
        var_sm = torch.clamp(self.gp_sm_model.var, min=-30.0, max=30.0)
        var_sm = torch.exp(var_sm)
        weights_sm = torch.clamp(self.gp_sm_model.weights, min=-30.0, max=30.0)
        weights_sm = torch.softmax(weights_sm, dim=-2)
        #Timer ends
        time_end = time.time()
        self.time_count = self.time_count + time_end - time_start

        if not self.is_no_mu:
          mu_sm = torch.clamp(self.gp_sm_model.mu, min=-30.0, max=30.0)
          mu_sm = torch.exp(mu_sm)
          self.sm_params.mu = mu_sm.detach()
        self.sm_params.var = var_sm.detach()
        self.sm_params.weights = weights_sm.detach()
  @property
  def X(self):
    return self.X_train

  @property
  def Y(self):
    return self.Y_train

class QuadratureKernelCustom:
  def __init__(self,gp_model,integral_bounds,variable_names: str=''):
    self.gp_model = gp_model
    reasonable_box_bounds = integral_bounds
    self.integral_bounds = BoxBounds(name=variable_names, bounds=integral_bounds)
    self.reasonable_box_bounds = BoxBounds(name=variable_names, bounds=reasonable_box_bounds)
    self.input_dim = self.reasonable_box_bounds.dim
    self.integral_lower_bounds = self.integral_bounds.lower_bounds
    self.integral_upper_bounds = self.integral_bounds.upper_bounds
  @property
  def std(self):
    return np.expand_dims( np.sqrt(self.gp_model.sm_params.var.cpu().numpy()), axis=1) # M X 1 X D

  @property
  def weights(self):
    return np.expand_dims(self.gp_model.sm_params.weights.cpu().numpy(), axis=1) # M X 1 X D

  def K(self, X: np.ndarray, x_2: np.ndarray) -> np.ndarray:
    X_torch = torch.from_numpy(X).float().to(self.gp_model.device)
    X_torch_2 = torch.from_numpy(x_2).float().to(self.gp_model.device)
    K = cal_kern_spec_mix_nomu_sep(X_torch, X_torch_2, self.gp_model.sm_params.var, self.gp_model.sm_params.weights)
    return K.detach().cpu().numpy()

  def qK(self, x2: np.ndarray) -> np.ndarray:
    """
    spectral mixture product kernel with the first component integrated out aka kernel mean
    :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
    :returns: kernel mean at location x2, shape (1, N)
    """
    vector_diff_low = (self.integral_lower_bounds - x2)[None,:,:] # 1 X N X D
    vector_diff_up = (self.integral_upper_bounds - x2)[None,:,:] # 1 X N X D
    vector_diff_low = vector_diff_low * np.sqrt(2) * self.std * math.pi # M X N X D
    vector_diff_up = vector_diff_up * np.sqrt(2) * self.std * math.pi # M X N X D
    erf_low = special.erf(vector_diff_low)
    erf_up = special.erf(vector_diff_up)
    integral = np.sum((erf_up - erf_low) * self.weights / (self.std * 2 * np.sqrt(2*math.pi)) , axis=0) # N X D
    kernel_mean = integral.prod(axis=1) # N
    return kernel_mean[None,:]
  def qKq(self) -> float:
    """
    spectral mixture product kernel integrated over both arguments x1 and x2
    :returns: double integrated kernel
    """
    diff_bounds = self.integral_upper_bounds - self.integral_lower_bounds # 1 X D
    scale = self.std.squeeze(-2) * math.pi * np.sqrt(2) # M X D
    diff_bounds_scaled = diff_bounds * scale # M X D
    erf_term = special.erf(diff_bounds_scaled) * diff_bounds_scaled * np.sqrt(math.pi) # M X D
    exp_term = np.exp(-diff_bounds_scaled**2) - 1. # M X D
    qKq_vec = np.sum( self.weights.squeeze(-2) * (erf_term + exp_term) / (scale**2), axis=0) #D
    return float(qKq_vec.prod())

class BaseGaussianProcessCustomModel(IBaseGaussianProcess):
  """
  Wrapper for self-created model
  An instance of this can be passed as 'base_gp' to a WarpedBayesianQuadratureModel object.
  Note that the GPy cannot take None as initial values for X and Y. Thus we initialize it with some values. These will
  be re-set in the WarpedBayesianQuadratureModel.
  """
  def __init__(self, kern: QuadratureKernelCustom, gp_model, noise_free: bool=True):
    """
    :param kern: a quadrature kernel
    :param gpy_model: A GPy GP regression model, GPy.models.GPRegression
    :param noise_free: if False, the observation noise variance will be treated as a model parameter,
    if True it is set to 1e-10, defaults to True
    """
    super().__init__(kern=kern)
    self.model = gp_model
    if noise_free:
        self.model.noise_level = gp_model.noise_level
  @property
  def X(self) -> np.ndarray:
    return self.model.X_train
  @property
  def Y(self) -> np.ndarray:
    return self.model.Y_train

  @property
  def observation_noise_variance(self) -> np.float:
    """
    Gaussian observation noise variance
    :return: The noise variance from some external GP model
    """
    return self.model.noise_level

  def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Sets training data in model
    :param X: New training features
    :param Y: New training outputs
    """
    self.model.set_data(X, Y)

  def predict(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predictive mean and covariance at the locations X_pred
    :param X_pred: points at which to predict, with shape (number of points, dimension)
    :return: Predictive mean, predictive variances shapes (num_points, 1) and (num_points, 1)
    """
    mu, var = self.model.predict(X_pred)
    return mu, var

  def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predictive mean and covariance at the locations X_pred
    :param X_pred: points at which to predict, with shape (num_points, input_dim)
    :return: Predictive mean, predictive full covariance shapes (num_points, 1) and (num_points, num_points)
    """
    mu, var = self.model.predict(X_pred)
    return mu, var

  def solve_linear(self, z: np.ndarray) -> np.ndarray:
    """
    Solve the linear system G(X, X)x=z for x.
    G(X, X) is the Gram matrix :math:`G(X, X) = K(X, X) + \sigma^2 I`, of shape (num_dat, num_dat) and z is a
    matrix of shape (num_dat, num_obs).
    :param z: a matrix of shape (num_dat, num_obs)
    :return: the solution to the linear system G(X, X)x = z, shape (num_dat, num_obs)
    """
    K11 = cal_kern_spec_mix_nomu_sep(self.model.X_train_torch, self.model.X_train_torch, self.model.sm_params.var, self.model.sm_params.weights)
    G = (K11 + torch.eye(K11.shape[0])*self.model.noise_level).detach().cpu().numpy()
    result = np.linalg.solve(G, z)
    return result

  def graminv_residual(self) -> np.ndarray:
    """
    The inverse Gram matrix multiplied with the mean-corrected data
    ..math::
        (K_{XX} + \sigma^2 I)^{-1} (Y - m(X))
    where the data is given by {X, Y} and m is the prior mean and sigma^2 the observation noise
    :return: the inverse Gram matrix multiplied with the mean-corrected data with shape: (number of datapoints, 1)
    """
    K11 = cal_kern_spec_mix_nomu_sep(self.model.X_train_torch, self.model.X_train_torch, self.model.sm_params.var, self.model.sm_params.weights)
    G= (K11 + torch.eye(K11.shape[0])*self.model.noise_level).detach().cpu().numpy()
    result = np.linalg.solve(G, self.Y)
    return result

  def optimize(self) -> None:
    """ Optimize the hyperparameters of the GP """
    self.model.optimize()




class GPyModelWrapperTime(GPyModelWrapper):
  def __init__(self, gpy_model, n_restarts: int = 1):
    GPyModelWrapper.__init__(self, gpy_model, n_restarts)
    self.time_count = 0

  def optimize(self):
    """
    Optimizes model hyper-parameters
    """
    time_start = time.time()
    self.model.randomize()
    self.model.optimize_restarts(self.n_restarts, robust=True)
    #print(-self.model.log_likelihood()/self.model.Y.size)
    time_end = time.time()
    self.time_count = self.time_count + time_end - time_start