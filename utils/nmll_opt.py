import torch
from easydict import EasyDict as edict
from utils.gp_sm_mll import GPMLL_SM
from model.gp.gp_helper import cal_kern_spec_mix_sep, cal_kern_spec_mix_nomu_sep
from model.gp.gp_helper import GP_noise
import time

def nmll_opt(data,model_params,settings):
  model = GPMLL_SM(model_params).to(settings.device)
  train_x = torch.from_numpy(data['X']).float().to(settings.device)
  train_y = torch.from_numpy(data['f']).float().unsqueeze(-1).to(settings.device)
  test_x = torch.from_numpy(data['X_2']).float().to(settings.device)
  test_y = torch.from_numpy(data['f_2']).float().unsqueeze(-1).to(settings.device)
  if settings.opt_is_lbfgs:
    optimizer = torch.optim.LBFGS([
      {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=settings.lr, max_iter=10, tolerance_grad=2.0e-4)
  else:
    optimizer = torch.optim.Adam([
      {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=settings.lr)   
  loss = model(train_x, train_y, settings.epsilon, settings.device)
  model.train()
  time_start = time.time()
  for j in range(settings.training_iter):
    def closure():
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      loss = model(train_x, train_y, settings.epsilon, settings.device)
      # Calc loss and backprop gradients
      loss.backward()
      if settings.is_print:
        grad_norm = 0
        for p in model.parameters():
          if p.requires_grad:
            param_norm = p.grad.data.norm()
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** (1./2)
        print('Iter %d/%d - Loss: %.3f - grad_norm: %.3e' % (j + 1, settings.training_iter, loss.item(), grad_norm))

      return(loss)
    optimizer.step(closure)


  data['nmll_opt_sm'] = model(train_x, train_y, settings.epsilon, settings.device).detach().cpu().numpy()
  data['nmll_opt_sm_test'] = model(test_x, test_y, settings.epsilon, settings.device).detach().cpu().numpy()
  sm_params = edict()
  var_sm = torch.clamp(model.var, min=-30.0, max=30.0)
  var_sm = torch.exp(var_sm)
  weights_sm = torch.clamp(model.weights, min=-30.0, max=30.0)
  weights_sm = torch.softmax(weights_sm, dim=-2)
  if not model_params.is_no_mu:
    mu_sm = torch.clamp(model.mu, min=-30.0, max=30.0)
    mu_sm = torch.exp(mu_sm)
    sm_params.mu = mu_sm.detach()
  sm_params.var = var_sm.detach()
  sm_params.weights = weights_sm.detach()
  time_end = time.time()
  time_elapsed = time_end - time_start
  return sm_params, time_elapsed

def nmll_opt_gp(data, model_params, settings):
  train_x = torch.from_numpy(data['X']).float().to(settings.device)
  train_y = torch.from_numpy(data['f']).float().unsqueeze(-1).to(settings.device)
  test_x = torch.from_numpy(data['X_2']).float().to(settings.device)
  test_y = torch.from_numpy(data['f_2']).float().unsqueeze(-1).to(settings.device)
  
  sm_params, time_elapsed = nmll_opt(data,model_params,settings)
  if not model_params.is_no_mu: 
    K11 = cal_kern_spec_mix_sep(train_x, train_x, sm_params.mu, sm_params.var, sm_params.weights)
    K12 = cal_kern_spec_mix_sep(train_x, test_x, sm_params.mu, sm_params.var, sm_params.weights)
    K22 = cal_kern_spec_mix_sep(test_x, test_x, sm_params.mu, sm_params.var, sm_params.weights)
  else:
    K11 = cal_kern_spec_mix_nomu_sep(train_x, train_x, sm_params.var, sm_params.weights)
    K12 = cal_kern_spec_mix_nomu_sep(train_x, test_x, sm_params.var, sm_params.weights)
    K22 = cal_kern_spec_mix_nomu_sep(test_x, test_x, sm_params.var, sm_params.weights)

  mu_test, var_test = GP_noise(train_y, K11, K12, K22, settings.epsilon, settings.device)
  mu_test = mu_test.detach().squeeze(-1).cpu().numpy()
  var_test = var_test.detach().squeeze(-1).cpu().numpy().diagonal()
  return mu_test, var_test, sm_params, time_elapsed