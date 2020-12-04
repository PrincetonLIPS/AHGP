import torch
import torch.nn as nn
from model.gp.gp_helper import cal_kern_spec_mix_sep, cal_kern_spec_mix_nomu_sep, cal_marg_likelihood_single

__all__ = ['GPMLL_SM']

class GPMLL_SM(nn.Module):
  def __init__(self,model_params):
    super(GPMLL_SM, self).__init__()
    self.is_no_mu = model_params.is_no_mu
    if model_params.warm_start:
      if not self.is_no_mu:
        self.mu = nn.Parameter(model_params.mu_init, requires_grad=True)
      self.var = nn.Parameter(model_params.var_init, requires_grad=True)
      self.weights = nn.Parameter(model_params.weights_init, requires_grad=True)
    else:
      if model_params.is_dim_diff:
        kernel_dim = model_params.input_dim
      else:
        kernel_dim = 1
      if not self.is_no_mu:
        self.mu = nn.Parameter(torch.randn((model_params.num_mix,kernel_dim), requires_grad=True, dtype=torch.float))
      self.var = nn.Parameter(torch.randn((model_params.num_mix,kernel_dim), requires_grad=True, dtype=torch.float))
      self.weights = nn.Parameter(torch.randn((model_params.num_mix,kernel_dim), requires_grad=True, dtype=torch.float))

    self.softmax = nn.Softmax(dim=-2)
  def forward(self, X, y, epsilon, device):
    var = torch.clamp(self.var,min=-20.0,max=20.0)
    var = torch.exp(var)
    # var = torch.log(1+var)
    weights = torch.clamp(self.weights,min=-20.0,max=20.0)
    weights = self.softmax(weights)*2
    if self.is_no_mu:
      kern_sm = cal_kern_spec_mix_nomu_sep(X, X, var, weights)
    else:
      mu = torch.clamp(self.mu,min=-20.0,max=20.0)
      mu = torch.exp(mu)
      kern_sm = cal_kern_spec_mix_sep(X, X, mu, var, weights)

    nmll = -cal_marg_likelihood_single(kern_sm, y, epsilon, device)
    return nmll