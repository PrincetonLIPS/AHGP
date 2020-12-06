import torch
import torch.nn as nn
import torch.nn.functional as F
from ahgp.nn.module import *
from ahgp.gp.gp_helper import cal_kern_spec_mix_nomu_sep, cal_marg_likelihood


__all__ = ['GpSMDoubleAttNoMu']
    
"""
    M: # of mixtures
    D: # input dimension
"""

class GpSMDoubleAttNoMu(nn.Module):

  def __init__(self, config):
    super(GpSMDoubleAttNoMu, self).__init__()
    self.epsilon = config.epsilon
    self.input_dim = config.input_dim + 1
    self.output_dim = 1
    self.is_param_dim_diff = config.is_param_dim_diff
    self.num_attentions1 = config.num_attentions1
    self.num_attentions2 = config.num_attentions2
    self.att1_hidden_dim = config.att1_hidden_dim
    self.att2_hidden_dim = config.att2_hidden_dim
    self.num_layer_var = config.num_layer_var
    self.num_layer_weights = config.num_layer_weights
    self.hidden_dim_var = config.hidden_dim_var
    self.hidden_dim_weights = config.hidden_dim_weights
    self.num_mix = config.num_mix
    self.dropout = config.dropout if hasattr(config,
                                                   'dropout') else 0.0
    if len(self.hidden_dim_var) != self.num_layer_var or len(self.hidden_dim_weights) != self.num_layer_weights:
      raise ValueError("number of layers and dimension list does not match!")

    dim_list_var = [self.att2_hidden_dim] + self.hidden_dim_var + [self.num_mix]

    dim_list_weights = [self.att2_hidden_dim] + self.hidden_dim_weights + [self.num_mix]

    self.input_projection = Linear(self.input_dim, self.att1_hidden_dim, bias=False)

    self.self_attentions1 = nn.ModuleList([
      Attention(self.att1_hidden_dim)
      for tt in range(self.num_attentions1)
    ])
    
    self.self_attentions2 = nn.ModuleList([
      Attention(self.att2_hidden_dim)
      for tt in range(self.num_attentions2)
    ])
    
    self.filter_var = nn.ModuleList([
      Linear(dim_list_var[tt], dim_list_var[tt + 1])
      for tt in range(self.num_layer_var+1)
    ])
    
    self.filter_weights = nn.ModuleList([
      Linear(dim_list_weights[tt], dim_list_weights[tt + 1])
      for tt in range(self.num_layer_weights+1)
    ])
    
    self.log_softmax = nn.LogSoftmax(dim=1)
    self.softmax = nn.Softmax(dim=-1)
  def forward(self, X_data_tr, X_data_val, F_data_tr, F_data_val, node_mask_tr, dim_mask, kernel_mask_val, diagonal_mask_val, N_data_val, device=torch.device('cpu'), eval_mode = False, X_data_test = None, F_data_test = None, kernel_mask_test=None, diagonal_mask_test=None, N_data_test=None):
    """
      X_data: B X N X D
      F_data: B X N
      initial node_mask: B X N
      X_input: (B X D) X N X 2
      dim_mask: B X D
      node_mask: B X N
    """
    #preprocess data to each dimension
    batch_size = X_data_tr.shape[0]
    max_dim = X_data_tr.shape[2]
    max_num_data = X_data_tr.shape[1]
    f_tr = F_data_tr.unsqueeze(-1) # B X N X 1
    f_val = F_data_val.unsqueeze(-1) # B X N X 1
    f_expand = f_tr.expand(-1,-1,max_dim) #B X N X D
    f_expand = f_expand * dim_mask.unsqueeze(-2)
    X_input = torch.cat((X_data_tr.unsqueeze(-1),f_expand.unsqueeze(-1)), -1) # B X N X D X 2
    X_input = X_input.permute(0,2,1,3)
    X_input = X_input.reshape(-1,max_num_data,2) # (B X D) X N X 2
    node_mask_tr = node_mask_tr.repeat(1,max_dim) # B X (D X N)
    node_mask_tr = node_mask_tr.reshape(-1,max_num_data) # (B X D) X N

    encoder_input = self.input_projection(X_input)
    # propagation
    for attention in self.self_attentions1:
      encoder_input, attns = attention(encoder_input,encoder_input,encoder_input,mask=node_mask_tr)


    N = torch.sum(node_mask_tr,dim=1) # N: (B x D) X 0
    encoder_input = encoder_input * node_mask_tr.unsqueeze(-1)
    
    dim_encoder_input = torch.sum(encoder_input,1)/(N.unsqueeze(-1)) # (B X D) X hidden_dim
    dim_encoder_input = dim_encoder_input.reshape(batch_size,max_dim,dim_encoder_input.shape[-1]) # B X D X hidden_dim


    #feed encoder_input to next attention network for dimension
    for attention in self.self_attentions2:
      dim_encoder_input, attns = attention(dim_encoder_input,dim_encoder_input,dim_encoder_input,mask=dim_mask)

    dim_encoder_mask = dim_mask.unsqueeze(-1) # B X D X 1
    dim_encoder_input = dim_encoder_input * dim_encoder_mask # B X D X hidden_dim
    state_var = dim_encoder_input.clone()
    state_weights = dim_encoder_input.clone()

    for tt in range(self.num_layer_var):
      state_var = F.relu(self.filter_var[tt](state_var))
      state_var = F.dropout(state_var, self.dropout, training=self.training)

    for tt in range(self.num_layer_weights):
      state_weights = F.relu(self.filter_weights[tt](state_weights))
      state_weights = F.dropout(state_weights, self.dropout, training=self.training)

    var = self.filter_var[-1](state_var)
    var = torch.clamp(var,min=-10,max=10)
    var = torch.exp(var)
    var = var * dim_mask.unsqueeze(-1)

    weights = self.filter_weights[-1](state_weights)
    weights = torch.clamp(weights,min=-10.0,max=10.0)
    weights = self.softmax(weights)

    var = var.permute(0,2,1) # B X M X D
    weights = weights.permute(0,2,1) # B X M X D
    K_val = cal_kern_spec_mix_nomu_sep(X_data_val, X_data_val, var, weights)
    nmll = -cal_marg_likelihood(K_val, f_val, self.epsilon, kernel_mask_val, diagonal_mask_val, N_data_val.float(), device)


    if eval_mode:
      f_test = F_data_test.unsqueeze(-1) # B X N X 1
      K_test = cal_kern_spec_mix_nomu_sep(X_data_test, X_data_test, var, weights)
      nmll_test = -cal_marg_likelihood(K_test, f_test, self.epsilon, kernel_mask_test, diagonal_mask_test, N_data_test.float(), device)
      return var, weights, nmll, nmll_test

    return var, weights, nmll