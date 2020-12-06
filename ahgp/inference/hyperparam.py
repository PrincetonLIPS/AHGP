import torch
from easydict import EasyDict as edict
import yaml
from model.nn import *
from model.gp.gp_helper import cal_kern_spec_mix_nomu_sep, cal_marg_likelihood_single, standardize, GP_noise

def hyperparam(x_t,y_t,x_v,model_config_filename,use_gpu=False):

  model_conf = edict(yaml.load(open(model_config_filename, 'r'),Loader=yaml.FullLoader))
  use_gpu = False
  device = torch.device('cuda' if use_gpu else 'cpu')
  pretrained_model_filename = model_conf.pretrained_model
  data_dim = x_t.shape[1]
  num_data = x_t.shape[0]

  data = {}
  data['X'] = x_t
  data['f'] = y_t
  data['X_2'] = x_v
  data['X_data'] =torch.from_numpy(data['X']).float().unsqueeze(0).to(device) # 1 X N X D
  data['F'] = torch.from_numpy(data['f']).float().unsqueeze(0).to(device) # 1 X N
  data['node_mask'] = torch.ones(num_data).unsqueeze(0).to(device) # 1 X N
  data['diagonal_mask'] = torch.zeros(num_data).unsqueeze(0).to(device) # 1 X N
  data['dim_mask'] = torch.ones(data_dim).unsqueeze(0).to(device) # 1 X D
  data['kernel_mask'] = torch.ones(num_data,num_data).unsqueeze(0).to(device) # 1 X N X N
  data['N'] = torch.ones(1).to(device) * num_data # 1
  #create model and load pretrained model
  model = eval(model_conf.name)(model_conf)
  model_snapshot = torch.load(pretrained_model_filename, map_location=device)
  model.load_state_dict(model_snapshot["model"], strict=True)
  model.to(device)
  model.eval()
  with torch.no_grad():
    if model_conf.name == 'GpSMDoubleAtt':
      mu, var, weights, nmll = model(data['X_data'],data['X_data'],data['F'],data['F'],data['node_mask'],data['dim_mask'],data['kernel_mask'],data['diagonal_mask'],data['N'], device = device)
    elif model_conf.name == 'GpSMDoubleAttNoMu':
      var, weights, nmll = model(data['X_data'],data['X_data'],data['F'],data['F'],data['node_mask'],data['dim_mask'],data['kernel_mask'],data['diagonal_mask'],data['N'], device = device)
    else:
      raise ValueError("No model of given name!")

  hyper_params = edict()
  hyper_params.var = var.squeeze(0)
  hyper_params.weights = weights.squeeze(0)

  if not model_conf.is_no_mu:
    hyper_params.mu = mu.squeeze(0)

  return hyper_params