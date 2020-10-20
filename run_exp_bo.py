import os
import sys
import pickle
import torch
from torch import nn
import logging
import traceback
import numpy as np
from pprint import pprint
from easydict import EasyDict as edict
from tqdm import tqdm
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_bo_config
from utils.gp_helper import cal_kern_spec_mix_sep, cal_kern_spec_mix_nomu_sep, GP_noise, standardize
from utils.bo_functions import *
from utils.bo_model import BO_GP_Model, GPyModelWrapperTime
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.model_wrappers import GPyModelWrapper
from model import *
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import GPy

### --- Figure config
LEGEND_SIZE = 15

torch.set_printoptions(profile='full')
torch.set_printoptions(precision=4,linewidth=200)
np.set_printoptions(precision=4,linewidth=150)

def bo_loop(config, image_path, ai_model=None):
  target_function, space = eval(config.name)()
  data_dim = config.data_dim
  num_mix = config.num_mix
  init_num_data = config.init_num_data
  interval_std = config.interval_std
  interval = np.zeros((1,data_dim))
  std = np.zeros((1,data_dim))
  mean = np.zeros((1,data_dim))
  #set up data, scaling
  for ii in range(data_dim):
    interval[0,ii] = space.parameters[ii].max - space.parameters[ii].min
    std[0,ii] = interval[0,ii]/interval_std
    mean[0,ii] = (space.parameters[ii].max + space.parameters[ii].min)/2
    space.parameters[ii].min = (space.parameters[ii].min - mean[0,ii])/std[0,ii]
    space.parameters[ii].max = (space.parameters[ii].max - mean[0,ii])/std[0,ii]


  results_list = [None] * config.repeated_runs
  best_value_per_iter = np.zeros((config.repeated_runs, config.bo_iter))
  npr = np.random.RandomState(123)
  for ii in tqdm(range(config.repeated_runs)):
    #initialize data points
    X_init = (npr.rand(init_num_data,data_dim)-0.5)*interval + mean
    X_init_norm = (X_init - mean)/std
    Y_init = target_function(X_init)
    Y_init_norm, mean_Y, std_Y = standardize(Y_init)
    
    # normalized function
    function_norm= lambda x: (target_function(x*std+mean)- mean_Y)/std_Y

    if config.is_GPY:
      kernel = GPy.kern.RBF(input_dim=data_dim, variance=npr.rand(1), lengthscale=npr.rand(data_dim), ARD=True)
      #kernel.variance.fix()
      for jj in range(num_mix-1):
        rbf_new = GPy.kern.RBF(input_dim=data_dim, variance=npr.rand(1), lengthscale=npr.rand(data_dim), ARD=True)
        #rbf_new.variance.fix()
        kernel = kernel + rbf_new
      if config.is_sparseGP:
        z = (np.random.rand(config.num_inducing_pts,data_dim)-0.5)*interval_std
        model_gp = GPy.models.SparseGPRegression(X_init_norm, Y_init_norm, kernel, Z=z)
      else:
        model_gp = GPy.models.GPRegression(X_init_norm, Y_init_norm, kernel)

      model_gp.Gaussian_noise.variance = config.epsilon
      model_gp.Gaussian_noise.variance.fix()
      model_emukit = GPyModelWrapperTime(model_gp)
      model_emukit.optimize()
    else:
      #Set up BO_GP_Model
      model_emukit = BO_GP_Model(X_init_norm, Y_init_norm, config, ai_model)
      model_emukit.optimize()
      model_emukit.set_kernel()

    expected_improvement = ExpectedImprovement(model=model_emukit)

    bayesopt_loop = BayesianOptimizationLoop(model=model_emukit,
                                            space=space,
                                            acquisition=expected_improvement,
                                            batch_size=1)
    max_iterations = config.bo_iter
    bayesopt_loop.run_loop(function_norm, max_iterations)
    results = bayesopt_loop.get_results()
    #scale back the x and y
    results_save = edict()
    results_save.best_found_value_per_iteration = results.best_found_value_per_iteration[init_num_data:] * std_Y.item() + mean_Y.item()
    best_value_per_iter[ii,:] = results_save.best_found_value_per_iteration
    results_save.minimum_value = results.minimum_value * std_Y.item() + mean_Y.item()
    results_save.minimum_location = results.minimum_location * std.squeeze(0) + mean.squeeze(0)
    results_save.time_elapsed = model_emukit.time_count
    print(model_emukit.time_count)
    results_list[ii] = results_save
  

  best_value_mean = np.mean(best_value_per_iter,0)
  best_value_std = np.std(best_value_per_iter,0)
  plt.figure(figsize=(12, 8))
  plt.fill_between(np.arange(max_iterations)+1, best_value_mean-0.2*best_value_std, best_value_mean+0.2*best_value_std, color='red', alpha=0.15)
  plt.plot(np.arange(max_iterations)+1, best_value_mean, 'or-', lw=2, label='Best found function value')
  plt.legend(loc=2, prop={'size': LEGEND_SIZE})
  plt.xlabel(r"iteration")
  plt.ylabel(r"$f(x)$")
  plt.grid(True)
  plt.savefig(image_path, format='pdf')

  return results_list

def main():
  args = parse_arguments()
  config = get_bo_config(args.config_file)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()
  device = torch.device('cuda' if config.use_gpu else 'cpu')

  # log info
  log_file = os.path.join(config.save_dir,
                          "log_exp_{}.txt".format(config.run_id))
  logger = setup_logging(args.log_level, log_file)
  logger.info("Writing log file to {}".format(log_file))
  logger.info("Exp instance id = {}".format(config.run_id))
  logger.info("Exp comment = {}".format(args.comment))
  logger.info("Config =")
  print(">" * 80)
  pprint(config)
  print("<" * 80)


  #load model
  model = eval(config.model.name)(config.model)
  model_snapshot = torch.load(config.model.pretrained_model, map_location=device)
  model.load_state_dict(model_snapshot["model"], strict=True)
  model.to(device)
  if config.use_gpu:
    model = nn.DataParallel(model, device_ids=config.gpus).cuda()

  if config.bo.is_GPY:
    if config.bo.is_sparseGP:
      image_path =os.path.join(config.save_dir, config.bo.name + str(config.bo.num_inducing_pts) + 'p_sparseGP.pdf')
    else:
      image_path =os.path.join(config.save_dir, config.bo.name + '_fullGP.pdf')
  else:
    if config.bo.is_ai:
      image_path =os.path.join(config.save_dir, config.bo.name + '_ai.pdf')
    else:
      image_path =os.path.join(config.save_dir, config.bo.name + 'iter' +str(config.bo.opt_iter) + 'lr' + str(config.bo.opt_lr) +'opt.pdf')
  # Run the experiment
  results_list = bo_loop(config.bo, image_path, model)
  if config.bo.is_GPY:
    if config.bo.is_sparseGP:
      pickle.dump(results_list,
                  open(os.path.join(config.save_dir, config.bo.name + str(config.bo.num_inducing_pts) + 'p_sparseGP_results.p'), 'wb'))
    else:
      pickle.dump(results_list,
                  open(os.path.join(config.save_dir, config.bo.name + '_fullGP_results.p'), 'wb'))
  else:
    if config.bo.is_ai:
      pickle.dump(results_list,
                  open(os.path.join(config.save_dir, config.bo.name + '_ai_results.p'), 'wb'))
    else:
      pickle.dump(results_list,
                  open(os.path.join(config.save_dir, config.bo.name + 'iter' +str(config.bo.opt_iter) + 'lr' + str(config.bo.opt_lr) + 'opt_results.p'), 'wb'))
  sys.exit(0)


if __name__ == "__main__":
  main()
