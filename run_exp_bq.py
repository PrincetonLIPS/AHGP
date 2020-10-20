import os
import sys
import pickle
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_bo_config
from utils.gp_helper import standardize
from utils.bo_functions import *
from utils.bo_model import BO_GP_Model, BaseGaussianProcessCustomModel, QuadratureKernelCustom
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.test_functions.quadrature.baselines import univariate_approximate_ground_truth_integral, bivariate_approximate_ground_truth_integral
from emukit.test_functions.quadrature import hennig1D, sombrero2D, hennig2D, circular_gaussian
from emukit.core.optimization import LocalSearchAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
from emukit.quadrature.acquisitions import IntegralVarianceReduction
from model import *
import matplotlib.pyplot as plt
import time

### Figure config
LEGEND_SIZE = 15
FIGURE_SIZE = (12, 8)
torch.set_printoptions(profile='full')
torch.set_printoptions(precision=4,linewidth=200)
np.set_printoptions(precision=4,linewidth=150)

def bq_loop(config, ai_model=None):
  user_function, integral_bounds = eval(config.name)()
  lb = integral_bounds[0][0] # lower bound
  ub = integral_bounds[0][1] # upper bound


  data_dim = config.data_dim
  init_num_data = config.init_num_data
  interval_std = config.interval_std
  interval = np.zeros((1,data_dim))
  std = np.zeros((1,data_dim))
  mean = np.zeros((1,data_dim))
  integral_bounds_scaled = integral_bounds.copy()
  for ii in range(data_dim):
    interval[0,ii] = integral_bounds[ii][1] - integral_bounds[ii][0]
    std[0,ii] = interval[0,ii]/interval_std
    mean[0,ii] = (integral_bounds[ii][1] + integral_bounds[ii][0])/2
    integral_bounds_scaled[ii] = ((integral_bounds[ii] - mean[0,ii])/std[0,ii]).tolist()

  lb_scaled = integral_bounds_scaled[0][0] # lower bound
  ub_scaled = integral_bounds_scaled[0][1] # upper bound
  lb = integral_bounds[0][0] # lower bound
  ub = integral_bounds[0][1] # upper bound


  results_list = [None] * config.repeated_runs
  npr = np.random.RandomState(config.seed)
  for kk in tqdm(range(config.repeated_runs)):

    integral_mean_list = np.zeros(config.bq_iter+1)
    integral_std_list = np.zeros(config.bq_iter+1)
    #initialize data points
    X_init = (npr.rand(init_num_data,data_dim)-0.5)*interval + mean
    Y_init = user_function.f(X_init)
    X_init_norm = (X_init - mean)/std


    Y_init_norm, mean_Y, std_Y = standardize(Y_init)
    
    X = X_init_norm
    Y = Y_init_norm
    X[np.abs(X)<1.0e-5] = 1.0e-5
    #normalized function
    function_norm= lambda x: (user_function.f(x*std+mean)- mean_Y)/std_Y
    
    if data_dim == 1:
      ground_truth = univariate_approximate_ground_truth_integral(function_norm, (lb_scaled, ub_scaled))[0]
    elif data_dim ==2:
      ground_truth = bivariate_approximate_ground_truth_integral(function_norm, integral_bounds_scaled)[0]




    #Set up BO_GP_Model
    emukit_gp_model = BO_GP_Model(X_init_norm, Y_init_norm, config, ai_model)
    emukit_gp_model.optimize()
    emukit_gp_model.set_kernel()
    emukit_quad_kern = QuadratureKernelCustom(emukit_gp_model, integral_bounds_scaled)
    emukit_model = BaseGaussianProcessCustomModel(kern=emukit_quad_kern, gp_model=emukit_gp_model)
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X, Y=Y)

    #set up Bayesian quadrature
    if config.plot:
      x_plot = np.linspace(integral_bounds_scaled[0][0], integral_bounds_scaled[0][1], 300)[:, None]
      y_plot = function_norm(x_plot)

      mu_plot, var_plot = emukit_method.predict(x_plot)

      plt.figure(figsize=FIGURE_SIZE)
      plt.plot(X_init_norm, Y_init_norm, "ro", markersize=10, label="Observations")
      plt.plot(x_plot, y_plot, "k", label="The Integrand")
      plt.plot(x_plot, mu_plot, "C0", label="Model")
      plt.fill_between(x_plot[:, 0],
                      mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                      mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)
      plt.fill_between(x_plot[:, 0],
                      mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                      mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)
      plt.fill_between(x_plot[:, 0],
                      mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                      mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
      plt.legend(loc=2, prop={'size': LEGEND_SIZE})
      plt.xlabel(r"$x$")
      plt.ylabel(r"$f(x)$")
      plt.grid(True)
      plt.xlim(lb_scaled, ub_scaled)
      plt.show()

    initial_integral_mean, initial_integral_variance = emukit_method.integrate()
    integral_mean_list[0] = initial_integral_mean
    integral_std_list[0] = np.sqrt(initial_integral_variance)

    if config.plot:

      x_plot_integral = np.linspace(initial_integral_mean-3*np.sqrt(initial_integral_variance), 
                                    initial_integral_mean+3*np.sqrt(initial_integral_variance), 200)
      y_plot_integral_initial = 1/np.sqrt(initial_integral_variance * 2 * np.pi) * \
      np.exp( - (x_plot_integral - initial_integral_mean)**2 / (2 * initial_integral_variance) )
      plt.figure(figsize=FIGURE_SIZE)
      plt.plot(x_plot_integral, y_plot_integral_initial, "k", label="initial integral density")
      plt.axvline(initial_integral_mean, color="red", label="initial integral estimate", \
                  linestyle="--")
      plt.axvline(ground_truth, color="blue", label="ground truth integral", \
                  linestyle="--")
      plt.legend(loc=2, prop={'size': LEGEND_SIZE})
      plt.xlabel(r"$F$")
      plt.ylabel(r"$p(F)$")
      plt.grid(True)
      plt.xlim(np.min(x_plot_integral), np.max(x_plot_integral))
      plt.show()

    
    print('The initial estimated integral is: ', round(initial_integral_mean, 4))
    print('with a credible interval: ', round(2*np.sqrt(initial_integral_variance), 4), '.')
    print('The ground truth rounded to 2 digits for comparison is: ', round(ground_truth, 4), '.')

    for ii in range(config.bq_iter):
      time_count = 0
      result = {}
      ivr_acquisition = IntegralVarianceReduction(emukit_method)
      space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
      num_steps = 200
      num_init_points = 5
      optimizer = LocalSearchAcquisitionOptimizer(space,num_steps,num_init_points)
      x_new,_ = optimizer.optimize(ivr_acquisition)
      y_new = function_norm(x_new)
      X = np.append(X, x_new, axis=0)
      Y = np.append(Y, y_new, axis=0)
      X[np.abs(X)<1.0e-5] = 1.0e-5
      emukit_method.set_data(X, Y)


      start_time = time.time()
      emukit_model.optimize()
      time_count = time_count + time.time() - start_time
      
      
      
      integral_mean, integral_variance = emukit_method.integrate()
      integral_mean_list[ii+1] = integral_mean
      integral_std_list[ii+1] = np.sqrt(integral_variance)

    if config.plot:

      mu_plot_final, var_plot_final = emukit_method.predict(x_plot)

      y_plot_integral = 1/np.sqrt(integral_variance * 2 * np.pi) * \
      np.exp( - (x_plot_integral - integral_mean)**2 / (2 * integral_variance) )

      plt.figure(figsize=FIGURE_SIZE)
      plt.plot(x_plot_integral, y_plot_integral_initial, "gray", label="initial integral density")
      plt.plot(x_plot_integral, y_plot_integral, "k", label="new integral density")
      plt.axvline(initial_integral_mean, color="gray", label="initial integral estimate", linestyle="--")
      plt.axvline(integral_mean, color="red", label="new integral estimate", linestyle="--")
      plt.axvline(ground_truth, color="blue", label="ground truth integral", \
                  linestyle="--")
      plt.legend(loc=2, prop={'size': LEGEND_SIZE})
      plt.xlabel(r"$F$")
      plt.ylabel(r"$p(F)$")
      plt.grid(True)
      plt.xlim(np.min(x_plot_integral), np.max(x_plot_integral))
      plt.show()

      plt.figure(figsize=FIGURE_SIZE)
      plt.plot(emukit_model.X, emukit_model.Y, "ro", markersize=10, label="Observations")
      plt.plot(x_plot, y_plot, "k", label="The Integrand")
      plt.plot(x_plot, mu_plot_final, "C0", label="Model")
      plt.fill_between(x_plot[:, 0],
                      mu_plot_final[:, 0] + np.sqrt(var_plot_final)[:, 0],
                      mu_plot_final[:, 0] - np.sqrt(var_plot_final)[:, 0], color="C0", alpha=0.6)
      plt.fill_between(x_plot[:, 0],
                      mu_plot_final[:, 0] + 2 * np.sqrt(var_plot_final)[:, 0],
                      mu_plot_final[:, 0] - 2 * np.sqrt(var_plot_final)[:, 0], color="C0", alpha=0.4)
      plt.fill_between(x_plot[:, 0],
                      mu_plot_final[:, 0] + 3 * np.sqrt(var_plot_final)[:, 0],
                      mu_plot_final[:, 0] - 3 * np.sqrt(var_plot_final)[:, 0], color="C0", alpha=0.2)
      plt.legend(loc=2, prop={'size': LEGEND_SIZE})
      plt.xlabel(r"$x$")
      plt.ylabel(r"$f(x)$")
      plt.grid(True)
      plt.xlim(lb_scaled, ub_scaled)
      plt.show()


    print('The estimated integral is: ', round(integral_mean, 4))
    print('with a credible interval: ', round(2*np.sqrt(integral_variance), 4), '.')
    print('The ground truth rounded to 2 digits for comparison is: ', round(ground_truth, 4), '.')



    integral_error_list = np.abs(integral_mean_list - ground_truth)
    result['integral_error_list'] = integral_error_list
    integral_error_list_scaledback = integral_error_list * std_Y.item()
    for jj in range(data_dim):
      integral_error_list_scaledback = integral_error_list_scaledback * std[0,jj]
    result['integral_error_list_scaledback'] = integral_error_list_scaledback
    result['integral_std_list'] = integral_std_list
    result['time_elapsed'] = time_count
    results_list[kk] = result
    print(time_count)


    if config.plot:
      plt.figure(figsize=(12, 8))
      plt.fill_between(np.arange(config.bq_iter+1)+1, integral_error_list-0.2*integral_std_list, integral_error_list+0.2*integral_std_list, color='red', alpha=0.15)
      plt.plot(np.arange(config.bq_iter+1)+1, integral_error_list, 'or-', lw=2, label='Estimated integral')
      plt.legend(loc=2, prop={'size': LEGEND_SIZE})
      plt.xlabel(r"iteration")
      plt.ylabel(r"$f(x)$")
      plt.grid(True)
      plt.show()
    
    #end of one run

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
  # Run the experiment
  results_list = bq_loop(config.bq, model)
  if config.bq.is_GPY:
    if config.bq.is_sparseGP:
      pickle.dump(results_list,
                  open(os.path.join(config.bq.save_dir, config.bq.name + '_sparseGP_init_p' + str(config.bq.init_num_data) + '_inducing_p'+ str(config.bq.num_inducing_pts) + '_results.p'), 'wb'))
    else:
      pickle.dump(results_list,
                  open(os.path.join(config.bq.save_dir, config.bq.name + '_fullGP_init_p' + str(config.bq.init_num_data) + '_results.p'), 'wb'))
  else:
    if config.bq.is_ai:
      pickle.dump(results_list,
                  open(os.path.join(config.bq.save_dir, config.bq.name + '_ai_init_p' + str(config.bq.init_num_data) + '_results.p'), 'wb'))
    else:
      pickle.dump(results_list,
                  open(os.path.join(config.bq.save_dir, config.bq.name + '_opt_init_p' + str(config.bq.init_num_data) + 'iter' +str(config.bq.opt_iter) + 'lr' + str(config.bq.opt_lr) + '_results.p'), 'wb'))
  sys.exit(0)

if __name__ == "__main__":
  main()