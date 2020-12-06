import numpy as np
from easydict import EasyDict as edict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from ahgp.inference.hyperparam import hyperparam
from ahgp.gp.gp_helper import cal_kern_spec_mix_nomu_sep, GP_noise, standardize

if __name__ == '__main__':

  # load data in to x (features) and y (labels)
  filename = "../data/regression_datasets/housing.data"
  rand_seed = 100
  npr = np.random.RandomState(rand_seed)
  data = np.loadtxt(filename)
  data = shuffle(data, random_state=npr)
  x, y = data[:, :-1], data[:, -1]
  data_dim = x.shape[1]

  # random split data to training and testing
  x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.1, random_state=npr)
  # normalize x and y, AHGP will take in normalized x and y only
  x_t, x_v, _, _ = standardize(x_t, x_v)
  x_t = x_t*0.1
  x_v = x_v*0.1
  y_t, mean_y_train, std_y_train = standardize(y_t)

  # yaml config file of the AHGP model
  model_config_filename = "../config/model.yaml"
  # hyper_params are the hyperparameters of the spectral mixture product kernel
  hyper_params = hyperparam(x_t,y_t,x_v,model_config_filename,use_gpu=False)
  print(hyper_params)  
  
