import numpy as np
from scipy import stats
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from ahgp.gp.gp_helper import standardize
from ahgp.inference.predict import predict

if __name__ == '__main__':

  filename = "../data/regression_datasets/housing.data"
  rand_seed = 10
  npr = np.random.RandomState(rand_seed)
  data = np.loadtxt(filename)
  data = shuffle(data, random_state=npr)
  x, y = data[:, :-1], data[:, -1]
  data_dim = x.shape[1]
  # normalize x and y, AHGP will take in normalized x and y only

  x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.1, random_state=npr)
  num_data = x_t.shape[0]
  # normalize x and y, AHGP will take in normalized x and y only
  x_t, x_v, _, _ = standardize(x_t, x_v)
  x_t = x_t*0.1
  x_v = x_v*0.1
  y_t, mean_y_train, std_y_train = standardize(y_t)


  model_config_filename = "../config/model.yaml"

  mu_test, var_test = predict(x_t,y_t,x_v,model_config_filename,use_gpu=False)

  mu_test = mu_test * std_y_train + mean_y_train
  var_test = var_test * std_y_train**2

  rmse = np.mean((mu_test - y_v) ** 2) ** .5
  log_likelihood = np.mean(np.log(stats.norm.pdf(
                              y_v,
                              loc=mu_test,
                              scale=var_test ** 0.5)))
  print(rmse)
  print(log_likelihood)