import numpy as np
from sklearn import linear_model
from tqdm import tqdm

def prepare_mnist_data(val_ratio=0.1, shuffle=True):
  from keras.datasets import mnist
  # load mnist data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # normalize the data into zero mean and unit variance
  x_train = x_train.astype(float) / 255.
  x_test = x_test.astype(float) / 255.
  mean, std = np.mean(x_train), np.std(x_train)
  x_train = (x_train - mean) / std
  x_train = np.reshape(x_train, [np.shape(x_train)[0], -1])
  x_test = (x_test - mean) / std
  x_test = np.reshape(x_test, [np.shape(x_test)[0], -1])

  if shuffle:
    # shuffle the data
    from sklearn.utils import shuffle
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

  # create the validation set
  num_val = int(len(x_train) * val_ratio)
  x_val, y_val = x_train[:num_val], y_train[:num_val]
  x_train, y_train = x_train[num_val:], y_train[num_val:]

  return x_train, y_train, x_val, y_val, x_test, y_test


def batch(iterable, n=1):
  l = len(iterable)
  for ndx in range(0, l, n):
    yield iterable[ndx:min(ndx + n, l)]


def logistic_regression_training(data, alpha=0.0001, l1_ratio=0.15, batch_size=10, num_epochs=100, eta0=1):
  x_train, y_train, x_val, y_val, x_test, y_test = data
  classes = np.unique(np.concatenate((y_train, y_val, y_test)))
  clf = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio, n_jobs=8, learning_rate='constant', eta0=eta0)
  for epoch_cnt in range(num_epochs):
    for data in tqdm(batch(list(zip(x_train, y_train)), batch_size)):
      x, y = zip(*data)
      clf = clf.partial_fit(x, y, classes=classes)
    print('epoch', epoch_cnt, 'batch_size =', batch_size, np.mean(clf.predict(x_train) == y_train), np.mean(clf.predict(x_val) == y_val), np.mean(clf.predict(x_test) == y_test))
  return np.mean(clf.predict(x_train) == y_train), np.mean(clf.predict(x_val) == y_val), np.mean(clf.predict(x_test) == y_test)


if __name__ == '__main__':
  data = prepare_mnist_data()
  print(logistic_regression_training(data))
