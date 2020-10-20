import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

def prepare_mnist_data(batch_size):
  train_val_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
  train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [50000, 10000])
  test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, val_loader, test_loader


class LogisticRegressionModel(nn.Module):
  def __init__(self, input_dim, output_dim, alpha_l1, alpha_l2, learning_rate):
    super(LogisticRegressionModel, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.learning_rate = learning_rate
    self.linear = nn.Linear(self.input_dim, self.output_dim)
    self.alpha_l1 = alpha_l1
    self.alpha_l2 = alpha_l2
    self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x):
    out = self.linear(x)
    return out

  def one_step(self, images, labels):
    images = images.view(-1, self.input_dim).requires_grad_()
    self.optimizer.zero_grad()
    outputs = self.forward(images)
    loss = self.criterion(outputs, labels) + self.alpha_l1 * torch.sum(torch.abs(self.linear.weight)) + self.alpha_l2 * torch.norm(self.linear.weight) ** 2
    loss.backward()
    self.optimizer.step()

  def calc_accuracy(self, dataloader):
    correct = 0
    total = 0
    for images, labels in dataloader:
      images = images.view(-1, self.input_dim).requires_grad_()
      outputs = self.forward(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum()
    accuracy = 100. * correct.item() / total
    return accuracy


def logistic_regression_training(parameters, time_limit=10, check_per_iterations=50):
  log_alpha_l1, log_alpha_l2, log_batch_size, log_learning_rate = parameters.flatten()
  alpha_l1, alpha_l2, batch_size, learning_rate = np.exp(log_alpha_l1), np.exp(log_alpha_l2), np.exp(log_batch_size), np.exp(log_learning_rate)
  batch_size = int(batch_size)
  train_loader, val_loader, test_loader = prepare_mnist_data(batch_size=batch_size)
  model = LogisticRegressionModel(input_dim=28*28, output_dim=10, alpha_l1=alpha_l1, alpha_l2=alpha_l2, learning_rate=learning_rate)
  starting_time = time.time()
  cnt = 0
  while True:
    for i, (images, labels) in enumerate(train_loader):
      model.one_step(images, labels)
      cnt += 1
      if cnt % check_per_iterations == 0:
        if time.time() - starting_time > time_limit:
          break
    if time.time() - starting_time > time_limit:
      break

  training_accuracy = model.calc_accuracy(train_loader)
  val_accuracy = model.calc_accuracy(val_loader)
  test_accuracy = model.calc_accuracy(test_loader)
  return -training_accuracy, -val_accuracy, -test_accuracy


if __name__ == '__main__':
  training_accuracy, val_accuracy, test_accuracy = logistic_regression_training([-8., -8., 3., -2.])
  print(training_accuracy, val_accuracy, test_accuracy)
