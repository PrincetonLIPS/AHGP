import numpy as np
import math
from emukit.core import ParameterSpace, ContinuousParameter

def stybtang2():
  parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 5), ContinuousParameter('x2', -5, 5)])
  return _stybtang, parameter_space
def stybtang5():
  parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 5), ContinuousParameter('x2', -5, 5), ContinuousParameter('x3', -5, 5), ContinuousParameter('x4', -5, 5), ContinuousParameter('x5', -5, 5)])
  return _stybtang, parameter_space
def stybtang10_scaled():
  """
  A single global mininimum `H(z) = -39.166166 * d` at `z = [-2.903534]^d`
  """
  parameter_space = ParameterSpace([ContinuousParameter('x1', -0.05, 0.05), ContinuousParameter('x2', -0.05, 0.05), ContinuousParameter('x3', -5, 5), ContinuousParameter('x4', -5, 5), ContinuousParameter('x5', -5, 5),
                                    ContinuousParameter('x6', -5, 5),  ContinuousParameter('x7', -5, 5),  ContinuousParameter('x8', -5, 5),  ContinuousParameter('x9', -0.05, 0.05),  ContinuousParameter('x10', -5, 5)])
  return _stybtang, parameter_space
def stybtang10():
  """
  A single global mininimum `H(z) = -39.166166 * d` at `z = [-2.903534]^d`
  """
  parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 5), ContinuousParameter('x2', -5, 5), ContinuousParameter('x3', -5, 5), ContinuousParameter('x4', -5, 5), ContinuousParameter('x5', -5, 5),
                                    ContinuousParameter('x6', -5, 5),  ContinuousParameter('x7', -5, 5),  ContinuousParameter('x8', -5, 5),  ContinuousParameter('x9', -5, 5),  ContinuousParameter('x10', -5, 5)])
  return _stybtang, parameter_space
def _stybtang(x):
  if len(x.shape) == 1:
    y = 0.5 * np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x)
    return y[:,None]
  else:
    y = 0.5 * np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x, axis=1)
    return y[:,None]

def michalewiczw2():
  parameter_space = ParameterSpace([ContinuousParameter('x1', 0, math.pi), ContinuousParameter('x2', 0, math.pi)])
  return _michalewicz, parameter_space

def michalewiczw10():
  parameter_space = ParameterSpace([ContinuousParameter('x1', 0, math.pi), ContinuousParameter('x2', 0, math.pi), ContinuousParameter('x3', 0, math.pi), ContinuousParameter('x4', 0, math.pi), ContinuousParameter('x5', 0, math.pi),
                                    ContinuousParameter('x6', 0, math.pi), ContinuousParameter('x7', 0, math.pi), ContinuousParameter('x8', 0, math.pi), ContinuousParameter('x9', 0, math.pi), ContinuousParameter('x10', 0, math.pi)])
  return _michalewicz, parameter_space

def _michalewicz(x):
  assert len(x.shape) == 2, 'x input must be 2 dimensional array'
  indx = np.arange(1.0, 1.0 + int(x.shape[1]))
  indx = np.expand_dims(indx, 0)
  y = -np.sum(np.sin(x) * np.sin(x * indx / np.pi) ** (2 * 10), axis=-1)
  return y[:,None]

def Hartmann6():
  parameter_space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1), ContinuousParameter('x3', 0, 1), ContinuousParameter('x4', 0, 1), ContinuousParameter('x5', 0, 1),
                                    ContinuousParameter('x6', 0, 1)])
  return _Hartmann6, parameter_space

def _Hartmann6(x):
  """
  x: N X D
  optimal value: -3.32237
  optimizer: [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
  """
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  A = np.array([
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ])

  P = np.array([
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ])
  x = np.expand_dims(x, axis=-2) # N X 1 X D
  A = np.expand_dims(A, axis=0) # 1 X 4 X D 
  P = np.expand_dims(P, axis=0) # 1 X 4 X D
  inner_sum = np.sum(A * (x - 0.0001*P)**2, axis =-1) # N X 4
  alpha = np.expand_dims(alpha, axis=0) # 1 X 4
  y = -np.sum(alpha * np.exp(-inner_sum), axis=-1) # N
  return y[:,None]

def Hartmann3():
  parameter_space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1), ContinuousParameter('x3', 0, 1)])
  return _Hartmann3, parameter_space

def _Hartmann3(x):
  """
  x: N X D
  optimal value: -3.86278
  optimizer: [(0.114614, 0.555649, 0.852547)]
  """
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])

  P = np.array([
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ])
  x = np.expand_dims(x, axis=-2) # N X 1 X D
  A = np.expand_dims(A, axis=0) # 1 X 4 X D 
  P = np.expand_dims(P, axis=0) # 1 X 4 X D
  inner_sum = np.sum(A * (x - 0.0001*P)**2, axis =-1) # N X 4
  alpha = np.expand_dims(alpha, axis=0) # 1 X 4
  y = -np.sum(alpha * np.exp(-inner_sum), axis=-1) # N
  return y[:,None]



  
def Levy():
  parameter_space = ParameterSpace([ContinuousParameter('x1', -10, 10), ContinuousParameter('x2', -10, 10),
                                    ContinuousParameter('x3', -10, 10), ContinuousParameter('x4', -10, 10),
                                    ContinuousParameter('x5', -10, 10), ContinuousParameter('x6', -10, 10),
                                    ContinuousParameter('x7', -10, 10), ContinuousParameter('x8', -10, 10),
                                    ContinuousParameter('x9', -10, 10), ContinuousParameter('x10', -10, 10)])
  return _Levy, parameter_space

def _Levy(x):
  """
  Global Minimum at `z_1 = (1, 1, ..., 1)` with `f(z_1) = 0`.
  """

  w = 1 + (x - 1) / 4
  w_mid = w[:, :-1]
  f = np.sum(np.multiply((w_mid - 1)**2, 1 + 10 * np.sin(np.pi * w_mid + 1)**2), axis = 1)

  f += np.sin(np.pi * w[:, 0])**2 + (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)

  return f[:,None]

def SixHumpCamel():
  parameter_space = ParameterSpace([ContinuousParameter('x1', -3, 3), ContinuousParameter('x2', -2, 2)])
  return _SixHumpCamel, parameter_space

def _SixHumpCamel(x):
  """
  Global Minimum at `z_1 = (0.0898, -0.7126)` and `z2 = (-0.0898, -0.7126)` with `f(z_1) = f(x_2) = -1.0316`.
  """
  x1, x2 = x[:, 0], x[:, 1]
  f = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (4 * x2 ** 2 - 4) * x2 ** 2
  return f[:,None]

def Ackley():
  parameter_space = ParameterSpace([ContinuousParameter('x1', -32.768, 32.768), ContinuousParameter('x2', -32.768, 32.768),
                                    ContinuousParameter('x3', -32.768, 32.768), ContinuousParameter('x4', -32.768, 32.768),
                                    ContinuousParameter('x5', -32.768, 32.768), ContinuousParameter('x6', -32.768, 32.768),
                                    ContinuousParameter('x7', -32.768, 32.768), ContinuousParameter('x8', -32.768, 32.768),
                                    ContinuousParameter('x9', -32.768, 32.768), ContinuousParameter('x10', -32.768, 32.768)])
  return _Ackley, parameter_space

def _Ackley(x):
  """
  f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
  """

  a, b, c = 20, 0.2, 2*np.pi

  part1 = -a * np.exp(-b * np.sqrt(np.mean(x**2, axis = 1)))
  part2 = -np.exp(np.mean(np.cos(c * x), axis = 1))
  f = part1 + part2 + a + np.exp(1)
  return f[:,None]

def Griewank():
  parameter_space = ParameterSpace([ContinuousParameter('x1', -600, 600), ContinuousParameter('x2', -600, 600),
                                    ContinuousParameter('x3', -600, 600), ContinuousParameter('x4', -600, 600),
                                    ContinuousParameter('x5', -600, 600), ContinuousParameter('x6', -600, 600),
                                    ContinuousParameter('x7', -600, 600), ContinuousParameter('x8', -600, 600),
                                    ContinuousParameter('x9', -600, 600), ContinuousParameter('x10', -600, 600)])
  return _Griewank, parameter_space

def _Griewank(x):

  """
  f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
  """

  part1 = np.sum(x ** 2 / 4000.0, axis = 1)
  d = x.shape[1]
  i = np.array(range(1, d + 1))

  part2 = -np.prod(
            np.cos(x / np.sqrt(i)), axis = 1)
  f = part1 + part2 + 1.0
  return f[:,None]