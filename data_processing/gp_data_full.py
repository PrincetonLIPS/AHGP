import os
import glob
import torch
import pickle
import numpy as np
from sklearn.utils import shuffle

__all__ = ['GPDataFull']


class GPDataFull(object):

  def __init__(self, config, split='train'):
    assert split == 'train' or split == 'dev' or split == 'test', "no such split"
    self.split = split
    self.config = config
    self.seed = config.seed
    self.is_val = config.is_val
    self.is_debug = config.is_debug
    self.num_pairs = config.train.num_pairs
    self.npr = np.random.RandomState(self.seed)
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name

    self.train_data_files = glob.glob(
        os.path.join(self.data_path, '*_train_*.p'))
    self.dev_data_files = glob.glob(
        os.path.join(self.data_path, '*_dev_*.p'))
    self.test_data_files = glob.glob(
        os.path.join(self.data_path, '*_test_*.p'))

    self.num_train = len(self.train_data_files)
    self.num_dev = len(self.dev_data_files)
    self.num_test = len(self.test_data_files)
    self.num_gps = self.num_train + self.num_dev + self.num_test

  def __getitem__(self, index):
    if self.split == 'train':
      return pickle.load(open(self.train_data_files[index], 'rb'), encoding='latin1')
    elif self.split == 'dev':
      return pickle.load(open(self.dev_data_files[index], 'rb'), encoding='latin1')
    else:
      return pickle.load(open(self.test_data_files[index], 'rb'), encoding='latin1')

  def __len__(self):
    if self.split == 'train':
      return self.num_train
    elif self.split == 'dev':
      return self.num_dev
    else:
      return self.num_test

  def collate_fn(self, batch):
    assert isinstance(batch, list)

    data = {}
    batch_size = len(batch)
    node_size = torch.zeros((batch_size)).int()
    dim_size = torch.zeros((batch_size)).int()
    node_size_tr = torch.zeros((batch_size)).int()
    node_size_val = torch.zeros((batch_size)).int()

    for ii, bb in enumerate(batch):
      node_size[ii] = bb['X'].shape[0]
      dim_size[ii] = bb['X'].shape[1]
      bb['X_shuffled'] = bb['X']
      # training and validation are set to be the same
      # todo: delete validation, as it is not used
      node_size_tr[ii] = node_size[ii]
      node_size_val[ii] = node_size[ii]

    max_node_size = max(node_size)  # value -> N
    max_dim_size = max(dim_size) # value -> D
    data['dim_size'] = dim_size
    data['max_node_size'] = max_node_size
    data['max_dim_size'] = max_dim_size
    pad_node_size = [max_node_size - nn for nn in node_size]
    pad_dim_size = [max_dim_size - nn for nn in dim_size]

    max_node_size_tr = max(node_size_tr)  # value -> N
    data['max_node_size_tr'] = max_node_size_tr
    pad_node_size_tr = [max_node_size_tr - nn for nn in node_size_tr]
    
    max_node_size_val = max(node_size_val)  # value -> N
    data['max_node_size_val'] = max_node_size_val
    pad_node_size_val = [max_node_size_val - nn for nn in node_size_val]

    # X_data: B X N X D
    data['X_data_full'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  bb['X'], ((0, pad_node_size[ii]), (0, pad_dim_size[ii])),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

    data['X_data_tr'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  bb['X_shuffled'][:node_size_tr[ii],:], ((0, pad_node_size_tr[ii]), (0, pad_dim_size[ii])),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

    data['X_data_val'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  bb['X_shuffled'][-node_size_val[ii]:,:], ((0, pad_node_size_val[ii]), (0, pad_dim_size[ii])),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

    data['X_data_test'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  bb['X_2'], ((0, pad_node_size[ii]), (0, pad_dim_size[ii])),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()
    # F: B X N
    data['F_full'] = torch.stack([
      torch.from_numpy(
          np.pad(
              bb['f'], (0, pad_node_size[ii]),
              'constant',
              constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

    data['F_tr'] = torch.stack([
      torch.from_numpy(
          np.pad(
              bb['f'][:node_size_tr[ii]], (0, pad_node_size_tr[ii]),
              'constant',
              constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

    data['F_val'] = torch.stack([
      torch.from_numpy(
          np.pad(
              bb['f'][-node_size_val[ii]:], (0, pad_node_size_val[ii]),
              'constant',
              constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

    data['F_test'] = torch.stack([
      torch.from_numpy(
          np.pad(
              bb['f_2'], (0, pad_node_size[ii]),
              'constant',
              constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()
    # node_mask: B X N
    data['node_mask_tr'] = torch.stack([
      torch.from_numpy(
          np.pad(
              np.ones(node_size_tr[ii]), (0, pad_node_size_tr[ii]),
              'constant',
              constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()
    # dim_mask: B X D
    data['dim_mask'] = torch.stack([
      torch.from_numpy(
          np.pad(
              np.ones(dim_size[ii]), (0, pad_dim_size[ii]),
              'constant',
              constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()
    #diagonal_mask: B X N (0,0,0,...,0,1,...)
    data['diagonal_mask_val'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  np.ones(pad_node_size_val[ii]), (node_size_val[ii],0),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()
    #kernel_mask: B X N X N
    data['kernel_mask_val'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  np.ones((node_size_val[ii],node_size_val[ii])), ((0, pad_node_size_val[ii]),(0, pad_node_size_val[ii])),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()
    #diagonal_mask: B X N (0,0,0,...,0,1,...)
    data['diagonal_mask_test'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  np.ones(pad_node_size[ii]), (node_size[ii],0),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()
    #kernel_mask: B X N X N
    data['kernel_mask_test'] = torch.stack([
          torch.from_numpy(
              np.pad(
                  np.ones((node_size[ii],node_size[ii])), ((0, pad_node_size[ii]),(0, pad_node_size[ii])),
                  'constant',
                  constant_values=0.0)) for ii, bb in enumerate(batch)
      ]).float()

    data['nmll'] = torch.cat(
          [torch.from_numpy(bb['nmll']) for bb in batch], dim=0).float()
    if self.is_val:
      data['nmll_opt_sm'] = torch.cat(
          [torch.from_numpy(bb['nmll_opt_sm']) for bb in batch], dim=0).float()
      data['nmll_opt_sm_test'] = torch.cat(
          [torch.from_numpy(bb['nmll_opt_sm_test']) for bb in batch], dim=0).float()
    if self.is_debug:
      data['sm_params'] = batch[0]['sm_params']
    data['nmll_test'] = torch.cat(
          [torch.from_numpy(bb['nmll_test']) for bb in batch], dim=0).float()
    data['N_val'] = node_size_val
    data['N_test'] = node_size
  
    return data