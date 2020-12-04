import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from scipy import stats
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from model.nn import *
from data_processing import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper, standardize
from model.gp.gp_helper import *
from utils.nmll_opt import nmll_opt_gp, nmll_opt
from utils.optimization import get_constant_schedule_with_warmup
from utils.train_helper import get_lr
import matplotlib.pyplot as plt
import time

logger = get_logger('exp_logger')
__all__ = ['GpSM2AttRunner']


class GpSM2AttRunner(object):

  def __init__(self, config):
    self.epsilon = config.dataset.epsilon
    self.config = config
    self.lambda_KL = config.model.lambda_KL
    self.dataset_conf = config.dataset
    self.data_path = self.dataset_conf.data_path
    self.model_conf = config.model
    self.num_mix = self.model_conf.num_mix
    self.is_no_mu = self.model_conf.is_no_mu
    self.is_param_dim_diff = self.model_conf.is_param_dim_diff
    self.is_debug = self.config.is_debug
    self.is_val = self.config.is_val
    self.train_conf = config.train
    self.warmup_setps = self.train_conf.warmup_steps
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.device = torch.device('cuda' if config.use_gpu else 'cpu')
    self.gpus = config.gpus
    self.subsample_size = config.subsample_size
    self.writer = SummaryWriter(config.save_dir)

  def cal_dataset_loss(self, model, data_loader):
    result_dic = {}
    loss = []
    nmll_loss = []
    nmll_loss_test = []
    nmll_loss_orig = []
    nmll_loss_orig_test = []
    win_pct_avg = []
    win_pct_avg_test = []
    nmll_opt_sm_list = []
    nmll_opt_sm_test_list = []
    win_pct_ai_list = []
    win_pct_ai_test_list = []
    for data in tqdm(data_loader):
      if self.use_gpu:
        data['max_node_size'],data['X_data_tr'],data['X_data_val'],data['X_data_test'],data['F_tr'],data['F_val'],data['F_test'],data['N_val'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_test'],data['kernel_mask_test'],data['diagonal_mask_test'],data['node_mask_tr'],data['dim_mask'], data['nmll'], data['nmll_test'] = data_to_gpu(
              data['max_node_size'],data['X_data_tr'],data['X_data_val'],data['X_data_test'],data['F_tr'],data['F_val'],data['F_test'],data['N_val'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_test'],data['kernel_mask_test'],data['diagonal_mask_test'],data['node_mask_tr'],data['dim_mask'], data['nmll'], data['nmll_test'])
      if self.model_conf.name == 'GpSMDoubleAtt':
        mu, var, weights, nmll, nmll_test = model(data['X_data_tr'],data['X_data_val'],data['F_tr'],data['F_val'],data['node_mask_tr'],data['dim_mask'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_val'],device = self.device,eval_mode = True,X_data_test = data['X_data_test'],F_data_test = data['F_test'],kernel_mask_test=data['kernel_mask_test'],diagonal_mask_test=data['diagonal_mask_test'],N_data_test=data['N_test'])
      elif self.model_conf.name == 'GpSMDoubleAttNoMu':
        var, weights, nmll, nmll_test = model(data['X_data_tr'],data['X_data_val'],data['F_tr'],data['F_val'],data['node_mask_tr'],data['dim_mask'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_val'],device = self.device,eval_mode = True,X_data_test = data['X_data_test'],F_data_test = data['F_test'],kernel_mask_test=data['kernel_mask_test'],diagonal_mask_test=data['diagonal_mask_test'],N_data_test=data['N_test'])
      else:
        raise ValueError("No model of given name!")
      nmll_orig_test = data['nmll_test']
      nmll_orig = data['nmll']
      win_pct = torch.sum(nmll<nmll_orig+0.01).float()/nmll.shape[0]
      win_pct_test = torch.sum(nmll_test<nmll_orig_test+0.01).float()/nmll_test.shape[0]

      if self.is_val:
        nmll_opt_sm = data['nmll_opt_sm'].to(self.device)
        nmll_opt_sm_test = data['nmll_opt_sm_test'].to(self.device)
        win_pct_ai = torch.sum(nmll<nmll_opt_sm+0.01).float()/nmll.shape[0]
        win_pct_ai_test = torch.sum(nmll_test<nmll_opt_sm_test+0.01).float()/nmll.shape[0]
        nmll_opt_sm_mean = torch.mean(nmll_opt_sm)
        nmll_opt_sm_test_mean = torch.mean(nmll_opt_sm_test)
        nmll_opt_sm_list += [nmll_opt_sm_mean.cpu().numpy()]
        nmll_opt_sm_test_list += [nmll_opt_sm_test_mean.cpu().numpy()]
        win_pct_ai_list += [win_pct_ai.cpu().numpy()]
        win_pct_ai_test_list += [win_pct_ai_test.cpu().numpy()]

      #calculate loss
      current_nmll_mean = torch.mean(nmll)
      current_nmll_mean_test = torch.mean(nmll_test)
      curr_loss = current_nmll_mean
      loss += [curr_loss.cpu().numpy()]
      nmll_loss += [current_nmll_mean.cpu().numpy()]
      nmll_loss_test += [current_nmll_mean_test.cpu().numpy()]
      nmll_loss_orig += [torch.mean(nmll_orig).cpu().numpy()]
      nmll_loss_orig_test += [torch.mean(nmll_orig_test).cpu().numpy()]
      win_pct_avg += [win_pct.cpu().numpy()]
      win_pct_avg_test += [win_pct_test.cpu().numpy()]

    result_dic['loss'] = float(np.mean(loss))
    result_dic['nmll'] = float(np.mean(nmll_loss))
    result_dic['nmll_test'] = float(np.mean(nmll_loss_test))
    result_dic['nmll_orig'] = float(np.mean(nmll_loss_orig))
    result_dic['nmll_test_orig'] = float(np.mean(nmll_loss_orig_test))
    result_dic['win_pct'] = float(np.mean(win_pct_avg))
    result_dic['win_pct_test'] = float(np.mean(win_pct_avg_test))
    if self.is_val:
      result_dic['nmll_opt_sm'] = float(np.mean(nmll_opt_sm_list))
      result_dic['nmll_opt_sm_test'] = float(np.mean(nmll_opt_sm_test_list))
      result_dic['win_pct_ai'] = float(np.mean(win_pct_ai_list))
      result_dic['win_pct_ai_test'] = float(np.mean(win_pct_ai_test_list))
    return result_dic


  def cal_sample_result(self, model, data_loader):
    data_loader_iter = iter(data_loader)
    for ii in range(self.config.sample_size):
      results_sample = {}
      try:
        data = next(data_loader_iter)
      except StopIteration:
        data_loader_iter = iter(data_loader)
        data = next(data_loader_iter) 
        #StopIteration is thrown if dataset ends
        #reinitialize data loader 

      if self.use_gpu:
        data['max_node_size'],data['X_data_tr'],data['X_data_val'],data['X_data_test'],data['F_tr'],data['F_val'],data['F_test'],data['N_val'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_test'],data['kernel_mask_test'],data['diagonal_mask_test'],data['node_mask_tr'],data['dim_mask'], data['nmll'] = data_to_gpu(
              data['max_node_size'],data['X_data_tr'],data['X_data_val'],data['X_data_test'],data['F_tr'],data['F_val'],data['F_test'],data['N_val'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_test'],data['kernel_mask_test'],data['diagonal_mask_test'],data['node_mask_tr'],data['dim_mask'], data['nmll'])

      if self.model_conf.name == 'GpSMDoubleAtt':
        mu, var, weights, nmll_sample, nmll_test = model(data['X_data_tr'],data['X_data_val'],data['F_tr'],data['F_val'],data['node_mask_tr'],data['dim_mask'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_val'],device = self.device,eval_mode = True,X_data_test = data['X_data_test'],F_data_test = data['F_test'],kernel_mask_test=data['kernel_mask_test'],diagonal_mask_test=data['diagonal_mask_test'],N_data_test=data['N_test'])
      elif self.model_conf.name == 'GpSMDoubleAttNoMu':
        var, weights, nmll_sample, nmll_test = model(data['X_data_tr'],data['X_data_val'],data['F_tr'],data['F_val'],data['node_mask_tr'],data['dim_mask'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_val'],device = self.device,eval_mode = True,X_data_test = data['X_data_test'],F_data_test = data['F_test'],kernel_mask_test=data['kernel_mask_test'],diagonal_mask_test=data['diagonal_mask_test'],N_data_test=data['N_test'])
      else:
        raise ValueError("No model of given name!")

      if self.is_val:
        nmll_sample_opt = data['nmll_opt_sm'].to(self.device)
        nmll_sample_compare = torch.cat((nmll_sample.unsqueeze(-1),nmll_sample_opt.unsqueeze(-1)),1) #calulated first, original second
        win_count = torch.sum(nmll_sample < nmll_sample_opt+0.01)
      else:
        nmll_sample_orig = data['nmll']
        nmll_sample_compare = torch.cat((nmll_sample.unsqueeze(-1),nmll_sample_orig.unsqueeze(-1)),1) #calulated first, original second
        win_count = torch.sum(nmll_sample < nmll_sample_orig+0.01)

      #calculate loss
      nmll_loss_sample= torch.mean(nmll_sample)
      results_sample['nmll_sample_compare'] = nmll_sample_compare.cpu().numpy()
      results_sample['win_pct'] = float(win_count.cpu().numpy()/nmll_sample.shape[0])
      results_sample['nmll_loss_sample'] = float(nmll_loss_sample.cpu().numpy())
      results_sample['var'] = var.cpu().numpy()
      results_sample['weights'] = weights.cpu().numpy()
      if not self.is_no_mu:
        results_sample['mu'] = mu.cpu().numpy()

      if self.is_debug:
        length = 1/(math.sqrt(2)*math.pi*torch.sqrt(var))
        length_avg = torch.sum(length * weights,-2).squeeze(0)
        sm_params_opt = data['sm_params']
        length_opt = 1/(math.sqrt(2)*math.pi*torch.sqrt(sm_params_opt.var))
        length_avg_opt = torch.sum(length_opt * sm_params_opt.weights,-2).squeeze(0)
        print(length_avg_opt)
        print(nmll_sample_opt)
        print(length_avg)
        print(nmll_sample)
        pdb.set_trace()


    return results_sample

  def train(self):
    # create data loader
    train_dataset = eval(self.dataset_conf.loader_name)(
        self.config, split='train')
    dev_dataset = eval(self.dataset_conf.loader_name)(self.config, split='dev')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)
    subset_indices = range(self.subsample_size)
    train_loader_sub = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=False,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False,
        sampler=SubsetRandomSampler(subset_indices))
    dev_loader_sub = torch.utils.data.DataLoader(
          dev_dataset,
          batch_size=self.train_conf.batch_size,
          shuffle=False,
          num_workers=self.train_conf.num_workers,
          collate_fn=dev_dataset.collate_fn,
          drop_last=False,
          sampler=SubsetRandomSampler(subset_indices))

    # create models
    model = eval(self.model_conf.name)(self.model_conf)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(
          params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_setps)

    # reset gradient
    optimizer.zero_grad()

    # resume training or use prxetrained model
    if self.train_conf.is_resume:
      if self.train_conf.pretrain:
        model_snapshot = torch.load(self.train_conf.resume_model,map_location=self.device)
        model.load_state_dict(model_snapshot["model"],strict=False)
        model.to(self.device)
      else:
        model_snapshot = torch.load(self.train_conf.resume_model,map_location=self.device)
        model.load_state_dict(model_snapshot["model"],strict=True)
        model.to(self.device)

    # Training Loop
    num_train = len(train_dataset)
    iter_count = 0
    best_val_loss = np.inf
    best_val_loss_test = np.inf
    best_win_pct_val = 0
    best_win_pct_val_test = 0

    results = defaultdict(list)
    for epoch in range(self.train_conf.max_epoch):

      # --------------------------------validation---------------------------------------------
      if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:

        #calculate validation loss
        model.eval()
        with torch.no_grad():
          result_dataset_val = self.cal_dataset_loss(model,dev_loader_sub)

        if self.is_val:
          logger.info("-----------------Avg. Validation Loss = {:.4f}, "
          "NMLL = {:.4f}, NMLL_opt = {:.4f}, Win_pct = {:.2f}%, "
          "NMLL_test = {:.4f}, NMLL_test_opt = {:.4f}, "
          "Win_pct_test = {:.2f}%--------------------".format(
            result_dataset_val['loss'], 
            result_dataset_val['nmll'], result_dataset_val['nmll_opt_sm'],
            result_dataset_val['win_pct_ai']*100, 
            result_dataset_val['nmll_test'], result_dataset_val['nmll_opt_sm_test'],
            result_dataset_val['win_pct_ai_test']*100))
          self.writer.add_scalar('nmll_opt_val', result_dataset_val['nmll_opt_sm'], iter_count)
          self.writer.add_scalar('nmll_opt_test_val', result_dataset_val['nmll_opt_sm_test'], iter_count)
          self.writer.add_scalar('win_pct_ai_val', result_dataset_val['win_pct_ai'], iter_count)
          self.writer.add_scalar('win_pct_ai_test_val', result_dataset_val['win_pct_ai_test'], iter_count)
        else:
          logger.info("-----------------Avg. Validation Loss = {:.4f}, "
            "NMLL = {:.4f}, NMLL_orig = {:.4f}, Win_pct = {:.2f}%, "
            "NMLL_test = {:.4f}, NMLL_test_orig = {:.4f}, "
            "Win_pct_test = {:.2f}%--------------------".format(
              result_dataset_val['loss'], 
              result_dataset_val['nmll'], result_dataset_val['nmll_orig'],
              result_dataset_val['win_pct']*100, 
              result_dataset_val['nmll_test'], result_dataset_val['nmll_test_orig'],
              result_dataset_val['win_pct_test']*100))

        self.writer.add_scalar('val_loss', result_dataset_val['loss'], iter_count)
        self.writer.add_scalar('nmll_loss_val', result_dataset_val['nmll'], iter_count)
        self.writer.add_scalar('nmll_loss_orig_val', result_dataset_val['nmll_orig'], iter_count)
        self.writer.add_scalar('nmll_loss_test_val', result_dataset_val['nmll_test'], iter_count)
        self.writer.add_scalar('nmll_loss_test_orig_val', result_dataset_val['nmll_test_orig'], iter_count)
        self.writer.add_scalar('win_pct_val', result_dataset_val['win_pct'], iter_count)
        self.writer.add_scalar('win_pct_val_test', result_dataset_val['win_pct_test'], iter_count)
        results['val_loss'] += [result_dataset_val['loss']]
        results['nmll_loss_val'] += [result_dataset_val['nmll']]
        results['nmll_loss_orig_val'] += [result_dataset_val['nmll_orig']]
        results['nmll_loss_test_val'] += [result_dataset_val['nmll_test']]
        results['nmll_loss_test_orig_val'] += [result_dataset_val['nmll_test_orig']]
        results['win_pct_val'] += [result_dataset_val['win_pct']]
        results['win_pct_val_test'] += [result_dataset_val['win_pct_test']]

        # save best model
        if result_dataset_val['loss'] < best_val_loss:
          best_val_loss = result_dataset_val['loss']
          best_val_loss_test = result_dataset_val['nmll_test']
          if self.is_val:
            best_win_pct_val = result_dataset_val['win_pct_ai']
            best_win_pct_val_test = result_dataset_val['win_pct_ai_test']
          else:
            best_win_pct_val = result_dataset_val['win_pct']
            best_win_pct_val_test = result_dataset_val['win_pct_test']
          snapshot(
              model.module if self.use_gpu else model,
              optimizer,
              self.config,
              epoch + 1,
              tag='best')

        logger.info("Current Best Validation Loss = {:.4f}".format(best_val_loss))

        # check early stop
        if early_stop.tick([result_dataset_val['loss']]):
          snapshot(
              model.module if self.use_gpu else model,
              optimizer,
              self.config,
              epoch + 1,
              tag='last')
          self.writer.close()
          break

      # --------------------------------------training-----------------------------------
      model.train()
      for data in train_loader:
        optimizer.zero_grad()

        if self.use_gpu:
          data['max_node_size'],data['X_data_tr'],data['X_data_val'],data['X_data_test'],data['F_tr'],data['F_val'],data['F_test'],data['N_val'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_test'],data['kernel_mask_test'],data['diagonal_mask_test'],data['node_mask_tr'],data['dim_mask'], data['nmll'], data['dim_size'] = data_to_gpu(
                data['max_node_size'],data['X_data_tr'],data['X_data_val'],data['X_data_test'],data['F_tr'],data['F_val'],data['F_test'],data['N_val'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_test'],data['kernel_mask_test'],data['diagonal_mask_test'],data['node_mask_tr'],data['dim_mask'], data['nmll'], data['dim_size'])

        if self.model_conf.name == 'GpSMDoubleAtt':
          mu, var, weights, nmll, nmll_test = model(data['X_data_tr'],data['X_data_val'],data['F_tr'],data['F_val'],data['node_mask_tr'],data['dim_mask'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_val'],device = self.device,eval_mode = True,X_data_test = data['X_data_test'],F_data_test = data['F_test'],kernel_mask_test=data['kernel_mask_test'],diagonal_mask_test=data['diagonal_mask_test'],N_data_test=data['N_test'])
        elif self.model_conf.name == 'GpSMDoubleAttNoMu':
          var, weights, nmll, nmll_test = model(data['X_data_tr'],data['X_data_val'],data['F_tr'],data['F_val'],data['node_mask_tr'],data['dim_mask'],data['kernel_mask_val'],data['diagonal_mask_val'],data['N_val'],device = self.device,eval_mode = True,X_data_test = data['X_data_test'],F_data_test = data['F_test'],kernel_mask_test=data['kernel_mask_test'],diagonal_mask_test=data['diagonal_mask_test'],N_data_test=data['N_test'])
        else:
          raise ValueError("No model of given name!")
        # print("Outside: input size", data['X_data'].shape, "output_size", nmll.shape)

        nmll_orig = data['nmll']
        win_pct_train = torch.sum(nmll<nmll_orig+0.01).float()/nmll.shape[0]

        data_dim_vec = data['X_data_tr'].shape[-1]
        nmll_loss_train = torch.mean(nmll)

        train_loss = nmll_loss_train

        # calculate gradient
        train_loss.backward()

        nmll_loss_orig = torch.mean(nmll_orig)

        # calculate gradient norm
        grad_norm = 0
        for p in model.parameters():
          if p.requires_grad:
            param_norm = p.grad.data.norm()
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** (1./2)
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        train_loss = float(train_loss.data.cpu().numpy())
        nmll_loss_train = float(nmll_loss_train.data.cpu().numpy())
        nmll_loss_train_orig = float(nmll_loss_orig.data.cpu().numpy())
        win_pct_train = float(win_pct_train.data.cpu().numpy())


        self.writer.add_scalar('train_loss', train_loss, iter_count)
        self.writer.add_scalar('nmll_loss_train', nmll_loss_train, iter_count)
        self.writer.add_scalar('nmll_loss_train_orig', nmll_loss_train_orig, iter_count)
        self.writer.add_scalar('win_pct_train', win_pct_train, iter_count)
        self.writer.add_scalar('grad_norm', grad_norm, iter_count)

        results['nmll_loss_train'] += [nmll_loss_train]
        results['nmll_loss_train_orig'] += [nmll_loss_train_orig]
        results['train_loss'] += [train_loss]
        results['win_pct_train'] += [win_pct_train]
        results['train_step'] += [iter_count]
        results['grad_norm'] += [grad_norm]

        # display loss
        if (iter_count + 1) % self.train_conf.display_iter == 0:
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {:.4f}, NMLL = {:.4f}, NMLL_orig = {:.4f}, Win_pct = {:.2f}%, Grad_norm = {:.4f}, LR = {:.2e}".format(
              epoch + 1, iter_count + 1, train_loss, nmll_loss_train, nmll_loss_train_orig, win_pct_train*100, grad_norm, get_lr(optimizer)))

        iter_count += 1

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module
                 if self.use_gpu else model, optimizer, self.config, epoch + 1)

      lr_scheduler.step()


    #look at predictions, for debug purpose
    model.eval()
    with torch.no_grad():
      results_sample_tr = self.cal_sample_result(model,train_loader_sub)
      results_sample_dev = self.cal_sample_result(model,dev_loader_sub)
      result_dataset_tr = self.cal_dataset_loss(model,train_loader_sub)
      result_dataset_dev = self.cal_dataset_loss(model,dev_loader_sub)

    
    train_loss = result_dataset_tr['loss']
    results['best_val_loss'] = best_val_loss
    results['win_count_tr'] = results_sample_tr['win_pct']
    results['win_count_dev'] = results_sample_dev['win_pct']
    results['nmll_loss_sample_tr'] = results_sample_tr['nmll_loss_sample']
    results['nmll_loss_sample_dev'] = results_sample_dev['nmll_loss_sample']
    pickle.dump(results,
                open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    logger.info("Best Validation Loss = {:.4f}, "
      "Best Win_pct_val = {:.2f}%, " 
      "Best Val Loss on Test = {:.4f}, "
      "Best Win_pct_val_test = {:.2f}%, "
      "Final Training NMLL = {:.4f}, "
      "Training NMLL original = {:.4f}, "
      "Win_pct_train = {:.2f}%, "
      "Final Dev NMLL = {:.4f}, "
      "Dev NMLL original = {:.4f}, "
      "Win_pct_dev = {:.2f}%, "
      "Final Dev Test NMLL = {:.4f}, "
      "Dev Test NMLL original = {:.4f}, "
      "Win_pct_test_dev = {:.2f}%.".format(
        best_val_loss, \
        best_win_pct_val*100, \
        best_val_loss_test, \
        best_win_pct_val_test*100, \
        result_dataset_tr['nmll'], \
        result_dataset_tr['nmll_orig'], \
        result_dataset_tr['win_pct']*100, \
        result_dataset_dev['nmll'], \
        result_dataset_dev['nmll_orig'], \
        result_dataset_dev['win_pct']*100, \
        result_dataset_dev['nmll_test'], \
        result_dataset_dev['nmll_test_orig'], \
        result_dataset_dev['win_pct_test']*100))


    avg_nmll_tr = np.mean(results_sample_tr['nmll_sample_compare'],0)
    logger.info('% of GPs with higher marginal likelihood = {:.2f}%'.format(results_sample_tr['win_pct']*100))
    logger.info('Average NMLL on training samples: true = {}, learned = {}'.format(avg_nmll_tr[1],avg_nmll_tr[0]))
    avg_nmll_dev = np.mean(results_sample_dev['nmll_sample_compare'],0)
    logger.info('% of GPs with higher marginal likelihood = {:.2f}%'.format(results_sample_dev['win_pct']*100))
    logger.info('Average NMLL on testing samples: true = {}, learned = {}'.format(avg_nmll_dev[1],avg_nmll_dev[0]))
    snapshot(
        model.module if self.use_gpu else model,
        optimizer,
        self.config,
        self.train_conf.max_epoch + 1,
        tag='final')
    return None

  def validate(self):
    # create data loader
    dev_dataset = eval(self.dataset_conf.loader_name)(self.config, split='dev')
    subset_indices = range(self.subsample_size)
    dev_loader_sub = torch.utils.data.DataLoader(
          dev_dataset,
          batch_size=self.train_conf.batch_size,
          shuffle=False,
          num_workers=self.train_conf.num_workers,
          collate_fn=dev_dataset.collate_fn,
          drop_last=False,
          sampler=SubsetRandomSampler(subset_indices))
    # create models
    model = eval(self.model_conf.name)(self.model_conf)

    if self.use_gpu:
      model = nn.DataParallel(model, device_ids=self.gpus).cuda()

    # resume training or use prxetrained model
    if self.train_conf.is_resume:
      if self.train_conf.pretrain:
        model_snapshot = torch.load(self.train_conf.resume_model,map_location=self.device)
        model.load_state_dict(model_snapshot["model"],strict=False)
        model.to(self.device)
      else:
        model_snapshot = torch.load(self.train_conf.resume_model,map_location=self.device)
        model.load_state_dict(model_snapshot["model"],strict=True)
        model.to(self.device)

    with torch.no_grad():
      results_sample_dev = self.cal_sample_result(model,dev_loader_sub)
      result_dataset_dev = self.cal_dataset_loss(model,dev_loader_sub)
    
    results = defaultdict(list)
    results['nmll_dev_orig'] = result_dataset_dev['nmll_orig']	
    results['nmll_dev_test'] = result_dataset_dev['nmll_test']
    results['nmll_dev_test_orig'] = result_dataset_dev['nmll_test_orig']
    if self.is_val:
      results['win_pct_ai'] = result_dataset_dev['win_pct_ai']*100
      results['win_pct_ai_test'] = result_dataset_dev['win_pct_ai_test']*100
      results['nmll_dev_opt'] = result_dataset_dev['nmll_opt_sm']
      results['nmll_dev_test_opt'] = result_dataset_dev['nmll_opt_sm_test']

    results['win_pct'] = result_dataset_dev['win_pct']*100
    results['win_pct_test'] = result_dataset_dev['win_pct_test']*100
    logger.info(
      "Final Dev NMLL = {:.4f}, "
      "Dev NMLL original = {:.4f}, "
      "Win_pct_dev = {:.2f}%, "
      "Final Dev Test NMLL = {:.4f}, "
      "Dev Test NMLL original = {:.4f}, "
      "Win_pct_test_dev = {:.2f}%.".format(
        result_dataset_dev['nmll'], \
        result_dataset_dev['nmll_orig'], \
        result_dataset_dev['win_pct']*100, \
        result_dataset_dev['nmll_test'], \
        result_dataset_dev['nmll_test_orig'], \
        result_dataset_dev['win_pct_test']*100))
    if self.is_val:
      logger.info(
        "Dev NMLL Opt = {:.4f}, "
        "Dev Test NMLL Opt = {:.4f}, "
        "Win_pct_ai_vs_opt = {:.2f}%, "
        "Win_pct_test_ai_vs_opt = {:.2f}%.".format(
          result_dataset_dev['nmll_opt_sm'], \
          result_dataset_dev['nmll_opt_sm_test'], \
          result_dataset_dev['win_pct_ai']*100, \
          result_dataset_dev['win_pct_ai_test']*100))
    pickle.dump(results,
                open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()

  def test(self):
    npr = np.random.RandomState(self.config.test.seed)
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(root_path, self.config.test.data_path)
    filename = os.path.join(DATA_PATH, self.config.test.file_name + '.data')
    data = np.loadtxt(filename)
    data = shuffle(data, random_state=npr)
    x, y = data[:self.config.test.num_data, :-1], data[:self.config.test.num_data, -1]
    # add noise dimension
    if self.config.test.add_noise:
      noise = npr.randn(x.shape[0],1)*self.config.test.add_noise_level
      x = np.append(x,noise,1)
    if self.config.test.add_label:
      label = y[:,None]
      x = np.append(x,label,1)

    result_dic = {}
    nmll_opt_train = np.zeros(self.config.test.repeat)
    nmll_opt_test = np.zeros_like(nmll_opt_train)
    nmll_ai = np.zeros_like(nmll_opt_train)
    nmll_ai_test = np.zeros_like(nmll_opt_train)
    rmse_opt = np.zeros_like(nmll_opt_train)
    ll_opt = np.zeros_like(nmll_opt_train)
    time_elapsed_opt = np.zeros_like(nmll_opt_train)
    rmse_ai = np.zeros_like(nmll_opt_train)
    ll_ai = np.zeros_like(nmll_opt_train)
    time_elapsed_ai = np.zeros_like(nmll_opt_train)

    if self.config.test.warm_start:
      nmll_warmstart = np.zeros_like(nmll_opt_train)
      nmll_warmstart_test = np.zeros_like(nmll_opt_train)
      rmse_warmstart = np.zeros_like(nmll_opt_train)
      ll_warmstart = np.zeros_like(nmll_opt_train)
      time_elapsed_warmstart = np.zeros_like(nmll_opt_train)
    data_dim = x.shape[1]
    lengthscale_ai = np.zeros((self.config.test.repeat,data_dim))
    
    for ii in tqdm(range(self.config.test.repeat)):
      x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.1, random_state=npr)
      num_data = x_t.shape[0]
      x_t, x_v, _, _ = standardize(x_t, x_v)
      x_t = x_t*0.1
      x_v = x_v*0.1
      y_t, y_v, _, std_y_train = standardize(y_t, y_v)
      data = {}
      data['X'] = x_t
      data['f'] = y_t
      data['X_2'] = x_v
      data['f_2'] = y_v

      #-----------------------Perform marginal likelihood optimization using PyTorch AutoDiff--------------------
      if self.config.test.mll_opt:
        torch.manual_seed(0)
        settings = edict()
        settings.epsilon = self.config.test.epsilon
        settings.lr = self.config.test.lr
        settings.training_iter = self.config.test.train_iter
        settings.is_print = self.config.test.is_print
        settings.device = self.device
        settings.opt_is_lbfgs = self.config.test.opt_is_lbfgs
        model_params = edict()
        model_params.input_dim = x_t.shape[1]
        model_params.num_mix = 10
        model_params.is_dim_diff = True
        model_params.is_no_mu = self.config.test.is_no_mu
        model_params.warm_start = False
        mu_pred, var_pred, sm_params, time_elapsed = nmll_opt_gp(data, model_params, settings)



        rmse = np.mean((mu_pred - y_v) ** 2) ** .5 * std_y_train
        log_likelihood = np.mean(np.log(stats.norm.pdf(
                            y_v,
                            loc=mu_pred,
                            scale=var_pred ** 0.5))) - np.log(std_y_train)


        nmll_opt_train[ii] = data['nmll_opt_sm']
        nmll_opt_test[ii] = data['nmll_opt_sm_test']
        rmse_opt[ii] = rmse
        ll_opt[ii] = log_likelihood
        time_elapsed_opt[ii] = time_elapsed

      # ----------------------------Use Amortized Model---------------------------------
      train_x = torch.from_numpy(data['X']).float().to(self.device)
      train_y = torch.from_numpy(data['f']).float().unsqueeze(-1).to(self.device)
      test_x = torch.from_numpy(data['X_2']).float().to(self.device)
      test_y = torch.from_numpy(data['f_2']).float().unsqueeze(-1).to(self.device)
      data['X_data'] =torch.from_numpy(data['X']).float().unsqueeze(0).to(self.device) # 1 X N X D
      data['F'] = torch.from_numpy(data['f']).float().unsqueeze(0).to(self.device) # 1 X N
      data['node_mask'] = torch.ones(num_data).unsqueeze(0).to(self.device) # 1 X N
      data['diagonal_mask'] = torch.zeros(num_data).unsqueeze(0).to(self.device) # 1 X N
      data['dim_mask'] = torch.ones(data_dim).unsqueeze(0).to(self.device) # 1 X D
      data['kernel_mask'] = torch.ones(num_data,num_data).unsqueeze(0).to(self.device) # 1 X N X N
      data['N'] = torch.ones(1).to(self.device) * num_data # 1
      #create model and load pretrained model
      model = eval(self.model_conf.name)(self.model_conf)
      model_snapshot = torch.load(self.test_conf.test_model, map_location=self.device)
      model.load_state_dict(model_snapshot["model"], strict=True)
      model.to(self.device)
      if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).cuda()

      model.eval()
      time_start = time.time()
      with torch.no_grad():
        if self.model_conf.name == 'GpSMDoubleAtt':
          mu, var, weights, nmll = model(data['X_data'],data['X_data'],data['F'],data['F'],data['node_mask'],data['dim_mask'],data['kernel_mask'],data['diagonal_mask'],data['N'], device = self.device)
        elif self.model_conf.name == 'GpSMDoubleAttNoMu':
          var, weights, nmll = model(data['X_data'],data['X_data'],data['F'],data['F'],data['node_mask'],data['dim_mask'],data['kernel_mask'],data['diagonal_mask'],data['N'], device = self.device)
        else:
          raise ValueError("No model of given name!")
      time_end= time.time()
      time_ai = time_end - time_start
      time_elapsed_ai[ii] = time_ai

      epsilon = self.config.test.epsilon
      var = var.squeeze(0)
      weights = weights.squeeze(0)
      if self.is_no_mu:
        K11 = cal_kern_spec_mix_nomu_sep(train_x, train_x, var, weights)
        K12 = cal_kern_spec_mix_nomu_sep(train_x, test_x, var, weights)
        K22 = cal_kern_spec_mix_nomu_sep(test_x, test_x, var, weights)
      else:
        mu = mu.squeeze(0)
        K11 = cal_kern_spec_mix_sep(train_x, train_x, mu, var, weights)
        K12 = cal_kern_spec_mix_sep(train_x, test_x, mu, var, weights)
        K22 = cal_kern_spec_mix_sep(test_x, test_x, mu, var, weights)
      nmll = -cal_marg_likelihood_single(K11, train_y, epsilon, self.device)
      nmll_test = -cal_marg_likelihood_single(K22, test_y, epsilon, self.device)
      mu_test, var_test = GP_noise(train_y, K11, K12, K22, epsilon, self.device)
      mu_test = mu_test.detach().squeeze(-1).cpu().numpy()
      var_test = var_test.detach().squeeze(-1).cpu().numpy().diagonal()
      rmse = np.mean((mu_test - y_v) ** 2) ** .5 * std_y_train
      log_likelihood = np.mean(np.log(stats.norm.pdf(
                                  y_v,
                                  loc=mu_test,
                                  scale=var_test ** 0.5))) - np.log(std_y_train)
      nmll_ai[ii] = nmll.cpu().item()
      nmll_ai_test[ii] = nmll_test.cpu().item()
      rmse_ai[ii] = rmse
      ll_ai[ii] = log_likelihood

      # perform mll opt from hyper-parameters initized by our model
      if self.config.test.warm_start:
        settings_warmstart = edict()
        settings_warmstart.epsilon = self.config.test.epsilon
        settings_warmstart.lr = self.config.test.lr_warmstart
        settings_warmstart.training_iter = self.config.test.train_iter_warmstart
        settings_warmstart.is_print = self.config.test.is_print
        settings_warmstart.device = self.device
        settings_warmstart.opt_is_lbfgs = self.config.test.opt_is_lbfgs
        model_params_warmstart = edict()
        model_params_warmstart.input_dim = x_t.shape[1]
        model_params_warmstart.num_mix = 10
        model_params_warmstart.is_dim_diff = True
        model_params_warmstart.is_no_mu = self.config.test.is_no_mu
        model_params_warmstart.warm_start = True
        if not model_params_warmstart.is_no_mu:
          model_params_warmstart.mu_init = torch.log(mu.detach())
        model_params_warmstart.var_init = torch.log(var.detach())
        model_params_warmstart.weights_init = torch.log(weights.detach())

        mu_pred, var_pred, sm_params, time_warmstart = nmll_opt_gp(data, model_params_warmstart, settings_warmstart)


        rmse = np.mean((mu_pred - y_v) ** 2) ** .5 * std_y_train
        log_likelihood = np.mean(np.log(stats.norm.pdf(
                            y_v,
                            loc=mu_pred,
                            scale=var_pred ** 0.5))) - np.log(std_y_train)

        nmll_warmstart[ii] = data['nmll_opt_sm']
        nmll_warmstart_test[ii] = data['nmll_opt_sm_test']
        rmse_warmstart[ii] = rmse
        ll_warmstart[ii] = log_likelihood
        time_elapsed_warmstart[ii] = time_warmstart + time_ai

    result_dic['nmll_opt_train'] = nmll_opt_train
    result_dic['nmll_opt_test'] = nmll_opt_test
    result_dic['nmll_ai'] = nmll_ai
    result_dic['nmll_ai_test'] = nmll_ai_test
    result_dic['rmse_opt'] = rmse_opt
    result_dic['ll_opt'] = ll_opt
    result_dic['rmse_ai'] = rmse_ai
    result_dic['ll_ai'] = ll_ai
    result_dic['time_opt'] = time_elapsed_opt
    result_dic['time_ai'] = time_elapsed_ai  

    logger.info("RMSE OPT mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(rmse_opt),np.std(rmse_opt),np.nanmin(rmse_opt),np.max(rmse_opt)))
    logger.info("MLL OPT mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(ll_opt),np.std(ll_opt),np.nanmax(ll_opt),np.min(ll_opt)))
    logger.info("Time OPT mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(time_elapsed_opt),np.std(time_elapsed_opt),np.min(time_elapsed_opt),np.max(time_elapsed_opt)))
    logger.info("RMSE AI mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(rmse_ai),np.std(rmse_ai),np.nanmin(rmse_ai),np.max(rmse_ai)))
    logger.info("MLL AI mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(ll_ai),np.std(ll_ai),np.nanmax(ll_ai),np.min(ll_ai)))
    logger.info("Time AI mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(time_elapsed_ai),np.std(time_elapsed_ai),np.min(time_elapsed_ai),np.max(time_elapsed_ai)))
    if self.config.test.warm_start:
      logger.info("RMSE Warmstart mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(rmse_warmstart),np.std(rmse_warmstart),np.nanmin(rmse_warmstart),np.max(rmse_warmstart)))
      logger.info("MLL Warmstart mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(ll_warmstart),np.std(ll_warmstart),np.nanmax(ll_warmstart),np.min(ll_warmstart)))
      logger.info("Time Warmstart mean = {:.4f}, std = {:.4f}, best = {:.4f}, worst = {:.4f}".format(np.mean(time_elapsed_warmstart),np.std(time_elapsed_warmstart),np.min(time_elapsed_warmstart),np.max(time_elapsed_warmstart)))


    if self.config.test.is_save:
      if self.config.test.opt_is_lbfgs:
        pickle.dump(result_dic,
                    open(os.path.join(self.config.test.save_dir, self.config.test.file_name + str(self.config.test.num_data) + str(self.config.test.add_noise) +'_lbfgs_results.p'), 'wb'))
      else:
        pickle.dump(result_dic,
                    open(os.path.join(self.config.test.save_dir, self.config.test.file_name + str(self.config.test.num_data) + str(self.config.test.add_noise) +'_Adam_results.p'), 'wb'))
    
    return None