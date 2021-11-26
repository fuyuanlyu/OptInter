import os
import numpy as np
import pandas as pd
import torch
import shutil
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def count_parameters_in_GB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e8

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def save_arch(model, model_path):
    np.save(model_path, model.get_arch_parameters().numpy())

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def load_arch(model_path):
    return np.load(model_path)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

