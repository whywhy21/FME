import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

#建立資料集
class EventDataDataset(Dataset):
  def __init__(self, root_dir,csv_dir,output_len = 100,shuffle=True,gaussian=False):
    self.root_dir = root_dir
    self.rcsv = pd.read_csv(csv_dir)
    self.shuffle = shuffle
    self.gaussian = gaussian
    self.output_len = output_len

  def __len__(self):
    return len(self.rcsv) 
  
  def __getitem__(self, index):
    data_path = os.path.join(self.root_dir, self.rcsv.iloc[index,0])
    data_path = data_path+'.npy'
    data_slice = self.rcsv.iloc[index,1]
    mag = self.rcsv.iloc[index,2]

    if self.gaussian == True:
      min_val, max_val = 3,9
      sigma = 3
      mapped_center = (mag - min_val) / (max_val - min_val) * (self.output_len - 1)
      x = torch.arange(self.output_len)
      gaussian = torch.exp(-0.5 * ((x - mapped_center) / sigma) ** 2)
      gaussian /= gaussian.sum()
      mag = gaussian

    data = np.load(data_path,mmap_mode="r")
    data = data[int(data_slice)]

    # shuffled station domain
    if self.shuffle == True:
      shuffled = np.random.permutation(data.shape[1])
      data = data[:,shuffled,:]

    data = torch.tensor(data)
    target = torch.tensor(mag)
    return (data,target)

