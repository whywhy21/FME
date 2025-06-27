"""Model Predict and Write CSV"""
import sys
import torch
import numpy as np
import pandas as pd
sys.path.append('../FME_tools')
from glob import glob
from model import FMEModel

######################
check_point = '../pretrained_model/FME_new_save.pth.tar'
dataset_root = '../dataset'
output_csv = './predict_result.csv'
catalog = '../Dataset_info.csv'
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #cpu or cuda
write_csv = True
######################
model = FMEModel(device=device,loss='MSE')
model.load(check_point)
print(f'model device:{device}')

clog = pd.read_csv(catalog)
datadir = glob(f'{dataset_root}/**.npy')
all_predict = []
for i in range(len(datadir)):
  root = datadir[i]
  eventID = root.split('/')[-1][:-4]
  evlog = clog[clog['eventID']==eventID]

  target = float(pd.unique(evlog['magnitude']))
  data = np.load(root)

  y = torch.tensor(target).to(device=device)
  x = torch.tensor(data).to(device=device)

  output,loss = model.predict(x,y)
  output = torch.flatten(output)

  row_info = [(eventID, j+1, target, round(float(output[j]),4)) for j, (_) in enumerate(output)]
  all_predict.extend(row_info)

if write_csv:
    df = pd.DataFrame(all_predict, columns=['eventID', 'time', 'T_Mag','P_Mag'])
    df.to_csv(output_csv, index=False)
    print(f'csv save successful, please check. root:{output_csv}')