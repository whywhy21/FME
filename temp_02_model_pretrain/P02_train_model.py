"""Train Model and Write Log"""
import sys
import torch
import csv
import time
import torch
sys.path.append('../FME_tools')
from tqdm import tqdm
from model import FMEModel
from Loaddata import EventDataDataset
from FME_tools import Tools
from torch.utils.data import DataLoader

####################### roots
dataset_root = '../dataset'
dataset_train_csv = '../dataset.csv'
check_point = '../pretrained_model/FME_05Hz_model.pth.tar'
save_point = '../pretrained_model/FME_new_save.pth.tar'
log = './training_log.csv'
####################### prameters
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cpu or cuda
write_csv = True        # Save training log to csv
load_model = True       # Use pretrained model
batch_size = 256        # If gpu available input the number data at once
early_save_ep = 5       # Start save model after this epoch
num_epochs = 100        # The times of model training loop
dataset_ratio = (8,1,1) # Train,valid,test dataset ratio
####################### train model

# Load Data and Split Dataset
dataset = EventDataDataset(root_dir=dataset_root,csv_dir=dataset_train_csv,shuffle=True,gaussian=False)
ratio_all = sum(dataset_ratio)
train_size = int(len(dataset)*dataset_ratio[0]/ratio_all)
valid_size = int(len(dataset)*dataset_ratio[1]/ratio_all)
test_size = len(dataset)-valid_size-train_size
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_size ,valid_size , test_size])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True,pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,pin_memory=True)
print('train size:'+str(train_size)+' valid size:'+str(valid_size)+' test size:'+str(test_size))

# Get model and set optimizer
model = FMEModel(device=device,loss='MSE')
tools = Tools()
optimizer = model.optimizer
print(f'model device:{device}')
print(model.model)

# print Number of model parameters
pytorch_total_params = sum(p.numel() for p in model.model.parameters())
print('Number of model parameters:'+str(pytorch_total_params))

# load model parameter
if load_model:
    model.load(check_point, optim=False)

# write csv header
if write_csv:
  with open(log, 'w', newline='') as file:
      fieldnames = ['epoch','loss','valid_loss','corrcoef','valid_corrcoef',
                    'time']
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      
      # Write the header
      writer.writeheader()

# check accuracy in model trainning
def check_accuracy(loader,model):
  losses = []
  corcoef_list = []
  for x, y in loader:
    # data to device
    x = x.to(device=device)
    y = y.to(device=device)
    # forward
    scores,loss = model.predict(x,y)
    losses.append(loss.item())

    cor = tools.calculate_corrcoef(y,scores)
    if not torch.isnan(cor):
      corcoef_list.append(cor.item())

  mean_loss = sum(losses)/len(losses)
  mean_corcoef = sum(corcoef_list) / len(corcoef_list)

  return mean_loss,mean_corcoef

# Train Network
def train_network(loader,num_epochs,model):
  # record loss
  valid_mean_loss_list = []
  
  # record time and set epochs
  t0 = time.time()
  for epoch in range(num_epochs):
    losses = []
    corrcoef = []
    t1 = time.time()

    for x,y in tqdm(loader):

      data = x.to(device=device)
      targets = y.to(device=device)

      # forword
      scores,loss = model.train(data,targets)
      losses.append(loss.item())

      cor = tools.calculate_corrcoef(targets,scores)
      if not torch.isnan(cor):
        corrcoef.append(cor.item())

      # backward
      optimizer.zero_grad()
      loss.backward()

      # adam step
      optimizer.step()

    mean_loss = sum(losses)/len(losses)
    mean_corr = sum(corrcoef) / len(corrcoef)
    valid_mean_loss, valid_corcoef = check_accuracy(valid_loader,model)

    # record time
    t2 = time.time()
    epoch_time = t2-t1

    # write csv and print model state
    rloss,rtloss = round(mean_loss,4),round(valid_mean_loss,4)
    rcc,rtcc = round(mean_corr,4),round(valid_corcoef,4)
    rt = round(epoch_time,2)
    with open(log, 'a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([epoch+1,rloss,rtloss,rcc,rtcc,rt])
    
    print(f'epoch{epoch+1} loss:{rloss} valid loss:{rtloss} '
          f'corrcoef:{rcc} valid corrcoef:{rtcc} time:{rt}s')

    # save checkpoint if epoch>5 have max accuracy
    if epoch > early_save_ep and min(valid_mean_loss_list)>valid_mean_loss:
        model.save(save_point)
    valid_mean_loss_list.append(valid_mean_loss)

  t3 = time.time()
  modle_time = t3-t0
  print(f'model trainning time:{round(modle_time,1)}s')

# train the network and test
train_network(train_loader,num_epochs,model)
test_loss,test_corcoef = check_accuracy(test_loader,model)
print(f'test loss:{round(test_loss,4)} test corrcoef:{round(test_corcoef,4)}')
print('Model Training is Successful')
