"""To Create PyTorch Model Dataset From npy file"""
import math
import os
import time
import sys
from glob import glob
import pandas as pd
import numpy as np
sys.path.append('../FME_tools')
from FME_tools import Tools
#############################################
start_time = time.time()
## dataset path, Event catalog, npy file fold
raw_data_fold = './event_npy'
dataset_save_path = '../dataset'
info_csv = '../Dataset_info.csv'
info_read = pd.read_csv(info_csv)
datadir = np.sort(glob(f'{raw_data_fold}/*'))
tools = Tools()

# some important prameters
params = {
         "firs_ppick" : 4,       # slice will start after this pick order
         "station_number" : 10,  # match the model, don't change
         "slice_number" : 36,    # how many slice after firs_ppick 
         "sw_step" : 50,         # interval time of every slice, 100=1s
         "sw_length" : 2000,     # slice waveform length, match the model, don't change
         "taper_len" : 50,       # The step taper takes to go from 0 to 1
         "save_npy": True,      # whether save to npy
         "sample_rate": 100.0,   # check sample rate
         "random_slice":True     # random the window when slice waveform 
         }
Buffer_time = 50
##############################################
# Create dataset
for i in range(len(datadir)):
   pkls = glob(os.path.join(datadir[i], '*.npy'))
   print(f"Processing {i+1}/{len(datadir)}: {datadir[i]}")

   # get eventID from file name and search event info
   eventID = datadir[i].split('/')[-1]
   info_df = info_read[info_read['eventID'] == eventID]
   if len(info_df)==0:
      print(f"eventID:{eventID} didn't exist in catalog csv.")
      continue
   mag = np.unique(info_df['magnitude'].tolist())
   if len(mag)!=1:
      print(f"eventID:{eventID} Mag:{mag} not only 1 value.")
      continue

   # if the P_index less than sw_length append zero
   P_arrivals = info_df['Z_P_idx'].values
   under_sw = False
   if any(P < params['sw_length'] for P in P_arrivals):
      under_sw = True

   # get the station info and putin list
   data_flow = []
   sta_selection_info = []
   for ct in range(len(pkls)):
      station = pkls[ct].split('/')[-1][:-4]
      sta_info = info_df.loc[info_df['station_name']==station]
      if len(sta_info)>1:
         try:
            sta_info = sta_info.loc[sta_info['trace_name'].str.contains("SMT")]
         except:
            print('eventID',eventID,'station:',station,'have over 1 station')
         continue
      elif len(sta_info)<1:
         print('eventID',eventID,'station:',station,'have no station')
         continue
      else:
         pass
      sta_info = sta_info.iloc[0]
      P_arrival = min([sta_info['E_P_idx'],sta_info['N_P_idx'],sta_info['Z_P_idx']])
      sta_lat, sta_lon = sta_info['sta_latitude'],sta_info['sta_longitude']

      # save station selection info
      sta_selection_info.append([P_arrival,station,sta_lat,sta_lon])

   # selection station, check distance from every station are <1km
   remove_sta = set(tools.real_time_sta_selection(sta_selection_info))
   data_flow = [row for row in sta_selection_info if row[1] not in remove_sta]

   # if station number <10(station number) ,go to next event
   #print(len(data_flow))
   if len(data_flow)< params['station_number']:
      print(f"eventID:{eventID} station record less than {params['station_number']}, don't use")
      continue

   # sorted by P_arrival and get first 10 P_arrival station
   sorted_data = sorted(data_flow, key=lambda data: data[0])
   sorted_data = sorted_data[0:params['station_number']]

   # get fifth P_arrival
   first_p_index = sorted_data[params['firs_ppick']-1][0]
   if under_sw:
      first_p_index+=params['sw_length']

   # get slice list
   st_index = first_p_index+Buffer_time
   slice_list = tools.gen_slice_list(st_index,params)

   # slice data
   station_wdata = []
   last_true_index = []
   for j in range(len(sorted_data)):
      sta_n = sorted_data[j][1]
      P_index = sorted_data[j][0]
      sta_info = info_df[info_df['station_name']==sta_n]
      # check sta_info is the only one
      if len(sta_info)>1:
         try:
            sta_info = sta_info.loc[sta_info['trace_name'].str.contains("SMT")]
         except:
            print('eventID',eventID,'station:',station,'have over 1 station')
         continue
      elif len(sta_info)<1:
         print('eventID',eventID,'station:',station,'have no station')
         continue
      else:
         pass
      
      sta_info = sta_info.iloc[0]
      pkl = f'{datadir[i]}/{sta_n}.npy'
      data = np.load(pkl)

      # get wave info
      E_wave,N_wave,Z_wave = data[0], data[1], data[2]
      hypo_dist = sta_info['hypo_dist']

      # append 0
      if under_sw:
         P_index+=params['sw_length']
         nzero = np.zeros(params['sw_length'], dtype=E_wave.dtype)
         E_wave = np.append(nzero,E_wave)
         N_wave = np.append(nzero,N_wave)
         Z_wave = np.append(nzero,Z_wave)

      # taper and slice wave
      E_wave = tools.taper(E_wave,params,t_index=P_index)
      N_wave = tools.taper(N_wave,params,t_index=P_index)
      Z_wave = tools.taper(Z_wave,params,t_index=P_index)

      E_sw = tools.slice_data(slice_list,E_wave)[:params['slice_number']]
      N_sw = tools.slice_data(slice_list,N_wave)[:params['slice_number']]
      Z_sw = tools.slice_data(slice_list,Z_wave)[:params['slice_number']]

      # if P_index in slice list ,distence is True, if not is False, and also the waveform
      dis_is_list = [True if interval[0] <= P_index <= interval[1] else False for interval in slice_list[:params['slice_number']]]
      last_true_index.append(np.where(dis_is_list)[0][-1])
      wave_is_list = [True if (P_index-interval[1]) <= -Buffer_time else False for interval in slice_list[:params['slice_number']]]

      # append distance to array
      ENZ_sw = []
      for E_slice, N_slice, Z_slice, dis_p,wave_p in zip(E_sw, N_sw, Z_sw, dis_is_list,wave_is_list):

         E_slice, N_slice, Z_slice = map(np.array, (E_slice, N_slice, Z_slice))

         # decide whether hypo distance and waveform is exist in dataset
         if wave_p is False:
            E_slice = np.zeros(params['sw_length'])
            N_slice = np.zeros(params['sw_length'])
            Z_slice = np.zeros(params['sw_length'])
         if dis_p and wave_p == True:
            append_value = np.array([hypo_dist])
         elif dis_p or wave_p == False:
            append_value = np.array([0])
         else:
            continue

         E_ap, N_ap, Z_ap = [np.append(arr, append_value)[np.newaxis, :] for arr in (E_slice, N_slice, Z_slice)]
         ENZ_sw.append(np.concatenate((E_ap,N_ap,Z_ap),axis=0))

      station_wdata.append(ENZ_sw)
   
   sl_len = min(last_true_index)+1
   # Transpose list of lists and get dataset
   station_wdata = list(map(list,zip(*station_wdata)))
   dataset = [
            np.stack([np.array(wave)[np.newaxis, :, :] for wave in sl_data], axis=1)
            for sl_data in station_wdata
             ]
   dataset = np.squeeze(dataset).transpose(0,2,1,3)
   dataset = dataset[:sl_len,:,:,:]
   dataset = np.array(dataset, dtype=np.float32)
   ds_shape = dataset.shape
   assert ds_shape == (sl_len,3 ,10 ,2001)

   if params['save_npy']:
      npy_save_root = f'{dataset_save_path}/{eventID}.npy'
      os.makedirs(dataset_save_path, exist_ok=True)
      np.save(npy_save_root, dataset)

   #for i in range(len(dataset)):
   #   data = dataset[i,:,:,:]
   #   tools.plt_dataset(data,mag,eventID=eventID,save=False)

print(f'Dataset done!')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.8f} s")
