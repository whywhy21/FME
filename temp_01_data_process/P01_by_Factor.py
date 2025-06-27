"""Data Process Before Create dataset"""
import os
import sys
from tqdm import tqdm
from glob import glob
sys.path.append('../FME_tools')
from FME_tools import Tools
import numpy as np
import pandas as pd
import time
#############################################
start = time.time()
## Sac file, Event catalog, npy file save fold
datadir = np.sort(glob('./event_data/*'))
clog = pd.read_csv('../Dataset_info.csv')
save_fold = './event_npy'
tools = Tools()

## some important prameters
params = {
            "taper_len": 50,      # 100Hz length, 50 means 0.5 second
            "save_npy": True,     # whether save to npy file
            "sample_rate": 100.0  # seting sample rate, if there are diffrent with sac records, skip
         }                        # please confirm sample rate are the same
############################################
for i in range(len(datadir)):
    print(f"Processing {i+1}/{len(datadir)}: {datadir[i]}")
    sacs = glob(os.path.join(datadir[i], '*.sac'))

    # get eventID and event info from catalog
    eventID = datadir[i].split('/')[-1]
    event_info_df = clog.loc[clog['eventID']==eventID]

    # map get_glob_idx to sac files using map
    glob_idx = np.unique([tools.get_glob_idx(x) for x in sacs])

    for ct in tqdm(range(len(glob_idx))):
        files = np.sort(glob(os.path.join(datadir[i], glob_idx[ct])))
        # check files have 3 component
        if not len(files)==3 :
            continue

        # Get 3component station name and network code
        E_sta_name = files[0].split('/')[-1][:-4]
        N_sta_name = files[1].split('/')[-1][:-4]
        Z_sta_name = files[2].split('/')[-1][:-4]

        # Get station id -> NETWORK.STATION.LOCATION.CHANNEL
        sta_info = tools.station_name_transform(E_sta_name,Z_sta_name)
        if sta_info['skip']:
            continue

        # Search catalog, Get station info, check station is unique
        sta_df = event_info_df[event_info_df['station_name']==sta_info['station_id']]
        sta_id = str(sta_df['station_name'].values[0])
        if len(sta_df)!=1:
            print(f"Compare station !=1 event:{eventID} staID:{sta_id}")
            continue
        sta_df = sta_df.iloc[0]

        # get ENZ station factor and check factor value
        E_factor = float(sta_df['E_factor'])
        N_factor = float(sta_df['N_factor'])
        Z_factor = float(sta_df['Z_factor'])
        if np.isnan(E_factor) or np.isnan(E_factor) or np.isnan(E_factor):
            print(f'event:{eventID} station:{sta_id} factor is nan')
            continue

        # Read data for 3 components, transform to float and filter out acceleration stations
        Eskip, E_array = tools.read_sac(files[0],sta_df,params)
        Nskip, N_array = tools.read_sac(files[1],sta_df,params)
        Zskip, Z_array = tools.read_sac(files[2],sta_df,params)
        if Eskip or Nskip or Zskip:
            continue

        # data process, factor, detrend, taper
        E_wave = tools.data_process(E_array, E_factor, params)
        N_wave = tools.data_process(N_array, N_factor, params)
        Z_wave = tools.data_process(Z_array, Z_factor, params)

        # shift the wave whitch is
        E_wave = tools.wave_shift(sta_df['E_shift(s)'],E_wave,sta_df['sampling_rate'])
        N_wave = tools.wave_shift(sta_df['N_shift(s)'],N_wave,sta_df['sampling_rate'])
        Z_wave = tools.wave_shift(sta_df['Z_shift(s)'],Z_wave,sta_df['sampling_rate'])

        # slice the waveform make sure 3 component len are the same
        if not (len(E_wave) == len(N_wave) == len(Z_wave)):
            E_wave, N_wave, Z_wave = tools.pad_waves_to_same_length(E_wave, N_wave, Z_wave)

        # create waveform list
        waveform_ACC = np.array([E_wave,N_wave,Z_wave],dtype="float32")

        # create np.array waveform and save to npy
        if params["save_npy"]:
            folder_path = f"{save_fold}/{eventID}/"
            os.makedirs(folder_path, exist_ok=True)
            np.save(f'{folder_path}{sta_id}.npy', waveform_ACC)

end = time.time()
print(f'time:{end-start}')
print('eventdata npy file done!')
