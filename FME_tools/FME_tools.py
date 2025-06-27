import os
import numpy as np
import scipy.signal
import torch
import itertools
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth
from copy import deepcopy
from obspy import read
  
class Tools():
    def __init__(self):
        pass

    def get_glob_idx(self,x):
        net, sta, chn, loc, _ = os.path.basename(x).split('.')
        return f'{net}.{sta}.{chn[:2]}?.{loc}.sac'
    
        # creat taper

    def taper(self,wave, params, t_index=None):
        length = len(wave)
        buffer = params['taper_len']

        if buffer > length:
            raise ValueError(f'taper_len ({buffer}) is greater than wave length ({length})')

        taper = np.ones(length)
        half_window = np.hanning(buffer * 2)[:buffer]

        if t_index is not None:
            if t_index > length:
                raise ValueError(f't_index ({t_index}) is greater than wave length ({length})')

            taper[:t_index - buffer] = 0  # 直接設定 0 避免多次索引
            taper[t_index - buffer:t_index] = half_window  # 使用切片加快賦值
        else:
            taper[:buffer] = half_window  # 頭部應用 taper

        wave *= taper  # 就地修改 wave，提高效率
        return wave
    
    def station_name_transform(self,E_name,Z_name):
        network,network_code,sta_type,location = Z_name.split('.')[:4]
        compon = str(E_name.split('.')[2][-1])
        station_id = f'{network}.{network_code}.{location}.{sta_type[:2]}'
        skip = False

        if compon not in ['E','N','1', '2']:
            print(f'station component error, imput com:{compon}')
            skip = True

        valid_locations = ['10', '11', '01', '00', '02']
        if location in valid_locations:
            if location == '20':
                skip = True  # location 20 is seabed seismometer, do not use
        else:
            print(f'location error, input: {location}, type: {type(location)}')
            skip = True

        if sta_type[-2] == 'H': # velocity typed(no used)
            skip = True #unit = 'm/s'
        elif sta_type[-2] not in ['L', 'N']:
            print(f"station type:{sta_type} can't find")

        return {"skip": skip, "station_id": station_id}
    
    def read_sac(self,sac_file,sta_df,params):
        trc = read(sac_file)
        trc = deepcopy(trc)
        rate = trc[0].stats['sampling_rate']
        set_rate = params['sample_rate']
        array = trc[0].data

        if np.isnan(array.any()):
            eventID = sta_df['eventID']
            sta_id = sta_df['station_name']
            print(f'event:{eventID} sta_name:{sta_id} compo:E, array have nan value')
            return True,array

        if rate != set_rate:
            eventID = sta_df['eventID']
            sta_id = sta_df['station_name']
            print(f'event:{eventID} sta_name:{sta_id}, sample rate:{rate} not {set_rate}')
            return True,array

        return False,array
    
    def data_process(self,array,factor,params):
        array *= factor
        array = scipy.signal.detrend(array)
        wave = self.taper(array,params)
        return wave

    def gen_slice_list(self,st_index,params,random = False):
        length = params['sw_length']
        step = params["sw_step"]
        range_len =  params["slice_number"]
        start = st_index-length
        random = params['random_slice']
        if random == False:
            return [(start, start + length) for start in range(start,start+step*range_len,step)]
        if random == True:
            #ran = int(round(np.random.rand()*params['sw_step']))
            result = []
            for i in range(range_len):
                base = start + i * step
                offset = int(round(np.random.rand() * params['sw_step']))
                result.append((base + offset, base + offset + length))
            return result
    # input slice list output slice data
    def slice_data(self, slice_list, data):
        return [data[start:end] for start, end in slice_list]
            
    def compare_units(self, sta1, sta2):
        # 解析站名
        _, _, num1, eng1 = sta1.split('.')
        _, _, num2, eng2 = sta2.split('.')

        # 定義優先級
        english_priority = {'HL': 0, 'HN': 1, 'HH': 2, 'EH': 3}
        numeric_priority = {'10': 0, '00': 1, '01': 2, '11': 3}

        # 取得對應的優先級
        pri_eng1, pri_eng2 = english_priority[eng1], english_priority[eng2]
        pri_num1, pri_num2 = numeric_priority[num1], numeric_priority[num2]

        # 直接比較 (先比英文字母優先級，再比數字優先級)
        return sta1 if (pri_eng1, pri_num1) > (pri_eng2, pri_num2) else sta2

    def calc_epi_distance(self,sta1_lat, sta1_lon, sta2_lat, sta2_lon):
        dh = gps2dist_azimuth(sta1_lat, sta1_lon, sta2_lat, sta2_lon)[0]/1000
        return float(dh)

    # output station selection list
    def real_time_sta_selection(self, data_flow):
        # 依據時間排序
        data_flow = sorted(data_flow, key=lambda data: data[0])
        
        # 取得站點位置
        sta_loc = [(data[2], data[3]) for data in data_flow]
        
        # 初始化集合來存放要移除的站點
        remove_sta = set()

        # 計算距離並比較單位，使用 itertools.combinations 減少計算次數
        for (i, data_i), (j, data_j) in itertools.combinations(enumerate(data_flow), 2):
            dist = self.calc_epi_distance(sta_loc[i][0], sta_loc[i][1], sta_loc[j][0], sta_loc[j][1])
            
            if dist <= 1:
                res = self.compare_units(data_i[1], data_j[1])
                remove_sta.add(res)
        return list(remove_sta)

    def pad_waves_to_same_length(self,*waves):
        """
        Pad multiple 1D waveforms with zeros to make them the same length.

        Args:
            *waves: Variable number of 1D NumPy arrays.

        Returns:
            List of padded NumPy arrays with equal length.
        """
        max_len = max(len(w) for w in waves)
        return [np.pad(w, (0, max_len - len(w)), mode='constant') for w in waves]
    
    def wave_shift(self,timedelta: float, wave: np.ndarray, sample_rate: float):
        shift_samples = int(round(timedelta * sample_rate))
        if shift_samples < 0:
            raise ValueError("Negative shift not supported.")
        new_wave = np.pad(wave, (shift_samples, 0), mode='constant')
        return new_wave

    def plt_dataset(self,dataset,mag,eventID=None,save=False):
        data = dataset

        #print(f"Data: {data}, Target: {target}")
        x1 = data[:,:,:2000]
        x2 = data[:,:,2000:2001]

        component_index = 2

        max_int = [0]
        max_list = []
        max0 = np.max(x1)
        plt.figure(figsize=(6,9))
        for i in range(10):
            station_data = x1[component_index,i, :]
            max1 = max_int[i]+max0
            max_int.append(max1)
            max_list.append(np.full(2000,max_int[i]))
            dis = x2[component_index,i,:]
            print(dis)

            # 画出波形
            plt.plot(station_data-max_list[i],linewidth=0.7)

        #print(max_int)
        plt.ylabel(f'Station Order (P arrival)') #dis:{int(hypo)}'
        plt.suptitle(f'{eventID} Waveform Mag:{mag}')
        plt.xlabel('Time (100Hz)')
        plt.tight_layout()
        if save == False:
          plt.show()
        else:
          plt.savefig(save, dpi=400)
          plt.show()

    def calculate_corrcoef(self,x,y):
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        if len(x.shape)==2:
            outlen = 100
            xprobs = torch.nn.functional.softmax(x, dim=-1)
            yprobs = torch.nn.functional.softmax(y, dim=-1)
            idx = (torch.arange(outlen)*6/outlen).to('cuda')
            x = torch.sum(xprobs*idx,1)
            y = torch.sum(yprobs*idx,1)
        cat = torch.stack((x, y), dim=0)
        corcoef = torch.corrcoef(cat)
        return corcoef[0,1]
