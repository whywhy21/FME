# FME
This is the official implementation of **Enhanced Rapid Estimation of Earthquake Magnitude : A Machine Learning Approach Utilizing Multi-Station Seismic Recordings**<br />

## Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Warning](#Warning)
* [FME Input Format](#FME-input-format)
* [Run Template Codes](#run-template-codes)

### Installation
To run this repository, we suggest to install packages with Anaconda.

Clone this repository:

```bash
git clone https://github.com/whywhy21/FME.git
cd FME
```

Create a new environment via conda & pip

```bash
conda update conda
conda create --name FME python=3.8.18
conda activate FME
pip install --upgrade pip
pip install -r requirements.txt
```

### Project Architecture

```bash
.
├── temp_00_generate_info                 ### *Temp00. Integration of station and earthquake data*
│   └── P01_dataset_info.py               # perform real-time integration of station and earthquake data
├── temp_01_data_processing               ### *Temp01. Using Sac files to create a dataset of magnitude estmation*
│   ├── P01_by_Factor.py                  # correct the raw data (counts) to m/s² and save as npy files
│   └── P02_create_dataset.py             # integrate data from various earthquake events and build a dataset
├── temp_02_model_pretrain                ### *Temp02. Fine-tune the FME with samples obtained with temp_01 outputs(dataset)* 
│   ├── P01_dataset_csv.py                # generate a csv file with data root, time(moving window),label(Mag) for training model
│   └── P02_train_model.py                # training Pytorch model and auto save model parameters
└── temp_03_offline_predict               ### *Temp03. Make prediction on moving window dataset using PyTorch model*
    └── P01_predict.py                    # load dataset, make predictions, save result to csv file


.
```
### Warning
**If you intend to use this code to evaluate earthquake magnitudes, please ensure that the input data meets the following requirements**
1. All data must come from acceleration stations or must have been converted to acceleration, with units in **m/s²**.
2. The start times of all waveform data for each event must be the same, or the difference must be **less than 0.02 seconds**.
3. The original waveform data must be **longer than 20 seconds**. Otherwise, please adjust the model parameters accordingly.


### FME Input Format
two input, event sac file and catalog

**1. sac file**<br />
file root: FME/temp_01_data_process/event_data/eventID/*<br />
sac file name: {net}.{station_name}.{seismometer_type}{component}.{location}.sac<br />
sac[0].stats['sampling_rate']: Data sampling rate(set 100Hz)<br />
sac[0].data: Instrument waveform data<br />

**2. Dataset_info.csv**<br />
file root: FME/<br />
csv file name: Dataset_info.csv<br />
**Columns**<br />
eventID: Custom event name.<br />
station_name: Station name, same as sac file name.<br />
origin_time: The start time of the earthquake rupture.<br />
start_time: The start time of the waveform.<br />
channel_name: Seismometer type and component.<br />
network: The network to which the instrument belongs.<br />
sta_longitude, stasta_latitude: Station longitude and latitude.<br />
ev_longitude, ev_latitude: Event longitude and latitude.<br />
E_P_idx, N_P_idx, Z_P_idx: P arrival index(calculate) for each component.<br />
E_shift(s), N_shift(s), Z_shift(s): For each event, align the waveform data by shifting each station's waveform according to the time difference from the earliest start time.<br />
sampling_rate: seismometer sampling rate(set 100Hz).<br />
Z_npts, N_npts, E_npts: Samples number for each component.<br />
epi_dist: epicenter distance for station location.<br />
hypo_dist: hypocenter distance for station location.<br />
magnitude: Tawain local magnitude for each event.<br />

### Run template codes
In this repository, we provide four template scripts:<br />

**(1) temp_01_data_processing**<br /> 
generate some samples to dataset using NCKU dataset.<br />
```bash
cd temp_01_data_processing
python P01_by_Factor.py
python P02_create_dataset.py
```

**(2) temp_02_model_pretrain**<br />
fine-tune the trained model using few samples generated in dataset.<br />
```bash
cd temp_02_model_pretrain
python P01_dataset_csv.py
python P02_train_model.py
```

**(3) temp_03_offline_predict**<br />
Use a pretrained model to predict event data in the dataset.<br />
```bash
cd temp_03_offline_predict
python P01_predict.py
```

