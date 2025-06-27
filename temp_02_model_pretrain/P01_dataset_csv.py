"""Write Dataset to CSV"""
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
####################################
# Dataset Fold and Catalog
FOLD_NAME = '../dataset'
CSV_NAME = '../dataset.csv'
Catalog = '../Dataset_info.csv'
####################################
def create_dataset_csv(event_list, catalog, output_csv):
    """process event data，store to CSV"""
    all_data = []  # save all event data
    clog = pd.read_csv(catalog)
    for event_path in tqdm(event_list, desc="Processing events"):
        eventid = event_path.split('/')[-1][:-4]
        elog = clog[clog['eventID']==eventid]
        mag = pd.unique(elog['magnitude'])

        # read npy file
        data = np.load(event_path)

        # transform to (eventID, index, target) format
        all_data.extend([(eventid, j, float(mag)) for j, (_) in enumerate(data)])

    # transform to DataFrame and write to CSV
    df = pd.DataFrame(all_data, columns=['eventID', 'time', 'Mag'])
    df.to_csv(output_csv, index=False)

datadir = np.sort(glob(f'{FOLD_NAME}/**.npy'))
print('event count:',len(datadir))
create_dataset_csv(datadir,Catalog,CSV_NAME)
