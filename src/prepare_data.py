"""
    Script to prepare data from raw
"""
import pandas as pd
import config as cfg
import os, sys
from tqdm import tqdm

def count_loans():
    """ Return a dict with loan counts by ID """
    counts_dict = {}   
    data_files = os.listdir(cfg.DATA_RAW)
    data_files.remove('.gitkeep')
    print('\n...Computing counts...')
    for idx, file in enumerate(data_files):
        
        print(f'Current file: {file} {idx+1}/{len(data_files)}')
        file_path = os.path.join(cfg.DATA_RAW, file)
        with open(file_path, 'r') as f:
            for line in tqdm(f):
                line_splitted = line.split('|')
                loan_id, date = line_splitted[1], line_splitted[51]

                if loan_id in counts_dict.keys():
                    counts_dict[loan_id]['count'] += 1
                else:
                    counts_dict[loan_id] = {
                        'foreclosure_status': False,
                        'count': 1,
                    }
                if date != '':
                    counts_dict[loan_id]['foreclosure_status'] = True
        
    return counts_dict

def get_loans():
    """ Get individual loans """

    print('\n...Collecting loans...')

    data = pd.DataFrame()
    data_files = os.listdir(cfg.DATA_RAW)
    data_files.remove('.gitkeep')
    
    for idx, file in enumerate(data_files):
        print(f'Current file: {file} {idx+1}/{len(data_files)}')
        file_path = os.path.join(cfg.DATA_RAW, file)
        for chunk in tqdm(pd.read_csv(file_path, sep='|', header=None, names=cfg.HEADERS, usecols=cfg.SELECT, chunksize=cfg.CHUNKSIZE)):
            data = data.append(chunk.drop('Foreclosure_Date', axis=1))
            data = data.drop_duplicates()
            
        print(file, data.shape[0])

    return data

def get_value(counts_dict, loan_id, key):
    return counts_dict[str(loan_id)][key]

def make_dataset():
    """ Make dataframe with loans with counts and foreclosure_status """
    os.chdir(sys.path[0])

    counts = count_loans()
    loans = get_loans()
    print('\n...Adding counts...')
    loans['count'] = loans['Loan_ID'].apply(lambda loan_id: get_value(counts, loan_id, 'count'))
    loans['foreclosure'] = loans['Loan_ID'].apply(lambda loan_id: get_value(counts, loan_id, 'foreclosure_status'))

    # store dataframe in interim data folder
    loans.to_csv(cfg.DATA_INTERIM, index=False)
    print('Final dataset:', loans.shape)

if __name__ == '__main__':
    make_dataset()