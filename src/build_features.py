

import pandas as pd
import numpy as np
import config as cfg
import os, sys
from sklearn.model_selection import StratifiedKFold

def load_data():
    """ Load dataframe from csv """
    os.chdir(sys.path[0])
    data = pd.read_csv(cfg.DATA_INTERIM)
    return data

def process_date(date_str, key):
    """ Parse date string and return month/year value according to key parameter """
    date_str = date_str[:-2] # remove '.0'
    if key == 'month':
        month = date_str[:-4]
        return month
    elif key == 'year':
        year = date_str[-4:]
        return year
    else:
        raise ValueError("invalid key, only accepts 'month' or 'year'")

def build_features(df):
    # dropping columns with many nan
    df = df.drop(cfg.DROP_COLS, axis=1)

    df = df[df['count'] > 4]
    # new features from date columns
    df['orig_month'] = df['Origination_Date'].astype(str).apply(lambda date: process_date(date, key='month'))
    df['orig_year'] = df['Origination_Date'].astype(str).apply(lambda date: process_date(date, key='year'))

    df['first_payment_month'] = df['First_Payment_Date'].astype(str).apply(lambda date: process_date(date, key='month'))
    df['first_payment_year'] = df['First_Payment_Date'].astype(str).apply(lambda date: process_date(date, key='year'))

    return df

def split_data(df):
    skf = StratifiedKFold(n_splits=3)
    t = df['foreclosure']
    for train_index, test_index in skf.split(np.zeros(len(t)), t):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

    return train, test

def store_data(train, test):
    train.to_csv(cfg.TRAIN_PATH, index=False)
    test.to_csv(cfg.TEST_PATH, index=False)

    return cfg.TRAIN_PATH+'\n'+cfg.TEST_PATH

if __name__ == '__main__':
    df = load_data()
    df = build_features(df)
    train, test = split_data(df)
    print(store_data(train, test))
