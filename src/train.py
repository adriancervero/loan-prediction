

from numpy import pi
import pandas as pd
from sklearn import pipeline
import config as cfg
import model_selector
import os, sys
import argparse
from collections import Counter


#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
import pickle

def create_pipeline(model, scaler=StandardScaler(), encoder=OneHotEncoder(handle_unknown='ignore')):
    """ 
        Return a preprocessing pipeline that scale numerical variables
        and encode categorical ones .        
    """
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler),     
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, cfg.NUMERICAL),
        ('cat', cat_pipeline, cfg.CATEGORICAL),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        #('over', SMOTE(sampling_strategy=0.2)),
        ('under', RandomUnderSampler(sampling_strategy=.9)), 
        ('model', model),
    ])

    return pipeline

def load_data():
    """ Load dataframe from csv """
    os.chdir(sys.path[0])
    data = pd.read_csv(cfg.TRAIN_PATH)
    return data

def prepare_data(df):
    
    X = df[cfg.NUMERICAL+cfg.CATEGORICAL].copy()
    y = df[cfg.TARGET].values    

    return X, y

def train(m, X, y):
    pipeline = create_pipeline(m)

    #X_resample, y_resample = pipeline.fit_resample(X, y)
    #print(Counter(y_resample))


    scores = cross_val_score(pipeline, X, y, cv=3, scoring='f1', verbose=2)
    
    print(f'Scores: {scores}')
    print(f'Mean: {scores.mean()}')

def hypertuning(model, X, y):
    m = model_selector.models[model]
    param_grid = model_selector.param_grid[model]
    pipeline = create_pipeline(m)
    grid_search = GridSearchCV(pipeline,
                                param_grid,
                                cv=3,
                                scoring='f1',
                                verbose=2)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    print('Best score:', best_score)
    print('Best params:', grid_search.best_params_)

    model_path = f'../models/{model}_({best_score:.2f}).pkl'
    pickle.dump(best_model, open(model_path, 'wb'))
    print('\nmodel stored: '+model_path)


def run(model, tuning):
    m = model_selector.models[model]
    df = load_data()
    X, y = prepare_data(df)
    if tuning == False:
        train(m, X, y)
    else:
        hypertuning(model, X, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str) # model to use
    parser.add_argument('-t', '--tuning', default=False, action='store_true')

    args = parser.parse_args()

    run(model=args.model,
        tuning=args.tuning)


