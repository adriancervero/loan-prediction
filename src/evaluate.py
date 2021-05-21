import pandas as pd
import pickle
import os, sys
import config as cfg
import argparse
#from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

def load_test_data():
    os.chdir(sys.path[0])
    data = pd.read_csv(cfg.TEST_PATH)
    return data

def prepare_data(df):
    
    X = df[cfg.NUMERICAL+cfg.CATEGORICAL].copy()
    y = df[cfg.TARGET].values

    #undersample = RandomUnderSampler(sampling_strategy=0.9)
    #X_under, y_under = undersample.fit_resample(X, y)

    return X, y

def get_scores(y_test, y_pred):
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f'f1-score: {f1:.2f}, precision: {precision:.2f}')
    print(confusion_matrix(y_test, y_pred))

def evaluate(model_name):
    """ Evaluate trained model on test """

    test = load_test_data()
    model = pickle.load(open('../models/'+model_name, 'rb'))

    X_test, y_test = prepare_data(test)
    y_pred = model.predict(X_test)

    get_scores(y_test, y_pred)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='logreg_(0.77).pkl')

    args = parser.parse_args()

    evaluate(model_name=args.model)