# Implementation of svm regressor
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from datetime import datetime
from train import plot_results_multiple
import dataload
import pickle

def main():
    description = 'svm_smoothed'
    results_fname = '{}_{}'.format(description, int(datetime.now().timestamp()))
    model_fname = "model.sav"

    seq_len = 50
    predict_len = 10

    X_train, y_train, X_test, y_test = dataload.load_data('daily_spx.csv', seq_len, normalise_window=True, smoothing=True, smoothing_window_length=5, smoothing_polyorder=3, reshape=False)

    print('> Data Loaded')

    # Grid Search

    param_grid = dict(
            C=np.logspace(-4,4),
            gamma=np.logspace(-9,3),
            kernel=['rbf','linear'],
            )

    grid = GridSearchCV(estimator=SVR(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

    start_time = datetime.now()
    grid_result = grid.fit(X_train, y_train)
    end_time = datetime.now() - start_time

    print('> Time elapsed: ', end_time)
    print('> Best parameters:')
    print(grid.best_params_)
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)

    # Build best model
    model = SVR(kernel=grid.best_params_['kernel'], C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Save Results
    os.makedirs(results_fname)
    results.to_csv('{}/results.csv'.format(results_fname))
    pickle.dump(model, open('{}/{}'.format(results_fname,model_fname), 'wb'))

if __name__=='__main__':
    main()
