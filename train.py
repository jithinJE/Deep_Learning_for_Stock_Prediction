# Handles training, saving of results

import lstm
import cnn_batchnorm_lstm
import multiple_branch_cnn
import time
import matplotlib.pyplot as plt
import dataload
from datetime import datetime
import os
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from keras.callbacks import EarlyStopping, TensorBoard
from keras.backend import clear_session
import pandas as pd
import gc

def plot_results_multiple(predicted_data, true_data, prediction_len, fig_path=None):
    fig, axs = plt.subplots(len(predicted_data), 1, sharex=True)
    if (len(predicted_data) > 1):
        for x in range(len(predicted_data)):
            axs[x].plot(true_data[x], label='True Data')
            # Pad the list of predictions to shift it in the graph to it's correct start
            for i, data in enumerate(predicted_data[x][0]):
                padding = [None for p in range(i * prediction_len)]
                axs[x].plot(padding + data, label='Prediction')
            axs[x].set_title(predicted_data[x][1])
    else:
        axs.plot(true_data, label='True Data')
        for i, data in enumerate(predicted_data[0][0]):
            padding = [None for p in range(i * prediction_len)]
            axs.plot(padding + data, label='Prediction')
        axs.set_title(predicted_data[0][1])
    if (fig_path is not None):
        fig.savefig(fig_path)
    else:
        plt.show()

if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 100
    seq_lens = [50]
    predict_len = 10

    # description: several-word description of the purpose of the run
    description = 'stock_single_nosmooth'
    results_fname = '{}_{}'.format(description, int(datetime.now().timestamp()))

    early_stopping = EarlyStopping(patience=20)
    tensorboard = TensorBoard(log_dir='tensorboard', write_grads=True)

    for seq_len in seq_lens:
        print('Seq len: {}'.format(seq_len))
        print('> Loading data... ')

        X_train, y_train, X_test, y_test = dataload.load_data('daily_spx.csv', seq_len, normalise_window=True, smoothing=False, smoothing_window_length=5, smoothing_polyorder=3, reshape=True)
        #X_train, y_train, X_test, y_test = dataload.load_sin_data(seq_len, normalise_window=True)

        print('> Data Loaded. Compiling...')

        # Grid search parameters
        kernel_sizes = [5,9]
        step_sizes = [2]
        single_branch = True
        stride = [3]
        lstm_units = [200,400]
        branches = [3]
        model = False
        if (single_branch):
            model = KerasRegressor(build_fn=cnn_batchnorm_lstm.build_model, validation_split = 0.20)
        else:
            model = KerasRegressor(build_fn=multiple_branch_cnn.build_model, validation_split = 0.20)
        cnn_layers = [3]
        filter_nums=[128]
        batch_size = [32]
        single_lstm = [True]
        cat_branches = [True]
        param_grid = {}
        if (single_branch):
            param_grid=dict(
                    layers=[(1, seq_len)], 
                    epochs=[epochs], 
                    cnn_layers=cnn_layers,
                    lstm_units=lstm_units,
                    kernel_size=kernel_sizes,
                    stride_1=stride,
                    filter_num=filter_nums,
                    batch_size=batch_size
                    )
        else:
            param_grid = dict(
                    layers = [(1, seq_len)],
                    epochs = [epochs],
                    ksize1 = kernel_sizes,
                    step_size = step_sizes,
                    batch_size = batch_size,
                    lstm_units = lstm_units,
                    single_lstm_layer = single_lstm,
                    num_branches = branches,
                    concat = cat_branches
                    )

        # Grid search
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_result = grid.fit(X_train, y_train, callbacks=[early_stopping])
        print('Best parameters:')
        print(grid.best_params_)

        results = pd.DataFrame(grid.cv_results_)
        results.sort_values(by='rank_test_score', inplace=True)

        print(results.to_string())

        # Build top 3 models
        idx = 0
        top_models = []
        for index, row in results.iterrows():
            top_model = False
            if (single_branch):
                top_model = cnn_batchnorm_lstm.build_model(row['param_layers'], row['param_cnn_layers'], row['param_kernel_size'])
            else:
                top_model = multiple_branch_cnn.build_model(row['param_layers'], row['param_ksize1'], row['param_step_size'], row['param_lstm_units'], row['param_single_lstm_layer'])
            top_models.append(top_model)
            top_models[-1].batch_size = row['param_batch_size']
            idx = idx + 1
            if idx > 2:
                break

        early_stopping = EarlyStopping(patience=20)
        fit_models = []
        for i in reversed(range(len(top_models))):
            model = top_models[i]
            callbacks = []
            if (i == 0):
                callbacks = [early_stopping]
            else:
                callbacks = [early_stopping]
            model.fit(
                    X_train,
                    y_train,
                    batch_size=model.batch_size,
                    nb_epoch=epochs,
                    validation_split=0.10,
                    callbacks=[early_stopping, tensorboard]
                    )
            fit_models.append(model)

        predictions = [dataload.predict_sequences_multiple(model, X_test, seq_len, predict_len)
                for model in top_models]
        scores = [model.evaluate(X_test, y_test, verbose=0)
                for model in top_models]

        # Save results
        os.makedirs(results_fname)
        folder_name = 'seq_len_{}'.format(seq_len)
        os.makedirs('{}/{}'.format(results_fname, folder_name))
        results.to_csv('{0}/{1}/results.csv'.format(results_fname, folder_name))
        top_model_plots = [(predictions[i], 'Model {}'.format(i+1)) for i in range(len(predictions))]
        plot_results_multiple(top_model_plots, y_test, predict_len, fig_path = '{0}/{1}/plots.pdf'.format(results_fname, folder_name))
        index = 1
        for model in fit_models:
            model.save('{}/{}/model-{}.h5'.format(results_fname, folder_name, index))
            index = index + 1
