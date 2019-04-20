import datetime
import dataload
from train import plot_results_multiple
import os
import pickle
import numpy as np

def main():    
    svm_model_path = 'svm_smoothed_1542978748/model.sav'
    results_fname = 'test_svm_smooth_byyear_{}'.format(int(datetime.datetime.now().timestamp()))
    #dataset_path = 'daily_spx.csv'
    dataset_path = '../2018_data/Yahoo_2000_to_2018.csv'
    
    seq_len = 50
    predict_len = 7
    os.makedirs(results_fname)

    # Define date ranges
    
    date_ranges = [(datetime.date(2016,1,1),datetime.date(2016,6,1)),
                    (datetime.date(2017,1,1),datetime.date(2017,6,1)),
                    (datetime.date(2018,1,1),datetime.date(2018,6,1))]
    '''
    date_ranges = [(datetime.date(2016,1,1),datetime.date(2016,6,1)),
                    (datetime.date(2017,1,1),datetime.date(2017,6,1))]
    
    '''
    
    
    # Load data
    model = pickle.load(open(svm_model_path, 'rb'))
    test_data = [dataload.load_data(dataset_path, seq_len, normalise_window=True, smoothing=False, date_range=date_range, train=False) for date_range in date_ranges]

    # Generate predictions
    #[[print(seq.shape) for seq in test_date_range[0]] for test_date_range in test_data]
    #predictions = [[np.asscalar(model.predict(seq.transpose())) for seq in test_date_range[0]] for test_date_range in test_data]
    predictions = [dataload.predict_sequences_multiple(model, test[0], seq_len, predict_len) for test in test_data]

    for prediction_index in range(len(predictions)):
        for sequence_index in range(len(predictions[prediction_index])):
            predictions[prediction_index][sequence_index] = dataload.denormalize_sequence(test_data[prediction_index][2][sequence_index*7], predictions[prediction_index][sequence_index])

    for test_data_index in range(len(test_data)):
        for y_index in range(len(test_data[test_data_index][1])):
            test_data[test_data_index][1][y_index] = dataload.denormalize_point(test_data[test_data_index][2][y_index], test_data[test_data_index][1][y_index])


    # Save plot
    model_plot = [(predictions[0], '2016'), (predictions[1], '2017'), (predictions[2], '2018')]
    #model_plot = [(predictions[0], '2016'), (predictions[1], '2017')]
    plot_results_multiple(model_plot, [t[1] for t in test_data], predict_len, fig_path = results_fname + '/plots.pdf')

if __name__=='__main__':
    main()
