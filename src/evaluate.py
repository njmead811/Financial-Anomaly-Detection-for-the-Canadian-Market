"""
In this file we evaluate the performance of the methods we have implemented in detecting extreme events in the CFSI dataset for Canada.
We use the dates of extreme events provided by the CFSI dataset and evaluate the recall and precision of our methods in predicting these events.
We also plot the number of anomalous graphs detected by each method over time, and mark the dates of extreme events on the plot.
"""

# import the basic packages
import itertools
from re import L
from turtle import mode
import numpy as np
import networkx as nx
import statistics
import pandas as pd
import math
import yfinance as yf
import gklearn as gt



import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # or 'Qt5Agg' / 'QtAgg'
from matplotlib.dates import DateFormatter
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.spatial.distance import mahalanobis
import matplotlib.dates as mdates


from datetime import datetime

# the extreme events in the CFSI for Canada are as follows:
extreme_event_dates_CFSI = [('2007', '9'), ('2009', '1'), ('2011', '10'), ('2013', '6'), ('2015', '1'), ('2016', '2'), ('2020', '4')] 
    
# the extreme events in the DJIA for the US are as follows:
extreme_event_dates_DJIA = [('2007', '10'), ('2008', '11'), ('2010', '6'), ('2011', '10'),
                                ('2015', '9'), ('2018', '2'), ('2020', '4')]

def evaluate_prediction(extreme_event_indices, indices):
    
    num_correct_predictions = 0
    for i in range(len(indices)):
        #later_events = extreme_event_indices
        later_events = extreme_event_indices[np.where(extreme_event_indices > indices[i])[0]]
        distances = [int(abs(e-indices[i])) for e in later_events]
        min_dist = min(distances) if distances else None
        if min_dist != None and min_dist <= 50:
            # if the anomaly is within 60 days of the extreme event, we consider it a correct prediction
            num_correct_predictions += 1
    if len(indices) > 0: 
           recall = num_correct_predictions / len(indices)
    else:
        recall = 0
    num_events_predicted = 0 
    for i in range(len(extreme_event_indices)):
        #earlier_indices = indices
        earlier_indices = indices[np.where(indices < extreme_event_indices[i])[0]]
        distances = [int(abs(e-extreme_event_indices[i])) for e in earlier_indices]
        min_dist = min(distances) if distances else None
        if min_dist != None and min_dist <= 50:
            num_events_predicted += 1

    precision = num_events_predicted / len(extreme_event_indices)

    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    f1 = 0
    if precision != 0 or recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1
def evaluate_prediction_method(extreme_event_indices, values, method_name):
    values = np.asarray(values)               # coerce lists to ndarray
    if values.ndim == 1:                      # if it is a 1-D array of indices, wrap
        values = values.reshape((1, values.shape[0]))
    # now safe to iterate over rows
    results = []
    for i in range(values.shape[0]):
        print(i)
        curr_values = values[i]
        recall, precision, f1 = evaluate_prediction(extreme_event_indices, curr_values)
        results.append((recall, precision, f1, curr_values))
    # calculate the best f1 score and the corresponding recall and precision
    best_f1 = max(results, key=lambda x: x[2])
    best_recall = best_f1[0]
    best_precision = best_f1[1]
    # now we can plot the extreme graphs for the best recall and precision
    #find_extreme_graphs(method_name + " Best Precision", best_f1[3])
    return best_recall, best_precision, best_f1[2], best_f1[3]

# get the indices of the dates of extreme events for CFSI in Canada 

def get_extreme_indices(extreme_event_dates):
    # collect the indices of the dates of in the range of the extreme events; we use the stock data for BMO.TO to get the dates, since it has the same date range as the CFSI dataset for Canada.
    stock = yf.download("BA", start="2005-01-01", end="2022-01-01")
    indices = stock.index
    indices = stock.index
    dates = []
    # record the month and year of each day
    for i in range(len(indices)):
        dates.append((indices[i].strftime('%Y'), indices[i].strftime('%m')))
    extreme_event_indices = []
    # find the first index corresponding to each extreme event date; we consider the first index of the month of the extreme event as the index of the extreme event, since we are looking at monthly graphs.
    for i in range(len(extreme_event_dates)):
        year, month = extreme_event_dates[i]
        print(year)
        month = month.zfill(2)                # normalize to '01'
                            # normalize to '01'
        target = (year, month)
        print(target)
        indices = [i for i, d in enumerate(dates) if d == target]
        extreme_event_indices.append(min(indices))
    return extreme_event_indices

    
# here we apply the evaluation procedure for each of the methods we have implemented for the canadian dataset, and save the results in a csv file.
#  We also plot the number of anomalous graphs detected by each method over time, and mark the dates of extreme events on the plot.
# The US dataset can be implemented in the same way, we just need to change the extreme event dates and use the appropriate files with results for the US dataset.
def main():
    
    # get the indices of extreme events. For US use extreme_event_dates_DJIA instead of extreme_event_dates_CFSI
    extreme_event_indices = np.asarray(get_extreme_indices(extreme_event_dates_DJIA))
    # load the values for the different methods
    print(extreme_event_indices)
    
    # we load the values for the different methods; for the methods that have multiple hyperparameter, we evaluate the recall and precision for each hyperparameter and report the best one.
    
    labels_nn = ["DIF", "OCGIN", "KDGIN", "OCGTL"]
    filenames_nn = ["anomalies_gnn_dif_cad.npy", "anomalies_gnn_ocgin_cad.npy", "anomalies_gnn_kdgin_cad.npy", "anomalies_gnn_ocgtl_cad.npy" ]
    
    
    
    data_nn = []
    for f in filenames_nn:
        curr_nn_data = np.load(f)
        curr_nn_data_mod = []
        for i in range(curr_nn_data.shape[0]):
          
          curr_ind = (np.where(curr_nn_data[i] == 1)[0])
          print(curr_ind.shape)
          #if curr_ind.shape[0] == 100:
          curr_nn_data_mod.append(curr_ind)
          
          #else: 
          #   print("Error")
        data_nn.append(np.asarray(curr_nn_data_mod))
    
    
    
    

    filenames_pca = ["anomalies_pca_lof_d10_cad", "anomalies_pca_lof_d100_cad", "anomalies_pca_lof_raw_cad",  "anomalies_pca_mah_d10_cad", "anomalies_pca_mah_d100_cad", "anomalies_pca_mah_raw_cad"]
    
    data_pca = [np.load(filenames_pca[i] + ".npy") for i in range(len(filenames_pca))]
    for i in range(3, 6):
        data_pca[i] = data_pca[i].reshape(1, -1)
    data_pca = np.concatenate(data_pca)
    print(data_pca.shape)
    
    filenames_tda = ["anomalies_tda_lof_l1_cad", 
                    "anomalies_tda_mah_l1_cad", "anomalies_tda_lof_l2_cad", "anomalies_tda_mah_l2_cad"]
    data_tda_l1 = np.concatenate([np.load(filenames_tda[0] + ".npy"), np.reshape(np.load(filenames_tda[1] + ".npy"), (1, -1))])


    data_tda_l2 = np.concatenate([np.load(filenames_tda[2] + ".npy"), np.reshape(np.load(filenames_tda[3] + ".npy"), (1, -1))])

    
    data = [data_pca, data_tda_l1, data_tda_l2] + data_nn
    
    labels = ["PCA", "TDA_L1", "TDA_L2"] + labels_nn
   
    results = []

    rows = []
    for i in range(len(data)):
        recall, precision, f1, row = evaluate_prediction_method(extreme_event_indices, data[i], labels[i])
        print(labels[i] + " Recall: " + str(recall))
        print(labels[i] + " Precision: " + str(precision))
        print(labels[i] + " F1 Score: " + str(f1))
        results.append([labels[i], round(recall, 2), round(precision, 2), round(f1, 2)])
        rows.append(row[0:99])
    # create a dataframe to store the results
    df = pd.DataFrame(results, columns=["Method", "Recall", "Precision", "F1 Score"])
    # save the dataframe to a csv file
    df.to_csv("results-final(CAD).csv", index=False)
    np.save("Anomalies.npy", np.vstack(rows))
    



if __name__=="__main__":
    main()
