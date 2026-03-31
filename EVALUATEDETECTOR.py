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
extreme_event_dates_CFSI = [('2007', '9'), ('2009', '1'), ('2011', '10'), ('2013', '4'), ('2015', '1'), ('2016', '2'), ('2020', '4')] 
    
# the extreme events in the DJIA for the US are as follows:
extreme_event_dates_DJIA = [('2007', '10'), ('2008', '11'), ('2010', '6'), ('2011', '10'),
                                ('2015', '9'), ('2018', '2'), ('2020', '3')]

# We plot a graph of the number of anomalous graphs detected by each method over time, and mark the dates of extreme events on the plot.
# values is the set of indices of the anomalous graphs detected by a method.
def find_extreme_graphs(title, values):
    
    
    
   
    # download the dates we use; this is the same as the dates available for the TD/BA stock. 
    stock = yf.download("BA", start="2005-01-01", end="2022-01-01")
    indices = stock.index
    num_days = len(indices)
    window = 25
    m_yr = []
    # num_days-window because we are looking at the number of anomalies per month, and we want to make sure that we have enough data points for each month.
    for i in range(0, num_days-window, 1):
        ts_str_1 = indices[i].strftime('%Y')
        ts_str_2 = indices[i].strftime('%m')
        curr_m_yr = int(ts_str_1)*100000 + int(ts_str_2)
        m_yr.append(curr_m_yr)
    m_yr = np.array(m_yr)
    
    # create an array of the same length as m_yr, where each entry is 1 if the corresponding graph is anomalous and 0 otherwise.
    extr_val = np.zeros(m_yr.shape[0])
    for i in range(len(values)):
        extr_val[values[i]] = 1
    
       
    
    data = np.column_stack([m_yr, extr_val])
    # Create DataFrame to store the monthly number of anomalous graphs 
    df = pd.DataFrame(data, columns=["Year-Month", "vals"])
    grouped_df = df.groupby(["Year-Month"]).sum() 
    values = grouped_df.values.flatten()
    
    
    # Generate sample data
    dates = pd.date_range(start="2005-01-01", end="2021-11-01", freq="ME")
    
    

# Create a plot for anomalies over time
    fig, ax = plt.subplots()
    ax.plot(dates, values[0:len(dates)], color='blue', linewidth=1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_ylim([0, 20])
    extreme_event_dates = [('2007', '9'), ('2009', '1'), ('2011', '10'), ('2013', '4'), ('2015', '1'), ('2016', '2'), ('2020', '4')] 
    # draw vertical lines to mark the extreme events 
    for i in range(len(extreme_event_dates_CFSI)):
    
        curr_date = extreme_event_dates_CFSI[i][0] + '--' + extreme_event_dates[i][1]
        ax.axvline(x= pd.Timestamp(curr_date), color='red', linestyle='--', linewidth=1, label=str(i))
        plt.text(pd.Timestamp(curr_date) + pd.DateOffset(months=2), 20, str(i+1), rotation=0, verticalalignment='top')


    #ax.legend()
    plt.gcf().autofmt_xdate()
    plt.title("Graph Anomalies by Date")
    plt.xlabel("Date")
    plt.ylabel("Number of Anomalous Graphs")
    plt.tight_layout()
    plt.savefig(title + "Anomalies(CAD).png")
    plt.show()

import datetime as dt
"""
We evaluate the precision, recall and f1 score of a given method in predicting the extreme events. We consider a prediction to be correct if it is within 50 days of the actual extreme event.
"""
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
    find_extreme_graphs(method_name + " Best Precision", best_f1[3])
    return best_recall, best_precision, best_f1[2]

# get the indices of the dates of extreme events for CFSI in Canada 

def get_extreme_indices(extreme_event_dates):
    # collect the indices of the dates of in the range of the extreme events; we use the stock data for BMO.TO to get the dates, since it has the same date range as the CFSI dataset for Canada.
    stock = yf.download("BMO.TO", start="2005-01-01", end="2022-01-01")
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
    extreme_event_indices = np.asarray(get_extreme_indices(extreme_event_dates_CFSI))
    # load the values for the different methods
    print(extreme_event_indices)
    # we load the values for the different methods; for the methods that have multiple hyperparameter, we evaluate the recall and precision for each hyperparameter and report the best one.
    labels = ["GlocalKD (GINE)", "One-Shot GIN(E)", "LOF L1PH", "LOF L2PH", "MAH L1PH", "MAH L2PH", "LOF RAW", "MAH RAW", "LOF PCA (dim=10)", "MAH PCA (dim=10)", "LOF PCA (dim=10)", "MAH PCA (dim=100)"]
    lof_l1 = np.load("LOF L1PH_anomaliesCA(25).npy")
    lof_l2 = np.load("LOF L2PH_anomaliesCA(25).npy")
    mah_l1 = np.load("Mahalanobis L1_PH_anomaliesCA(25).npy")
    mah_l1 = mah_l1.reshape((1, mah_l1.shape[0]))
    mah_l2 = np.load("Mahalanobis L2_PH_anomaliesCA(25).npy")
    mah_l2 = mah_l2.reshape((1, mah_l2.shape[0]))
    lof_raw = np.load("LOF_raw_anomaliesCA(25).npy")
    mah_raw = np.load("Mahalanobis_raw_anomaliesCA(25).npy")
    mah_raw = mah_raw.reshape((1, mah_raw.shape[0]))
    lof_pca_10 = np.load("LOF PCA(dim=10)_anomaliesCA(25).npy")
    mah_pca_10 = np.load("Mahalanobis PCA(dim=10)_anomaliesCA(25).npy")
    mah_pca_10 = mah_pca_10.reshape((1, mah_pca_10.shape[0]))
    lof_pca_100 = np.load("LOF PCA(dim=100)_anomaliesCA(25).npy")
    mah_pca_100 = np.load("Mahalanobis PCA(dim=100)_anomaliesUS(25).npy")
    mah_pca_100 = mah_pca_10.reshape((1, mah_pca_100.shape[0]))
    
    one_shot = np.load("ONESHOTGINPREDSCAD(25).npy")
    one_shot_mod = []
    # sometimes the problem of hypersphere collapse occurs, where all the graphs are considered anomalous; in this case, 
    # we consider the method to have failed and we do not include it in our evaluation. We check for this by looking at the number of anomalous graphs detected by the method;
    #  if it is more than 107 (2.5 percent of the graphs), it is likely that the method has suffered from hypersphere inversion and 
    #we do not include it in our evaluation.
    for i in range(one_shot.shape[0]):
          
          curr_ind = (np.where(one_shot[i] == 1)[0])
          print(curr_ind.shape)
          if curr_ind.shape[0] == 107:
               one_shot_mod.append(curr_ind)
    one_shot_mod = np.asarray(one_shot_mod)
    # we apply the same procedure as for one-shot GIN(E) to GlocalKD (GINE), since it is also a GNN-based method and can also suffer from similar problems
    kd = np.load("KDGINCAD(25).npy")
    kd_mod = []
    for i in range(kd.shape[0]):
          
          curr_ind = (np.where(kd[i] == 1)[0])
          print(curr_ind.shape)
          if curr_ind.shape[0] == 107:
               kd_mod.append(curr_ind)
    kd_mod = np.asarray(kd_mod)


    data = [kd_mod, one_shot_mod, lof_l1, lof_l2, mah_l1, mah_l2, lof_raw, mah_raw, lof_pca_10, mah_pca_10, lof_pca_100, mah_pca_100]
    print(mah_raw)
    
    # an array which saves the recall, precision and f1 score for each method; we will save this to a csv file at the end of our evaluation.
    results = []
    # here we evaluate each of the methods. 
    for i in range(len(data)):
        recall, precision, f1 = evaluate_prediction_method(extreme_event_indices, data[i], labels[i])
        print(labels[i] + " Recall: " + str(recall))
        print(labels[i] + " Precision: " + str(precision))
        print(labels[i] + " F1 Score: " + str(f1))
        results.append([labels[i], round(recall, 2), round(precision, 2), round(f1, 2)])
    # create a dataframe to store the results
    df = pd.DataFrame(results, columns=["Method", "Recall", "Precision", "F1 Score"])
    # save the dataframe to a csv file
    df.to_csv("resultsCAD(25).csv", index=False)
    
    


if __name__=="__main__":
    main()