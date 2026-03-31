"""
In this file we will compute the CCM correlation matrices for the stock prices of the US and Canadian stock markets. We will use the closing prices of the stocks, and we will compute the correlation matrices for each day in the period from 2005 to 2022, using a window of 25 days. 
The CCM correlation is computed using the log returns of the closing prices. The resulting correlation matrices are saved as numpy arrays for later use in machine learning models. 
"""

# Here we import necessary packages
# Of particular importance is the package for CCM correlation. 
from os.path import dirname, join as pjoin
from tkinter import Y
import scipy.io as sio
from causal_ccm.causal_ccm import ccm
import random
import numpy as np
import pandas as pd
import os
from scipy.stats import invweibull
import yfinance as yf 
from os import listdir
from os.path import isfile, join
import math

# stock tickers for the US and Canadian stock markets.
tickersCAD =  {
"ABX",	"AEM","AQN",	"ATD",	"BAM",	"BCE",	"BIP.UN", "BMO",	"BN",	"BNS",	"CAE",	"CAR.UN",	
"CCL.B",	"CCO",	"CM"	,"CNQ","CNR",	"CP"	,"CSU"	,"CTC.A"	,"CVE"	,"DOL"	,"EMA"	,"ENB"	, "FM",
"FNV", "FSV", "FTS"	,"GIB.A"	,"GIL"	,"H"	,"IFC",	"IMO","K",	"L",	"MFC"	,
"MG",	"MRU",	"NA"	,"NTR"	,"OTEX"	,"POW",	"QSR",	"RCI.B",	"RY"	,"SAP"	,"SHOP"	,
"SLF"	,"SU"	,"T"	,"TD",	"TECK.B","TOU"	,"TRI","TRP","WCN",	"WN"	,"WPM"	, "WSP"}

tickersUS = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "VZ", "BA", "CAT",
    "NFLX", "INTC", "T", "CSCO", "PFE", "PEP", "KO", "XOM", "ABT", "CVX", "AXP"
    }

# download the stock for TD as a reference for the number of business days in the period.
td_data = yf.download("TD.TO", start="2005-01-01", end="2022-01-01")
num_business_days = len(td_data.index)

# the window that we will use for the CCM correlation. We will compute the correlation matrices for each day in the period, using the previous 25 days as the window.
window = 25

time_series = []


for t in tickersCAD:
    print(t)
    # for the US stocks just use the ticker, for the Canadian stocks add the .TO suffix.
    curr_data = yf.download(t + ".TO", start="2005-01-01", end="2022-1-01")
    
    # if the data frames are empty skip this tickers
    if curr_data.empty:
        continue
    curr_close = curr_data["Close"].to_numpy()
    
    # if there is data for all the business days in the period, add the closing prices to the time series. Otherwise skip this ticker.
    if len(curr_close) >= num_business_days:
        time_series.append([curr_close[j][0] for j in range(num_business_days)])
       

# number of stocks 
num_stocks = len(time_series)
# hyperparameters for CCM correlation 
tau = 1 
E = 3 
L=window

# the array containing the correlation matrices
corr_matrices = []
print("the number of stocks is" + str(num_stocks))


for day in range(1, num_business_days-window, 1):
    
    print("We are now computing the correlations for day:" + str(day))
    # the correlation matrix for this day.
    corr_matrix = np.zeros((num_stocks, num_stocks))
    
    for i in range(num_stocks):
        # we compute the log returns. The CCM correlation is computed using the log returns of the closing prices
        window_1 = [math.log(time_series[i][day + k]/time_series[i][day + k-1]) for k in range(window) ]
        
        for j in range(num_stocks):
            # log returns for secod stock 
            window_2 = [math.log(time_series[j][day + k]/time_series[j][day + k-1]) for k in range(window) ]
            # correlation between the stock prices
            ccm_curr = ccm(window_1, window_2, tau, E, L)
            corr_matrix[i][j] = ccm_curr.causality()[0]
    corr_matrices.append(corr_matrix)
    
    print("The correlation matrices for this window are:")
    print(corr_matrix)
    
# save as a different file for the US and Canadian stocks. 
np.save('corr_matrices25(CAD).npy', corr_matrices)


