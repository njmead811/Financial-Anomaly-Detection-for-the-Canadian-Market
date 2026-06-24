

### Business Motivation

Periods of extreme market stress, such as the 1987 crash, the 2008 global financial crisis, and the COVID‑19 collapse, create significant risk for investors and institutions. Detecting early warning signs of these events is critical for risk management, portfolio hedging, and regulatory oversight.

A recurring insight in financial research is that market instability often appears first as anomalies in the correlation structure between assets. When relationships between stocks shift abruptly, it can signal emerging systemic stress.

This project builds a machine‑learning system to detect these structural anomalies in real time.

### Methodology 




### How to Reproduce the Experiments 

1. Run STOCKDATA.py to create correlation matrices for the stock prices
2. Run GraphAnomalyDetection.ipynb and TDAPCAAnomalyDetection to obtain the anomaly scores for the graphs according to each different method.
   The former computes the anomaly scores for the neural network methods (Glocal KD (GINE) and OCGIN(E)), whereas the latter computes the anomaly scores
   for PCA/TDA methods
4. Finally run EVALUATEDETECTOR.py to evaluate and visualize the results of each anomaly detection method.
5. The filenames must be slightly changed for the US versus Canada stocks, with details on how to do this contained in the commenting for each file. 
