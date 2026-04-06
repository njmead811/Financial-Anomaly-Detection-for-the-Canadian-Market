# Financial-Anomaly-Detection-for-the-Canadian-Market
This repository uses GNN and TDA based methods to detect financial stress events in the Canadian economy. It is the code for arxiv paper
https://arxiv.org/pdf/2604.02549. The pipeline is described in more detail there 

In order to run the experiments fully, complete the following steps.

1. Run STOCKDATA.py to create correlation matrices for the stock prices
2. Run GraphAnomalyDetection.ipynb and TDAPCAAnomalyDetection to obtain the anomaly scores for the graphs according to each different method.
   The former computes the anomaly scores for the neural network methods (Glocal KD (GINE) and OCGIN(E)), whereas the latter computes the anomaly scores
   for PCA/TDA methods
4. Finally run EVALUATEDETECTOR.py to evaluate and visualize the results of each anomaly detection method.
5. The filenames must be slightly changed for the US versus Canada stocks, with details on how to do this contained in the commenting for each file. 
