"""
The purpose of this file is to apply anomaly detection methods to the persistent homology features of graphs, as well as 
to the raw data of the graphs. We use local outlier factor (LOF) and Mahalanobis distance as our anomaly detection methods.
We will first compute the persistent homology features of the graphs using flagser, and then apply anomaly detection to the l2 and l1 norms of the persistence diagrams.
We will save the anomalies for each method and choice of hyperparameters (e.g. number of neighbors in LOF) and each norm in a different file.
We then apply anomaly detection methods to PCA applied to the raw data of the graphs and save the results as files.  
"""
# import the packages
import itertools
from re import L
from turtle import mode
import numpy as np
import networkx as nx
import statistics
import pandas as pd
import math
import pyflagser

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor


from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.spatial.distance import mahalanobis


# compute the persistent homology of the graph using flagser and compute the l2 and l1 norms of the persistence diagrams for H0 and H1. We will use these norms as features for anomaly detection. 
# We will also threshold the persistence diagrams to remove noise and outliers, by setting a lower and upper threshold for the birth and death times of the features in the persistence diagrams
def compute_persistence_features(samples, thresh_lower, thresh_upper):
    l2_norms = []
    l1_norms = []
    for i in range(samples.shape[0]):
        print("We are now processing Graph" + str(i))
        curr_sample = samples[i]
        # remove self loops
        for i in range(curr_sample.shape[0]):
            curr_sample[i][i] = 0
        
        
        
        print(curr_sample)
        # compute the persistent homology of the graph using flagser.
        homology = pyflagser.flagser_weighted(csr_matrix(curr_sample), max_edge_weight=1, min_dimension=0, max_dimension=1, filtration='max')['dgms']
        H0 = np.asarray([ [max(thresh_lower, homology[0][i][0]), min(thresh_upper, homology[0][i][1])] for i in range(homology[0].shape[0]) ]  )
        H1 = np.asarray([ [max(thresh_lower, homology[1][i][0]), min(thresh_upper, homology[1][i][1])] for i in range(homology[1].shape[0]) ]  )
        # compute the l2 and l1 norms of the persistence diagrams for H0 and H1. We will use these norms as features for anomaly detection.
        lifespans_0 = H0[:,1] - H0[:,0]
        lifespans_1 =  H1[:, 1] - H1[:, 0] 
        l2_norm0 = np.linalg.norm(lifespans_0, ord=2)
        l2_norm1 = np.linalg.norm(lifespans_1, ord=2)   
        l1_norm0 = np.linalg.norm(lifespans_0, ord=1)
        l1_norm1 = np.linalg.norm(lifespans_1, ord=1)    
        l2_norms.append([l2_norm0, l2_norm1])
        l1_norms.append( [l1_norm0, l1_norm1])
        
    print(l2_norms)
    print(l1_norms)

    return l1_norms, l2_norms


# compute the local outlier factor of the samples and return the indices of the anomalies.
# the anomalies are defined as the samples that have a local outlier factor greater than a certain percentile
# other than number of neighbors we use the default parameters for the local outlier factor,
def get_lof_anomalies(samples, percent, neighbors):
     print("Computing LOF anomalies...")
     lof = LocalOutlierFactor(n_neighbors=neighbors, contamination='auto', novelty=False)
     y_pred = lof.fit_predict(samples)
     lof_scores = -lof.negative_outlier_factor_  
     threshold = np.percentile(lof_scores, percent)   
     outlier_indices = np.where(lof_scores >= threshold)[0]
     return outlier_indices 

# compute the Mahalanobis distance of the samples and return the indices of the anomalies.
# the anomalies are defined as the samples that have a Mahalanobis distance greater than a certain percentile
def get_mahalanobis_anomalies(samples, percent):
    print("Computing Mahalanobis anomalies...")
    # Compute the mean and covariance of the samples
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    inv_cov = np.linalg.pinv(cov)  # Use pseudo-inverse for numerical stability
    if np.linalg.det(cov) > 10e-6:  # Check if covariance matrix is invertible and use inverse if it is
        inv_cov = np.linalg.inv(cov)
    # Compute Mahalanobis distances
    distances = np.array([mahalanobis(samples[i], mean, inv_cov) for i in range(samples.shape[0])])
    threshold = np.percentile(distances, percent)
    outlier_indices = np.where(distances >= threshold)[0]
    
    return outlier_indices

# combine the anomalies from the local outlier factor and the Mahalanobis distance 
# for the local outlier factor we will use different numbers of neighbors 
def get_combined_anomalies(samples, percent):
    neighbors = [5, 10, 15, 20, 25, 30]
    lof_anomalies = []
    for i in range(len(neighbors)):
        get_lof_anomalies(samples, percent, neighbors[i])
        lof_anomalies.append(get_lof_anomalies(samples, percent, neighbors[i]))
    return np.asarray(lof_anomalies),  get_mahalanobis_anomalies(samples, percent)


def main():
    
    print("now loading the samples")
    # the array containing the graphs representing stock correlations
    samples =  np.load("corr_matrices25(US).npy", allow_pickle=False) 
    Graphs = []
    for i in range(len(samples)):
        # remove the bias of the graph by removing self loops and threshholding the graph
        samples[i][samples[i] < 0] = 0
        # remove diagonal entries to remove self loops and bias of the graph; this is done by subtracting the diagonal matrix of the graph from the graph itself
        samples[i] = samples[i] - np.eye(samples[i].shape[0]) * samples[i].diagonal() + np.eye(samples[i].shape[0])
        G = nx.from_numpy_array(samples[i])
        Graphs.append(G)
    
    

    # first we compute the persistent homology features of the graph and apply anomaly detection to l2, l2 norms of the persistence diagrams. 
    # We save the anomalies for each method and each norm in a different file.
    l1_norm, l2_norm = compute_persistence_features(samples, 0.0, 1.0)
    norms = [np.asarray(l1_norm), np.asarray(l2_norm)]
    for i in range(len(norms)):
        lof, mahalanobis = get_combined_anomalies(norms[i], 97.5)
        np.save("LOF L" + str(i+1) + "PH_anomaliesUS(25).npy", np.asarray(lof))
        np.save("Mahalanobis L" + str(i+1) + "_PH_anomaliesUS(25).npy", np.asarray(mahalanobis))
    
    # flatten the sammples array so it can be put into an ordinary classifier
    samples = samples.reshape(samples.shape[0], -1)
    # apply standard scaling to the samples
    scaler = StandardScaler()
    samples_scaled = scaler.fit_transform(samples)
    
   
    # use pca to reduce the dimensionality of the samples and then feed into anomaly detection methods
    reduced_dimensions = [10, 100]
    for d in reduced_dimensions:
        
        pca = PCA(n_components=d)
        pca_components = pca.fit_transform(samples_scaled)
        lof, mahalanobis = get_combined_anomalies(pca_components, 97.5)
        # save the anomalies for each method and each dimension in a different file
        np.save("LOF PCA(dim=" + str(d) + ")_anomaliesUS(25).npy", np.asarray(lof))
        np.save("Mahalanobis PCA(dim=" + str(d) + ")_anomaliesUS(25).npy", np.asarray(mahalanobis))
        
        
    # finally just put raw data into the anomaly detection methods
    lof, mahalanobis = get_combined_anomalies(samples_scaled, 97.5)
    np.save("LOF_raw_anomaliesUS(25).npy", np.asarray(lof))
    np.save("Mahalanobis_raw_anomaliesUS(25).npy", np.asarray(mahalanobis))




        
        
        
     
if __name__=="__main__":
    main()