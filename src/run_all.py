import os

# Step 1 — Preprocessing
os.system("src/preprocessing.py")

# Step 2 — PCA + TDA anomalies
os.system("src/pca_tda_anomaly_scoring.py")

# Step 3 — GNN notebook (produces saved files)
os.system(
    "jupyter nbconvert --to notebook --execute notebooks/gnn_anomaly_scoring.ipynb "
    "--output notebooks/gnn_anomaly_output.ipynb"
)

# Step 4 — Evaluation
os.system("src/evaluate.py")

print("Pipeline completed.")
