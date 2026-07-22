import os

# Step 1 — Preprocessing
os.system("src/preprocessing.py")

# Step 2 — PCA + TDA anomalies
os.system("src/pca_tda_anomalies.py")

# Step 3 — GNN notebook (produces saved files)
os.system(
    "jupyter nbconvert --to notebook --execute notebooks/gnn_anomalies.ipynb "
    "--output notebooks/gnn_anomalies_output.ipynb"
)

# Step 4 — Evaluation
os.system("src/evaluate.py")

print("Pipeline completed.")
