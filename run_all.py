import subprocess

def run_python(path):
    subprocess.run(["python", path], check=True)

def run_notebook(path, output):
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", path,
        "--output", output
    ], check=True)

def main():
    print("Step 1 — Preprocessing")
    run_python("src/preprocessing.py")

    print("Step 2 — PCA + TDA anomalies")
    run_python("src/pca_tda_anomaly_scoring.py")

    print("Step 3 — GNN notebook")
    run_notebook(
        "notebooks/gnn_anomaly_scoring.ipynb",
        "notebooks/gnn_anomaly_output.ipynb"
    )

    print("Step 4 — Evaluation")
    run_python("src/evaluate.py")

    print("Pipeline completed.")

if __name__ == "__main__":
    main()
