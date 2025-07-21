**Fraud Detection with Graph Neural Networks and Reinforcement Learning**

Overview

**Please, read sourse article about this code https://ieeexplore.ieee.org/document/10892045. I am not the author of this code and approach, the code is based on the article by Yiwen Cui; Xu Han; Jiaying Chen; Xinguang Zhang; Jingyun Yang; Xuguang Zhang**

**There are bugs in it for now**

This project implements a federated learning framework for fraud detection using Graph Neural Networks (GNNs) combined with Deep Q-Learning (DQN). The system leverages TSSGC (Temporal-Spatial-Semantic Graph Convolution) for modeling transaction graphs and DQN for optimizing detection thresholds. It processes financial fraud datasets, applies SMOTE for class imbalance, constructs k-NN graphs, and evaluates performance using metrics like AUC-ROC, AUC-PR, F1-score, and adversarial robustness.
The implementation is designed to handle large-scale financial datasets, with support for GPU acceleration and federated learning across multiple clients. It includes preprocessing, model training, and evaluation with cross-validation, along with visualizations of performance metrics.

Features

* Federated Learning: Distributes training across multiple clients with model aggregation on a central server.
* Graph Neural Networks: Uses a TSSGC model to capture spatial, temporal, and semantic features of transaction graphs.
* Reinforcement Learning: Employs DQN to dynamically select optimal classification thresholds.
* Data Preprocessing: Handles missing values, normalizes features, applies SMOTE for class balancing, and constructs k-NN graphs.
* Evaluation: Computes AUC-ROC, AUC-PR, F1-score, Recall@k%, and adversarial robustness metrics.
* Visualization: Generates plots of validation metrics over epochs.
* GPU Support: Utilizes CUDA for accelerated training when available.
* Caching: Stores preprocessed data to reduce computation time.

Requirements
To run this project, install the following Python packages:

pip install torch torch-geometric pandas numpy scikit-learn imblearn matplotlib kagglehub pynvml tqdm

Optional:

* pynvml: For GPU memory monitoring (install via pip install pynvml).
* CUDA: For GPU acceleration (ensure compatible NVIDIA drivers and CUDA toolkit are installed).

Datasets
The code supports two Kaggle datasets:

* Credit Card Fraud Detection 2023 (nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
* Financial Fraud Detection (sriharshaeedala/financial-fraud-detection-dataset)

Datasets are downloaded automatically using kagglehub or can be provided locally via the local_paths configuration.

Usage

Configuration: Modify the config dictionary in the main() function to adjust hyperparameters, such as:

* seed: Random seed for reproducibility.
* num_clients: Number of federated learning clients.
* num_epochs: Number of training epochs.
* max_rows: Maximum dataset rows to process.
* n_neighbors: Number of neighbors for k-NN graph construction.
* cache_dir: Directory for caching preprocessed data.

Running the Code:
python fraud_detection.py

The script will:

* Check for GPU availability.
* Download and preprocess datasets.
* Train the model using federated learning with cross-validation.
* Save performance metrics and visualizations.

Output:

* Model Checkpoints: Saved as <dataset_name>_fold<fold_number>_best_model.pth.
* Plots: Validation metrics saved as <dataset_name>_metrics.png.
* Console Output: Displays progress, losses, and evaluation metrics for each fold and epoch.

Code Structure

* check_gpu(): Detects and reports GPU availability and memory.
* set_seed(): Ensures reproducibility across random operations.
* TSSGC: GNN model combining spatial, temporal, and semantic features.
* DQN: Reinforcement learning model for threshold optimization.
* FraudGNNRL: Integrates TSSGC and DQN for training and prediction.
* FederatedServer: Aggregates client models in federated learning.
* EarlyStopping: Implements early stopping based on validation F1-score.
* load_and_preprocess_data(): Handles dataset loading, preprocessing, SMOTE, and k-NN graph construction.
* balance_client_data(): Balances and distributes data across clients.
* main(): Orchestrates dataset processing, training, and evaluation.

Key Hyperparameters

* hidden_dim: 128 (GNN hidden layer size).
* tssgc_lr: 0.001 (TSSGC learning rate).
* dqn_lr: 0.001 (DQN learning rate).
* num_thresholds: 10 (number of classification thresholds).
* gamma: 0.99 (DQN discount factor).
* epsilon_start: 1.0, epsilon_decay: 0.999, epsilon_min: 0.01 (DQN exploration parameters).
* early_stopping_patience: 5, early_stopping_delta: 0.005 (early stopping criteria).

Evaluation Metrics

* AUC-ROC: Area under the Receiver Operating Characteristic curve.
* AUC-PR: Area under the Precision-Recall curve.
* F1-Score: Harmonic mean of precision and recall.
* Recall@k%: Recall at top k% of ranked predictions.
* Adversarial F1: F1-score under adversarial noise.

Notes

* The code assumes a balanced class distribution after SMOTE. Adjust max_rows for larger datasets.
* GPU memory usage is monitored if pynvml is installed.
* Preprocessed data is cached in the cache directory to speed up subsequent runs.
* The code includes error handling for robustness but may require tuning for specific datasets.

License
This project is licensed under the MIT License.
