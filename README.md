# secure_unl_on_ridge_regression

This repository implements an experimental framework for evaluating the accuracy and feasibility of performing machine unlearning on ridge regression models using homomorphic encryption. The framework simulates a secure unlearning protocol where a server performs Sherman-Morrison updates on encrypted data points provided by a client, without the server learning the actual values of the removed points.

The implementation serves as a benchmark to measure:
- The numerical error introduced by homomorphic encryption and polynomial approximations
- The disagreement between encrypted unlearning and exact retraining
- The accuracy degradation after removing multiple training points

## Overview

The system consists of three main components:

1. **Data Preparation**: Python script that loads datasets from OpenML, trains ridge regression models, computes the inverse Hessian, and chooses the unlearned points according to a random seed
2. **Homomorphic Unlearning**: C++ program using Microsoft SEAL that performs encrypted Sherman-Morrison updates with Remez polynomial approximation for division
3. **Analysis**: Python script that analyzes results, compares encrypted unlearning with exact retraining, and generates statistical reports

## System Requirements

### Dependencies

- **C++ Compiler**: GCC 9+
- **Microsoft SEAL**: Version 4.0 or higher
- **Python 3.8+** with packages:
  - OpenML
  - NumPy, Pandas, SciPy
  - Scikit-learn
  - Matplotlib
- **Sollya**

### Installation

```bash
# Install Microsoft SEAL
git clone https://github.com/microsoft/SEAL.git
cd SEAL
cmake -S . -B build -DSEAL_BUILD_SEAL_C=OFF
cmake --build build
sudo cmake --install build

# Install Python dependencies
pip install openml numpy pandas scipy scikit-learn matplotlib

# Install Sollya
sudo apt-get install sollya
```

## Experiment Pipeline

### Stage 1: Data Preparation

The Python script loads datasets using OpenML task splits from the Numerical Classification Benchmark (https://www.openml.org/search?type=benchmark&study_type=task&sort=tasks_included&id=337) of the paper "Why do decision trees outperform neural networks on tabular data?" (https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) and prepares all necessary data for the homomorphic protocol.

example:
```bash
python ridge_intervallo_diretto.py \
    --dataset_id 44120 \
    --n_punti 100 \
    --seed 42 \
    --max_train_size 50000 \
    --output_dir risultati_ridge_openml_dataset_44120
```

**Parameters:**

- `--dataset_id`: OpenML dataset ID (see Supported Datasets)
- `--n_punti`: Number of points to remove via unlearning
- `--seed`: Random seed for reproducibility
- `--max_train_size`: Maximum training samples (default: 50000)
- `--output_dir`: Output directory (auto-generated as "risultati_ridge_openml_dataset_<dataset_id>" if not specified)

**Supported Datasets:**

| Dataset ID | Dataset Name | Task ID | Type |
|------------|--------------|---------|------|
| 44089 | credit | 361600 | Classification |
| 44091 | wine | 361057 | Classification |
| 44120 | electricity | 361601 | Classification |
| 44121 | covertype | 361602 | Classification |
| 44122 | pol | 361603 | Classification |
| 44123 | house_16H | 361604 | Classification |
| 44125 | MagicTelescope | 361605 | Classification |
| 44126 | bank-marketing | 361606 | Classification |
| 44127 | phoneme | 361067 | Classification |
| 44128 | MiniBooNE | 361607 | Classification |
| 44129 | Higgs | 361608 | Classification |
| 44130 | eye_movements | 361609 | Classification |
| 44131 | jannis | 361071 | Classification |

In our experiments, we restrict our analysis to a subset of these datasets, namely electricity, covertype, pol, house_16H, MagicTelescope, bank-marketing, MiniBooNE, and jannis. These datasets were selected because the initial ridge regression model achieves a sufficiently high baseline accuracy, making them suitable for evaluating the impact of unlearning on model performance. 

**What the script does:**

1. Loads dataset using OpenML task splits
2. Applies quantile transformation to numerical features
3. Adds intercept column of ones
4. Performs grid search over validation set for optimal regularization parameter lambda
5. Trains ridge regression model with regularization strength $\alpha = lambda * n$
6. Computes Hessian matrix $H = 2X^TX + 2\lambda nI$ and its inverse
7. Selects random points to remove
8. Computes Remez polynomial coefficients for approximating 1/x over the interval [1-beta_max, 1-beta_min]
9. Computes retrained model weights (training without selected points)
10. Saves all data in the seed subdirectory

**Output Files:**

- `hessian_inverse.txt`: Hessian inverse in diagonal format (padded to power of two)
- `coefficienti_remez.txt`: Remez polynomial coefficients for the specified degree
- `parametri.txt`: Experiment parameters
- `X_train_scaled.txt`, `y_train.txt`: Scaled training data
- `X_test.txt`, `y_test.txt`: Test data
- `indices_*.txt`: Various index files
- `info_H.txt`: Hessian statistics and singular values

### Stage 2: Homomorphic Unlearning

Compile the C++ program and run it for each seed.

```bash
# Compile
g++ -std=c++17 -O3 -march=native \
    -I/usr/local/include/SEAL-4.0 \
    -L/usr/local/lib \
    -lseal \
    unlearning.c++ \
    -o unlearning

# Run for each seed
./unlearning <base_directory> <seed>
```
example:
```bash
./unlearning risultati_ridge_openml_dataset_44120 42
```

**Parameters:**

- `base_directory`: Directory containing the `seed_X` subdirectories
- `seed`: Seed used in data preparation


**What the program does:**

For each point to be removed:

1. Encrypts the point $x$ and label $y$
2. Computes encrypted delta $\Delta = 2\lambda n \cdot w + 2(y - w^T x) x$
3. Computes $u = H^{-1} x$ and $v = H^{-1} \Delta$ using diagonal packing
4. Computes $\beta = x^T u$ and $\alpha = x^T v$
5. Approximates $p = 1/(1-\beta)$ using the Remez polynomial
6. Computes $\text{term} = \alpha \cdot p \cdot u$
7. Decrypts and saves the update $\Delta w = \text{term}$
8. Updates $H^{-1} \leftarrow H^{-1} + p \cdot u \cdot u^T$

**Output Files (per seed directory):**

- `erroripar_grado_X.txt`: Per-point percentage and absolute errors on the update term
- `inversi_chiaro_grado_X.txt`: Inverse values 1/(1-beta) for each point (clear)
- `risultati_grado_X.txt`: Model updates and cumulative weights (encrypted vs clear)
- `nuova_hessiana_grado_X.txt`: Updated Hessian inverse after each point


### Stage 3: Analysis
Analyze results across multiple seeds and degrees.

```bash
python test_vari_benchmark.py \
    --dataset_ids <id> ... <id> \
```

example:
```bash
python test_vari_benchmark.py \
    --dataset_ids 44120 44121 44122 \
```

**Parameters:**

- `--dataset_ids`: One or more dataset IDs to analyze


**What the script does:**

1. Loads unlearning results for all seeds and degrees
2. Computes error statistics (absolute and relative errors) over the unlearning sequence
3. Generates comparative plots:
   - Absolute errors over unlearning steps
   - Percentage errors over unlearning steps
   - Weight norm comparisons
   - Inverse values histograms
4. Performs retraining comparison:
   - Trains new models on datasets with points removed
   - Compares accuracy of encrypted unlearned models vs retrained models
   - Computes disagreement between models
   - Calculates Mean Absolute Error between predictions
5. Generates tables at fixed point counts:
   - Relative errors on weights and deltas
   - Absolute errors on weights and deltas
   - Disagreement counts and percentages
   - Accuracy values

**Output Files:**

- `errori_assoluti_grado_X_dataset_Y.png`: Absolute error plots
- `errori_percentuali_grado_X_dataset_Y.png`: Percentage error plots
- `norma_w_grado_X_dataset_Y.png`: Weight norm comparison
- `errori_assoluti_tutti_gradi_dataset_Y.png`: Comparative absolute errors across degrees
- `istogramma_inversi_dataset_Y.png`: Histogram of inverse values
- `errore_relativo_w_dataset_Y.csv`: Relative error table for weights
- `errore_relativo_delta_dataset_Y.csv`: Relative error table for deltas
- `errore_assoluto_w_dataset_Y.csv`: Absolute error table for weights
- `errore_assoluto_delta_dataset_Y.csv`: Absolute error table for deltas
- `disagreement_dataset_Y.csv`: Disagreement between encrypted and retrained models
- `confronto_retraining_mae_dataset_Y.png`: MAE comparison plot
- `confronto_retraining_accuracy_dataset_Y.png`: Accuracy comparison plot
- `risultati_analisi_dataset_Y.txt`: Detailed numerical results
- `tabella_accuracy_riassuntiva.csv`: Summary table across all datasets
- `tempi_dataset_Y.csv`: Execution time per point, for every seed and degree
- `tempo_per_punto_confronto_dataset.png`: Cross-dataset comparison of mean execution time per point, with datasets ordered by increasing number of features





