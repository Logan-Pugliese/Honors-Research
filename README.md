
# Non-negative Matrix Factorization (NMF) Research

## Overview

This repository contains Python implementations of various Non-negative Matrix Factorization (NMF) algorithms and their applications. The code is structured to analyze and compare different initialization techniques and optimization algorithms for NMF, including:

- **Multiplicative Updates (MU)**
- **HALS (Hierarchical Alternating Least Squares)**
- **Coordinate Descent (CD)**
- **Block Coordinate Descent (BCD)**
- **Scikit-Learn’s NMF Implementations**

Additionally, experiments have been conducted on real-world datasets, such as the **Olivetti Faces dataset**, to analyze the effectiveness of different methods in feature extraction and image reconstruction.

## Features

- Custom implementations of various NMF optimization techniques.
- Comparison of different initialization methods:
  - Random initialization
  - NNDSVD (Non-negative Double Singular Value Decomposition)
  - NNDSVDa and NNDSVDar (Modified NNDSVD Variants)
- Evaluation metrics using **Frobenius norm** to measure convergence and reconstruction error.
- Visualizations of factorized matrices, feature components, and image reconstructions.
- Training and testing performance analysis with different hyperparameters.

## Repository Structure


├── NMF_Research_Code_1.py   # Core NMF algorithms and initialization strategies

├── NMF_Research_Code_2.py   # Implementation on Olivetti Faces dataset with visualizations

├── NMF_Research_Code_3.py   # Hyperparameter tuning and evaluation of different n_components

├── README.md                # Project documentation


## Dependencies

Ensure you have the following Python libraries installed:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Use

1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/NMF_Research.git
   cd NMF_Research
   ```
2. Run `NMF_Research_Code_1.py` to execute the core NMF algorithms and analyze different initialization strategies.
3. Run `NMF_Research_Code_2.py` for face recognition experiments using the Olivetti dataset.
4. Run `NMF_Research_Code_3.py` for hyperparameter tuning and model evaluation.

## Results & Findings

- **Initialization Matters:** NNDSVD-based initializations converge faster and provide lower reconstruction errors compared to random initialization.
- **Multiplicative Updates vs. HALS:** HALS generally converges faster than the traditional MU method.
- **Effectiveness in Feature Extraction:** NMF can extract interpretable facial components in the Olivetti dataset.
- **Trade-offs in Component Selection:** The choice of `n_components` significantly impacts reconstruction quality.
- **Hyperparameter Optimization:** The selection of `n_components`, solver type, and initialization method plays a crucial role in achieving optimal results.

## Future Improvements

- Implementation of more advanced optimization techniques (e.g., projected gradient methods, deep NMF models).
- Exploring NMF applications beyond image processing, such as topic modeling in text data.
- Integration with deep learning frameworks for hybrid models.

## Contributors

- **Logan Pugliese** - [loganpugliese23@gmail.com](mailto:loganpugliese23@gmail.com)

---
Feel free to modify this README to fit your GitHub page structure!
```
