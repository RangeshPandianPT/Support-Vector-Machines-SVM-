# SVM Classification on Breast Cancer Dataset

## Description
This project applies Support Vector Machines (SVM) to classify tumors as malignant or benign using the Breast Cancer dataset.

## Contents
- `svm_breast_cancer.py`: Python script
- `svm_breast_cancer.ipynb`: Jupyter notebook version (optional)
- `breast-cancer.csv`: Dataset
- `svm_linear_pca.png`: Decision boundary (Linear Kernel)
- `svm_rbf_pca.png`: Decision boundary (RBF Kernel)

## Instructions
1. Install dependencies: `scikit-learn`, `pandas`, `matplotlib`, `numpy`.
2. Run the Python script or explore interactively using the Jupyter notebook.
3. Visualizations show SVM decision boundaries using PCA-reduced 2D features.

## Model Evaluation
Grid Search with Cross-Validation is used to tune `C` and `gamma` parameters.
