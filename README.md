# 🧠 SVM Breast Cancer Classification

This project applies **Support Vector Machines (SVM)** to classify tumors as **Malignant (M)** or **Benign (B)** using the **Breast Cancer Wisconsin Diagnostic Dataset**.

## 📂 Files Included

| File | Description |
|------|-------------|
| `svm_breast_cancer.py` | Python script that loads data, trains SVMs, and visualizes results |
| `svm_breast_cancer.ipynb` | Jupyter notebook for interactive exploration |
| `breast-cancer.csv` | Cleaned dataset |
| `svm_linear_pca.png` | Decision boundary using SVM with **Linear Kernel** |
| `svm_rbf_pca.png` | Decision boundary using SVM with **RBF Kernel** |
| `README.md` | Project overview and usage instructions |

---

## 🧪 Project Overview

- Binary classification: `M` = malignant, `B` = benign (converted to 1/0)
- Scikit-learn SVM models used:
  - `Linear Kernel`
  - `RBF Kernel` (Radial Basis Function)
- Feature scaling with `StandardScaler`
- Visualization of decision boundaries using **PCA-reduced 2D** data
- Hyperparameter tuning using **GridSearchCV**

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/svm-breast-cancer.git
   cd svm-breast-cancer
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Or manually:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

3. **Run the Python script**

   ```bash
   python svm_breast_cancer.py
   ```

   Or open the notebook:

   ```bash
   jupyter notebook svm_breast_cancer.ipynb
   ```

---

## 📊 Model Evaluation

Hyperparameter tuning done via **GridSearchCV** with cross-validation:

* `C`: \[0.1, 1, 10]
* `gamma`: \['scale', 0.01, 0.1, 1]

Evaluation includes:

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix (if extended)

---

## 📌 Key Concepts Covered

* Support Vectors
* Linear vs Non-linear Kernels
* PCA for dimensionality reduction
* Regularization (`C`)
* Kernel trick (`gamma`)

---

## ✅ Output Visualizations

![Image](https://github.com/user-attachments/assets/1ef05814-ecad-4a9c-8408-f19d7bd8e585)

![Image](https://github.com/user-attachments/assets/7c5670d6-1c67-4597-8ff1-8ec289f9fc2b)

These plots show how each kernel classifies the data in 2D PCA space.

---


## ✍️ Author

RANGESHPANDIAN PT

