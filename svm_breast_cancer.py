import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess
df = pd.read_csv('breast-cancer.csv')
df.drop(columns=['id'], inplace=True, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Train SVM models
svm_linear = SVC(kernel='linear', C=1.0).fit(X_train_pca, y_train)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train_pca, y_train)

# Grid Search
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_scaled, y)
print("Best parameters:", grid.best_params_)

# Evaluate
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
