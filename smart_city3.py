# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
# Load datasets# Load datasets
train_data = pd.read_csv('/home/badreddine/Downloads/smart-urban-analytics/urban_development_dataset.csv')
test_data = pd.read_csv('/home/badreddine/Downloads/smart-urban-analytics/urban_development_test_data.csv')


# Preprocessing steps
common_columns = train_data.columns.intersection(test_data.columns)
X = train_data[common_columns.drop('development_trend_score', errors='ignore')]
y = train_data['development_trend_score']
test_features = test_data[common_columns]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
test_data_imputed = imputer.transform(test_features)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_data_scaled = scaler.transform(test_data_imputed)

# Adjust target variable to start from 0 (if necessary)
y_class = y.astype(int) - 1  # Subtract 1 if your target starts from 1
# Split dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_class, test_size=0.2, random_state=40)

# Initialize LightGBM Classifier
lgb_model = lgb.LGBMClassifier(random_state=40)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [10, 20, -1],  # -1 means no limit
    'num_leaves': [31, 50, 100],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_lgb_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Predict on the validation set
y_val_pred = best_lgb_model.predict(X_val)

# Calculate validation accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f"LightGBM Validation Accuracy: {accuracy:.4f}")

# Predict on the test dataset
test_class_predictions = best_lgb_model.predict(test_data_scaled)

# Prepare the submission file
submission = pd.DataFrame({
    "ID": test_data.index + 1,
    "development_trend_score": test_class_predictions + 1  # Add 1 to match original class labels if needed
})

# Save the submission file
submission_file_path = '/home/badreddine/Desktop/development_trend_predictions_optimized.csv'
submission.to_csv(submission_file_path, index=False)
print(f"LightGBM Submission file saved at: {submission_file_path}")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
