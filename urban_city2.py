import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv('/home/badreddine/Downloads/smart-urban-analytics/urban_development_dataset.csv')
test_data = pd.read_csv('/home/badreddine/Downloads/smart-urban-analytics/urban_development_test_data.csv')

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

# Initialize Random Forest
rf_model = RandomForestClassifier(random_state=40)

# Hyperparameter GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

# model from GridSearch
best_rf_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Predict on the validation set
y_val_pred = best_rf_model.predict(X_val)

# Calculate validation accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Random Forest Validation Accuracy: {accuracy:.4f}")

# Predict on the test dataset
test_class_predictions = best_rf_model.predict(test_data_scaled)

# submission file
submission = pd.DataFrame({
    "ID": test_data.index + 1,
    "development_trend_score": test_class_predictions + 1  # Add 1 to match original class labels if needed
})
submission_file_path = '/home/badreddine/Desktop/development_trend_predictions_optimized.csv'
submission.to_csv(submission_file_path, index=False)
print(f"Submission file saved at: {submission_file_path}")

