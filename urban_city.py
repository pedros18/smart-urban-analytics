# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv('/home/badreddine/Downloads/smart-urban-analytics/urban_development_dataset.csv')
test_data = pd.read_csv('/home/badreddine/Downloads/smart-urban-analytics/urban_development_test_data.csv')

# Identify common columns between training and test datasets
common_columns = train_data.columns.intersection(test_data.columns)

# Select only the features present in both datasets
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

# Adjust target variable to start from 0
y_class = y - 1  # Subtracting 1 to shift class labels to start from 0

# Split the dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

# Train CatBoost model
print("Training CatBoost model...")
catboost_model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.1,
    depth=6,
    early_stopping_rounds=100,
    random_state=42,
    verbose=50
)
catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val))
y_val_pred_catboost = catboost_model.predict(X_val)
accuracy_catboost = accuracy_score(y_val, y_val_pred_catboost)
print(f"Validation Accuracy with CatBoost: {accuracy_catboost:.4f}")

# Predict on the test data
test_class_predictions = catboost_model.predict(test_data_scaled).flatten()

# Prepare the submission file
submission = pd.DataFrame({
    "ID": test_data.index + 1,  # Assuming IDs are 1-based indices
    "development_trend_score": test_class_predictions + 1  # Add 1 back to match the original labels
})

# Save the submission file
submission_file_path = '/home/badreddine/Desktop/development_trend_predictions_catboost.csv'
submission.to_csv(submission_file_path, index=False)
print(f"Submission file saved at: {submission_file_path}")

