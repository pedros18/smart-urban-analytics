import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Load datasets
train_data = pd.read_csv('/kaggle/input/smart-urban-analytics/urban_development_dataset.csv')
test_data = pd.read_csv('/kaggle/input/smart-urban-analytics/urban_development_test_data.csv')


def replace_negatives_and_missing_values(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        mean_value = df[col].median()  
        df[col] = df[col].apply(lambda val: mean_value if val < 0 or pd.isnull(val) else val)
    return df

train_data = replace_negatives_and_missing_values(train_data)
test_data = replace_negatives_and_missing_values(test_data)

# Feature engineering
train_data['population_parks_ratio'] = train_data['population_density'] / (train_data['number_of_parks'] + 1)  # Adding 1 to avoid division by zero
test_data['population_parks_ratio'] = test_data['population_density'] / (test_data['number_of_parks'] + 1)

common_columns = train_data.columns.intersection(test_data.columns)

#  features present in both datasets
X = train_data[common_columns.drop('development_trend_score', errors='ignore')]
y = train_data['development_trend_score']
test_features = test_data[common_columns]

# Handle missing values deuxieme fois in case
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
test_data_imputed = imputer.transform(test_features)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_data_scaled = scaler.transform(test_data_imputed)

# Split the dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=44)

# Hyperparameter grid for RandomForest
param_grid = { 
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']  
}

rf_model = RandomForestClassifier(random_state=42)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
y_val_pred = best_rf_model.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_val_pred)
print(f"Random Forest model accuracy : {accuracy_rf:.5f}")
print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")

# Cross-validation to get more reliable performance metrics
cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=5)
print(f"cv accuracy is: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predict on test data
test_predictions = best_rf_model.predict(test_data_scaled)

# Prepare submission file
submission = pd.DataFrame({
    "ID": test_data.index + 1,  # Assuming IDs are 1-based indices
    "development_trend_score": test_predictions
})
submission_file_path = '/kaggle/working/development_trend_predictions_rf.csv'
submission.to_csv(submission_file_path, index=False)
print(f"Submission file saved at: {submission_file_path}")

