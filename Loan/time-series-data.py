import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Example time series data with an 'id' column
time_series_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'value': range(100),
    'id': [1, 2, 3] * 33 + [1]  # Example ids to match static_data
})

# Example static data
static_data = pd.DataFrame({
    'id': [1, 2, 3],
    'feature1': ['A', 'B', 'C'],
    'feature2': [10, 20, 30]
})

# Merge datasets on 'id'
merged_data = pd.merge(time_series_data, static_data, on='id')

# Feature engineering
merged_data['lag_1'] = merged_data['value'].shift(1)
merged_data['rolling_mean'] = merged_data['value'].rolling(window=3).mean()

# Handle missing values
merged_data.bfill(inplace=True)

# Create a target variable (for example purposes, let's predict the next day's value)
merged_data['target'] = merged_data['value'].shift(-1)
merged_data.dropna(inplace=True)  # Drop rows with NaN values after shifting

# Normalize features
scaler = StandardScaler()
merged_data[['value', 'lag_1', 'rolling_mean', 'feature2']] = scaler.fit_transform(
    merged_data[['value', 'lag_1', 'rolling_mean', 'feature2']]
)

# Train-test split
train, test = train_test_split(merged_data, test_size=0.2, shuffle=False)

# Model training
model = RandomForestRegressor()
model.fit(train[['value', 'lag_1', 'rolling_mean', 'feature2']], train['target'])

# Model evaluation
predictions = model.predict(test[['value', 'lag_1', 'rolling_mean', 'feature2']])
