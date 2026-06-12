import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Load the Dataset
df = pd.read_csv('laptop_price.csv')
# Clean the data
df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).astype(int)

# Clean ROM to handle both GB and TB, converting everything to GB
def clean_storage(val):
    val = str(val)
    if 'TB' in val:
        return int(float(val.replace('TB', '').strip()) * 1024) 
    elif 'GB' in val:
        return int(float(val.replace('GB', '').strip()))
    return 0

df['ROM'] = df['ROM'].apply(clean_storage)


# Define Features (X) and Target (y) based on your specific CSV headers
features = ['brand', 'processor', 'GPU', 'Ram', 'ROM']
target = 'price'

X = df[features]
y = df[target]


# Split the Data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Set up the Preprocessor
categorical_features = ['brand', 'processor', 'GPU']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Build the Modeling Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the Model
print("Training the Random Forest model...")
model.fit(X_train, y_train)

# Evaluate Accuracy
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Training Complete!")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print("-" * 30)

# Test a Custom Prediction
custom_laptop = pd.DataFrame({
    'brand': ['Dell'],
    'processor': ['12th Gen Intel Core i7'], # Matched format from your CSV
    'GPU': ['NVIDIA RTX 3060'],
    'Ram': [16],        
    'ROM': [512]     
})

predicted_price = model.predict(custom_laptop)
print(f"Estimated Price for Custom Dell Laptop: ${predicted_price[0]:.2f}")