from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app) # This fixes the "Blocked by CORS" error between HTML and Python

# 1. Train the model ONCE when the server starts
print("Booting up API and training model...")
df = pd.read_csv('laptop_price.csv')

# Clean Data
df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).astype(int)
def clean_storage(val):
    val = str(val)
    if 'TB' in val: return int(float(val.replace('TB', '').strip()) * 1024) 
    elif 'GB' in val: return int(float(val.replace('GB', '').strip()))
    return 0
df['ROM'] = df['ROM'].apply(clean_storage)

# Build Pipeline
X = df[['brand', 'processor', 'GPU', 'Ram', 'ROM']]
y = df['price']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'processor', 'GPU'])],
    remainder='passthrough'
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X, y)
print("Model trained and API ready!")

@app.route('/')
def home():
    return render_template('index.html')

# EXISTING ROUTE: This handles the math
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    custom_laptop = pd.DataFrame([data])
    prediction = model.predict(custom_laptop)[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)