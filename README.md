# 💻 Laptop Price Predictor

An end-to-end Machine Learning pipeline designed to estimate the market value of laptops based on core hardware specifications. This project processes categorical and numerical hardware data to output an accurate, continuous price prediction using an ensemble learning approach.

---

## 🚀 Features

- **Automated Preprocessing:** Utilizes `ColumnTransformer` and `OneHotEncoder` to seamlessly convert text-based categories (Brand, CPU, GPU) into machine-readable matrices.
- **Robust ML Pipeline:** Encapsulates data preprocessing and regression logic into a unified scikit-learn `Pipeline`, ensuring consistency between training data and future inference data.
- **High-Accuracy Estimations:** Leverages a `RandomForestRegressor` to map complex, non-linear relationships between premium hardware tiers and their market prices.
- **MLOps Ready:** The modular architecture makes it simple to containerize or deploy as a cloud web service.

---

## 📂 Dataset

This model requires the **Laptop Price Prediction** dataset.

- **Source:** Kaggle (Search for "Laptop Price Prediction" by authors like Eslam Elsolya or Arnab Chaki).
- **Key Features:** `Company`, `Cpu`, `Gpu`, `Ram`, `Memory`
- **Target Variable:** `Price`

> **Note:** Download the dataset and place it in the root directory as `laptop_price.csv`.

---

## 🛠️ Prerequisites & Installation

Ensure your environment is set up with Python 3.8+ and the necessary libraries.

```bash
# Clone your repository or navigate to your project folder
cd laptop-price-predictor

# Install required dependencies
pip install pandas scikit-learn numpy
```

---

## 💻 Usage

### 1. Train the Model

Run the main script to train the Random Forest model on your local dataset. This will output the Mean Absolute Error (MAE) to verify accuracy.

```bash
python laptop_predictor.py
```

### 2. Make Predictions (Inference)

Once the pipeline is trained, you can easily pass new hardware configurations to predict prices.

```python
import pandas as pd

# Define a custom laptop configuration
new_laptop = pd.DataFrame({
    'Company': ['Dell'],
    'Cpu': ['Intel Core i7'],
    'Gpu': ['NVIDIA RTX 3060'],
    'Ram': [16],
    'Memory': [512]
})

# Predict using the trained pipeline
predicted_price = model.predict(new_laptop)

print(f"Estimated Price: ${predicted_price[0]:.2f}")
```

---

## ☁️ Deployment (Render)

If you plan to expose this model via an API or a dark-mode web dashboard, this pipeline is structured to be easily deployed as a Web Service on Render.

### Steps

1. Create an `app.py` using Flask or FastAPI to serve the model.
2. Generate a `requirements.txt` containing:

```txt
pandas
scikit-learn
numpy
gunicorn
flask
```

3. Push the project to GitHub.
4. Connect the repository to Render for automatic continuous deployment.

---

## 📈 Future Improvements

- Hyperparameter tuning using `GridSearchCV` to optimize the Random Forest estimators.
- Feature engineering to extract clock speeds and performance metrics from CPU strings.
- Expanding the dataset to include newer architectures such as Apple M-series chips and NVIDIA RTX 40-series GPUs.
- Model persistence using `joblib` for production deployment.
- REST API integration for real-time price predictions.
- Docker containerization for scalable cloud deployment.

---

## 📜 License

This project is intended for educational and portfolio purposes. Feel free to modify and extend it for your own learning and development.
