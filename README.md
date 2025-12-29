# Lifestyle Disease Risk Prediction

A professional multi-output machine learning web application for assessing the simultaneous risk of **Diabetes**, **Hypertension**, and **Obesity** based on health vitals and lifestyle factors.

## Features

- **Multi-Output Prediction**: Predicts three diseases simultaneously using a single unified model
- **RandomForest + MultiOutputClassifier**: Robust ensemble learning with balanced class weighting
- **Real-Time Web Interface**: Bootstrap-based professional UI for clinician and patient use
- **Scalable ML Pipeline**: Modular training, preprocessing, and inference layers
- **Confidence Scores**: Per-disease probability estimates for clinical interpretation

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone or extract the project
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Format

The training dataset (`data/health_data.csv`) must contain the following columns:
- **Numeric**: Age, BMI, SystolicBP, DiastolicBP, Glucose, SleepHours
- **Categorical**: DietQuality, SmokingStatus, AlcoholUse, PhysicalActivity
- **Targets**: DiabetesRisk, HypertensionRisk, ObesityRisk

Sample rows are provided in `data/health_data.csv`.

## Training the Model

Train the multi-disease predictor with:

```bash
python -m src.train
```

Options:
- `--data PATH`: Path to training CSV (default: `data/health_data.csv`)
- `--model-path PATH`: Where to save the trained pipeline (default: `models/multi_disease_model.pkl`)
- `--test-size FLOAT`: Test set fraction (default: 0.2)
- `--estimators INT`: Number of trees per classifier (default: 300)
- `--max-depth INT`: Max tree depth (default: None)

The training script outputs metrics to console and saves a `.metrics.json` file alongside the model.

## Running the Application

Ensure the model is trained first. Then start the Flask development server:

```bash
python app.py
```

Access the application at `http://localhost:5000/`. Enter patient vitals and lifestyle data to receive instant risk predictions.

## Project Structure

```
mlproject/
├── data/
│   └── health_data.csv          # Training dataset
├── models/                       # Persisted pipelines
├── src/
│   ├── __init__.py
│   ├── config.py                 # Feature & path constants
│   ├── data_loader.py            # CSV parsing
│   ├── model_pipeline.py         # Scikit-learn pipeline construction
│   ├── predictor.py              # Inference wrapper
│   └── train.py                  # Training entrypoint
├── static/
│   └── styles.css                # Professional styling
├── templates/
│   ├── base.html                 # Navigation & layout
│   ├── index.html                # Prediction form
│   └── result.html               # Risk summary display
├── tests/
│   ├── test_data_loader.py       # Data loading validation
│   └── test_pipeline.py          # Training & inference tests
├── app.py                        # Flask application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage Example

1. **Navigate to Home**: Open `http://localhost:5000/`
2. **Fill Form**: Enter age, BMI, blood pressure, glucose, sleep hours, and lifestyle habits
3. **Submit**: Click "Predict Risk"
4. **View Results**: See risk classifications (High/Low) with confidence percentages

## Technical Details

### Preprocessing
- **Numeric features** are standardized with `StandardScaler`
- **Categorical features** are one-hot encoded
- `ColumnTransformer` orchestrates both steps

### Model
- Base estimator: `RandomForestClassifier` with 300 trees and balanced class weights
- Multi-output wrapper: `MultiOutputClassifier` allows independent predictions per disease
- Training uses stratified train-test split (80-20) to preserve class distribution

### Inference
- Predictions include both binary labels and positive-class probabilities
- Percentages help clinicians assess confidence in recommendations

## Testing

Run unit tests to validate the pipeline:

```bash
pytest tests/
```

## Production Notes

- Change `app.secret_key` in `app.py` before deployment
- Enable HTTPS/SSL in production
- Validate and sanitize all form inputs serverside
- Consider adding authentication/authorization for patient data
- Expand the dataset for improved generalization

## Future Enhancements

- Integrate additional medical features
- Implement user accounts and historical predictions
- Export predictions as PDF reports
- Deploy with Docker for easy scaling
