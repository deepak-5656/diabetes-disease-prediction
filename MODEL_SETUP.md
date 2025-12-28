# Model Setup Instructions

## About the Model

The trained machine learning model (`diabetes_predictor_model.pkl`) is **not included** in this repository because it's too large (>450MB). GitHub has a 100MB file limit.

## How to Generate the Model

### Option 1: Generate the Model Locally (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the dataset is in place:**
   - The dataset should be located at: `data/cleaned_diabetes_obesity_dataset.csv`
   - Download from: [BRFSS 2021 Health Indicators Dataset](https://www.kaggle.com/datasets/)

3. **Train the model:**
   ```bash
   python -m src.train
   ```

   This will:
   - Load and balance the dataset using oversampling
   - Train a RandomForest multi-output classifier
   - Save the model to `models/diabetes_predictor_model.pkl`
   - Generate metrics to `models/diabetes_predictor_model.metrics.json`

### Option 2: Download Pre-trained Model

If a pre-trained model is available on Google Drive or another cloud service:
- Download and place in `models/` directory
- Extract if compressed: `unzip models.zip -d models/`

## Model Details

- **Type:** Multi-Output Random Forest Classifier
- **Outputs:** 
  - Diabetes Risk (3 classes: No Diabetes, Prediabetes, Diabetes)
  - Obesity Risk (4 classes: Underweight, Normal, Overweight, Obese)
- **Training Data:** BRFSS 2021 (223,550 records)
- **Features:** 8 health indicators
- **Algorithm:** RandomForestClassifier with class balancing
- **Performance:** ~68% weighted F1-score, 61% accuracy

## Running the Application

After generating the model:

```bash
# Start Flask server
python app.py

# Open browser to: http://127.0.0.1:5000
```

## Troubleshooting

- **Model not found error:** Run `python -m src.train` to generate it
- **Out of memory during training:** Reduce dataset size or use a machine with more RAM
- **Import errors:** Ensure all dependencies from `requirements.txt` are installed

## File Structure

```
models/
├── diabetes_predictor_model.pkl      (generated - not in repo)
├── diabetes_predictor_model.metrics.json (generated - not in repo)
└── .gitkeep                          (placeholder)
```
