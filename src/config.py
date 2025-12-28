"""Project-wide constants for reproducibility and maintainability."""

from pathlib import Path

# Feature configuration - matches cleaned_diabetes dataset
FEATURE_COLUMNS = [
    "BMI",
    "HighBP",
    "HighChol",
    "Smoker",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "GenHlth",
]

NUMERIC_FEATURES = [
    "BMI",
    "HighBP",
    "HighChol",
    "GenHlth",
]

CATEGORICAL_FEATURES = [
    "Smoker",
    "PhysActivity",
    "Fruits",
    "Veggies",
]

TARGET_COLUMN = "Diabetes_012"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "diabetes_predictor_model.pkl"
RANDOM_STATE = 42

# UI Configuration
DISEASE_INFO = {
    "Diabetes": {
        "description": "Type 2 Diabetes Mellitus (T2DM) is a chronic metabolic disorder",
        "icon": "🩺",
        "color": "#FF6B6B",
    }
}

RISK_LEVELS = {
    0: {"label": "No Diabetes Risk", "color": "#27AE60", "badge": "success"},
    1: {"label": "Prediabetes Risk", "color": "#F39C12", "badge": "warning"},
    2: {"label": "Diabetes Risk", "color": "#E74C3C", "badge": "danger"},
}

OBESITY_RISK_LEVELS = {
    0: {
        "label": "Underweight",
        "color": "#3498DB",
        "badge": "info",
        "status": "Underweight",
        "interpretation": "Your BMI is below normal range. Nutrition consultation recommended.",
        "advice": "Consult a nutritionist to ensure adequate calorie and protein intake."
    },
    1: {
        "label": "Normal Weight",
        "color": "#27AE60",
        "badge": "success",
        "status": "Normal Weight",
        "interpretation": "Your weight is within a healthy range for your height.",
        "advice": "Maintain your current lifestyle with regular exercise and balanced diet."
    },
    2: {
        "label": "Overweight",
        "color": "#F39C12",
        "badge": "warning",
        "status": "Overweight - Monitor Regularly",
        "interpretation": "Your weight is above the normal range. Monitoring is recommended.",
        "advice": "Increase physical activity to 150 minutes/week and reduce calorie intake."
    },
    3: {
        "label": "Obese",
        "color": "#E74C3C",
        "badge": "danger",
        "status": "High Obesity Risk",
        "interpretation": "Your BMI indicates obesity, which increases health risks.",
        "advice": "Consult a healthcare provider and consider a structured weight loss program."
    },
}
