#!/usr/bin/env python3
"""Test predictions with different health scenarios."""

from src.predictor import DiabetesPredictor
from pathlib import Path

predictor = DiabetesPredictor(Path('models/diabetes_predictor_model.pkl'))

# Test Case 1: HEALTHY PERSON (FOLLOWING ADVICE)
print("=" * 70)
print("SCENARIO 1: HEALTHY PERSON (Following Advice)")
print("=" * 70)
print("Inputs: BMI=22, No HighBP, No HighChol, No Smoking, Exercises, Eats Fruits & Veggies, Excellent Health")
test1 = {
    'BMI': 22.0,
    'HighBP': 0,
    'HighChol': 0,
    'Smoker': 0,
    'PhysActivity': 1,
    'Fruits': 1,
    'Veggies': 1,
    'GenHlth': 1
}
pred1 = predictor.predict(test1)
print(f"\n[GOOD] DIABETES: {pred1.diabetes.label}")
print(f"  Risk Level: {pred1.diabetes.risk_level} (0=No, 1=Pre, 2=Yes)")
print(f"  Confidence: {pred1.diabetes.probability*100:.1f}%")
print(f"\n[GOOD] OBESITY: {pred1.obesity.label}")
print(f"  Risk Level: {pred1.obesity.risk_level} (0=Normal, 1=Obese)")
print(f"  Confidence: {pred1.obesity.probability*100:.1f}%")

# Test Case 2: RISKY PERSON (IGNORING ADVICE)
print("\n" + "=" * 70)
print("SCENARIO 2: RISKY PERSON (Ignoring Advice)")
print("=" * 70)
print("Inputs: BMI=35, HighBP=Yes, HighChol=Yes, Smoker=Yes, No Exercise, No Fruits, No Veggies, Poor Health")
test2 = {
    'BMI': 35.0,
    'HighBP': 1,
    'HighChol': 1,
    'Smoker': 1,
    'PhysActivity': 0,
    'Fruits': 0,
    'Veggies': 0,
    'GenHlth': 5
}
pred2 = predictor.predict(test2)
print(f"\n[BAD] DIABETES: {pred2.diabetes.label}")
print(f"  Risk Level: {pred2.diabetes.risk_level} (0=No, 1=Pre, 2=Yes)")
print(f"  Confidence: {pred2.diabetes.probability*100:.1f}%")
print(f"\n[BAD] OBESITY: {pred2.obesity.label}")
print(f"  Risk Level: {pred2.obesity.risk_level} (0=Normal, 1=Obese)")
print(f"  Confidence: {pred2.obesity.probability*100:.1f}%")

# Test Case 3: MEDIUM RISK
print("\n" + "=" * 70)
print("SCENARIO 3: MEDIUM RISK (Mixed)")
print("=" * 70)
print("Inputs: BMI=28, HighBP=No, HighChol=Yes, No Smoking, Exercises, No Fruits, Veggies=Yes, Good Health")
test3 = {
    'BMI': 28.0,
    'HighBP': 0,
    'HighChol': 1,
    'Smoker': 0,
    'PhysActivity': 1,
    'Fruits': 0,
    'Veggies': 1,
    'GenHlth': 3
}
pred3 = predictor.predict(test3)
print(f"\n[MEDIUM] DIABETES: {pred3.diabetes.label}")
print(f"  Risk Level: {pred3.diabetes.risk_level} (0=No, 1=Pre, 2=Yes)")
print(f"  Confidence: {pred3.diabetes.probability*100:.1f}%")
print(f"\n[MEDIUM] OBESITY: {pred3.obesity.label}")
print(f"  Risk Level: {pred3.obesity.risk_level} (0=Normal, 1=Obese)")
print(f"  Confidence: {pred3.obesity.probability*100:.1f}%")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)
print(f"Diabetes changes based on: BMI, HighBP, HighChol, Smoker, PhysActivity, GenHlth")
print(f"Obesity changes based on: BMI (>=30 = Obese)")
print(f"\nBoth SHOULD change when you follow/ignore advice about exercise, diet, smoking!")
print("=" * 70)
