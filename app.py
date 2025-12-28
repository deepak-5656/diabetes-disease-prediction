"""Flask web application serving real-time diabetes risk predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

from src.config import DEFAULT_MODEL_PATH, FEATURE_COLUMNS, RISK_LEVELS, OBESITY_RISK_LEVELS
from src.predictor import DiabetesPredictor

app = Flask(__name__)
app.secret_key = "diabetes-predictor-key"

MODEL_PATH = Path(DEFAULT_MODEL_PATH)
try:
    predictor = DiabetesPredictor(MODEL_PATH)
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    predictor = None

# Form field configurations with user-friendly labels
FORM_FIELDS = {
    "personal": [
        {"name": "FullName", "label": "Full Name", "type": "text", "placeholder": "Enter your full name", "required": True},
        {"name": "Age", "label": "Age", "type": "number", "min": 18, "max": 120, "step": 1, "placeholder": "e.g., 45", "unit": "years", "required": True},
    ],
    "vital_signs": [
        {"name": "SystolicBP", "label": "Blood Pressure", "type": "number", "min": 70, "max": 250, "step": 1, "placeholder": "e.g., 120", "unit": "mmHg", "info": "Systolic (higher number, e.g., 120 in 120/80)"},
        {"name": "DiastolicBP", "label": "Blood Pressure", "type": "number", "min": 40, "max": 160, "step": 1, "placeholder": "e.g., 80", "unit": "mmHg", "info": "Diastolic (lower number, e.g., 80 in 120/80)"},
        {"name": "BMI", "label": "Body Mass Index (BMI)", "type": "number", "min": 10, "max": 60, "step": 0.1, "placeholder": "e.g., 28.5", "unit": "kg/m²"},
        {"name": "GenHlth", "label": "How is Your General Health?", "type": "select", "options": [
            {"value": "1", "label": "Excellent"},
            {"value": "2", "label": "Very Good"},
            {"value": "3", "label": "Good"},
            {"value": "4", "label": "Fair"},
            {"value": "5", "label": "Poor"}
        ], "placeholder": "Select health status"},
    ],
    "categorical": [
        {"name": "HighChol", "label": "Do you have High Cholesterol?", "options": [{"value": "0", "label": "No"}, {"value": "1", "label": "Yes"}], "placeholder": "Select"},
        {"name": "Smoker", "label": "Do you currently smoke?", "options": [{"value": "0", "label": "No"}, {"value": "1", "label": "Yes"}], "placeholder": "Select"},
        {"name": "PhysActivity", "label": "Do you exercise regularly?", "options": [{"value": "0", "label": "No"}, {"value": "1", "label": "Yes"}], "placeholder": "Select"},
        {"name": "Fruits", "label": "Do you eat fruits regularly?", "options": [{"value": "0", "label": "No"}, {"value": "1", "label": "Yes"}], "placeholder": "Select"},
        {"name": "Veggies", "label": "Do you eat vegetables regularly?", "options": [{"value": "0", "label": "No"}, {"value": "1", "label": "Yes"}], "placeholder": "Select"},
    ]
}


@app.route("/", methods=["GET"])
def index():
    """Render the landing page."""
    return render_template("landing.html")


@app.route("/form", methods=["GET"])
def form():
    """Render the prediction form."""
    if not predictor:
        return render_template("error.html", message="Model not loaded. Please train the model first."), 500
    return render_template("index.html", form_fields=FORM_FIELDS, risk_levels=RISK_LEVELS)


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predictions."""
    if not predictor:
        return jsonify({"error": "Model not available"}), 500
    
    data = request.get_json() or request.form
    
    try:
        full_name = data.get("FullName", "").strip()
        if not full_name:
            return jsonify({"error": "Full name is required"}), 400
        
        age = data.get("Age")
        systolic_bp = data.get("SystolicBP")
        diastolic_bp = data.get("DiastolicBP")
        
        form_data = _extract_form(data)
        if form_data is None:
            return jsonify({"error": "Invalid input data"}), 400
        
        prediction = predictor.predict(form_data)
        diabetes_risk_info = RISK_LEVELS[prediction.diabetes.risk_level]
        obesity_risk_info = OBESITY_RISK_LEVELS[prediction.obesity.risk_level]
        
        # Get BMI for context
        bmi = form_data.get('BMI', 25)
        
        # Calculate risk score based on classification level + confidence
        # Adjust based on BMI:
        # BMI 19-28: Keep at lower risk even if model predicts higher (Normal BMI range)
        # BMI < 19 or > 28: Allow higher risk scores
        
        diabetes_confidence = round(prediction.diabetes.probability * 100, 1)
        
        if prediction.diabetes.risk_level == 0:
            # No Diabetes: Always 0-33%
            if 19 <= bmi <= 28:
                # Normal BMI: 5-20% risk
                diabetes_risk_score = 10 + (diabetes_confidence - 50) * 0.2
            else:
                # Abnormal BMI: 0-33%
                diabetes_risk_score = (diabetes_confidence - 50) * 0.33
        elif prediction.diabetes.risk_level == 1:
            # Prediabetes: 34-66%
            if 19 <= bmi <= 28:
                # Normal BMI: reduce to 25-40%
                diabetes_risk_score = 32 + (diabetes_confidence - 50) * 0.15
            else:
                # Abnormal BMI: 34-66%
                diabetes_risk_score = 33.33 + (diabetes_confidence - 50) * 0.33
        else:  # Class 2: Diabetes
            # Diabetes: 67-100%
            diabetes_risk_score = 66.66 + (diabetes_confidence - 50) * 0.33
        
        diabetes_risk_score = max(0, min(100, diabetes_risk_score))  # Clamp to 0-100
        
        # Similar for obesity (0-50 for normal/overweight, 50-100 for obese)
        obesity_base_score = prediction.obesity.risk_level * 25
        obesity_confidence = round(prediction.obesity.probability * 100, 1)
        obesity_risk_score = obesity_base_score + (obesity_confidence - 50) * 0.5
        obesity_risk_score = max(0, min(100, obesity_risk_score))  # Clamp to 0-100
        
        return jsonify({
            "success": True,
            "full_name": full_name,
            "age": age,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "diabetes": {
                "risk_level": prediction.diabetes.risk_level,
                "label": prediction.diabetes.label,
                "probability": round(prediction.diabetes.probability * 100, 1),
                "risk_score": round(diabetes_risk_score, 1),  # New: dynamic risk score
                "color": diabetes_risk_info["color"],
            },
            "obesity": {
                "risk_level": prediction.obesity.risk_level,
                "label": prediction.obesity.label,
                "probability": round(prediction.obesity.probability * 100, 1),
                "risk_score": round(obesity_risk_score, 1),  # New: dynamic risk score
                "color": obesity_risk_info["color"],
            },
            "inputs": form_data,
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400


def _extract_form(form) -> Dict[str, Any] | None:
    """Extract and validate form data, converting BP readings to binary."""
    payload: Dict[str, Any] = {}
    try:
        payload["BMI"] = float(form.get("BMI", 0.0))
        payload["GenHlth"] = int(form.get("GenHlth", 0))
        
        # Convert blood pressure readings to HighBP binary (1 if high, 0 if normal)
        # High BP is typically >= 140/90 mmHg
        systolic = int(form.get("SystolicBP", 120))
        diastolic = int(form.get("DiastolicBP", 80))
        payload["HighBP"] = 1 if (systolic >= 140 or diastolic >= 90) else 0
        
        payload["HighChol"] = int(form.get("HighChol", 0))
    except (TypeError, ValueError):
        return None

    for field in ["Smoker", "PhysActivity", "Fruits", "Veggies"]:
        value = form.get(field)
        if value is None:
            return None
        payload[field] = int(value)

    # Ensure consistent ordering
    ordered_payload = {column: payload[column] for column in FEATURE_COLUMNS}
    return ordered_payload


@app.route("/api/risk-info", methods=["GET"])
def get_risk_info():
    """API endpoint for risk level information."""
    return jsonify(RISK_LEVELS)


@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    """Generate PDF report with prediction results."""
    try:
        data = request.get_json()
        
        full_name = data.get("full_name", "Unknown")
        age = data.get("age", "Unknown")
        systolic_bp = data.get("systolic_bp", "Unknown")
        diastolic_bp = data.get("diastolic_bp", "Unknown")
        bmi = data.get("inputs", {}).get("BMI", "Unknown")
        
        # Diabetes info
        diabetes_info = data.get("diabetes", {})
        diabetes_label = diabetes_info.get("label", "Unknown")
        diabetes_probability = diabetes_info.get("probability", 0)
        diabetes_risk_score = diabetes_info.get("risk_score", 0)
        
        # Obesity info
        obesity_info = data.get("obesity", {})
        obesity_label = obesity_info.get("label", "Unknown")
        obesity_probability = obesity_info.get("probability", 0)
        obesity_risk_score = obesity_info.get("risk_score", 0)
        
        inputs = data.get("inputs", {})
        
        # Create PDF in memory
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=20,
            alignment=1  # center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            spaceBefore=12,
            borderPadding=5
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leading=14
        )
        
        bold_style = ParagraphStyle(
            'BoldText',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            fontName='Helvetica-Bold',
            leading=14
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("Early Prediction of Lifestyle Diseases - Health Assessment Report", title_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Patient Information - Text format
        story.append(Paragraph("Patient Information", heading_style))
        story.append(Paragraph(f"<b>Name:</b> {full_name}", normal_style))
        story.append(Paragraph(f"<b>Age:</b> {age} years", normal_style))
        story.append(Paragraph(f"<b>Blood Pressure:</b> {systolic_bp}/{diastolic_bp} mmHg", normal_style))
        story.append(Paragraph(f"<b>BMI:</b> {bmi} kg/m²", normal_style))
        story.append(Paragraph(f"<b>Assessment Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # DIABETES RISK ASSESSMENT
        story.append(Paragraph("Diabetes Risk Assessment", heading_style))
        
        diabetes_advice = RISK_LEVELS.get(diabetes_info.get("risk_level", 0), {})
        story.append(Paragraph(f"<b>Risk Status:</b> {diabetes_label}", bold_style))
        story.append(Paragraph(f"<b>Risk Score:</b> {diabetes_risk_score}% (0% = No Risk, 100% = High Risk)", normal_style))
        story.append(Paragraph(f"<b>Assessment Confidence:</b> {diabetes_probability}%", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Assessment Details:</b>", bold_style))
        story.append(Paragraph(f"Based on the health parameters provided, the AI-generated assessment indicates {diabetes_label.lower()}. {diabetes_advice.get('interpretation', 'Regular monitoring is recommended.')}", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Recommended Actions:</b>", bold_style))
        if 'Diabetes' in diabetes_label:
            story.append(Paragraph("• <b>Schedule an urgent appointment with your physician or endocrinologist</b> for glucose screening tests including HbA1c and fasting glucose level.", normal_style))
            story.append(Paragraph("• Diabetes management typically involves blood glucose monitoring and medication.", normal_style))
        elif 'Prediabetes' in diabetes_label:
            story.append(Paragraph("• <b>Consult with your healthcare provider</b> for glucose tolerance testing and personalized management plan.", normal_style))
            story.append(Paragraph("• This is a critical stage where lifestyle modifications can prevent progression to Type 2 Diabetes.", normal_style))
        else:
            story.append(Paragraph("• <b>Maintain your current healthy lifestyle</b> to keep diabetes risk low.", normal_style))
            story.append(Paragraph("• Continue regular health checkups and monitor for any changes in health status.", normal_style))
        
        story.append(Paragraph(f"<b>Medical Advice:</b> {diabetes_advice.get('advice', 'Consult healthcare provider for personalized recommendations.')}", normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # OBESITY/WEIGHT STATUS ASSESSMENT
        story.append(Paragraph("Weight Status & Obesity Assessment", heading_style))
        
        obesity_advice = OBESITY_RISK_LEVELS.get(obesity_info.get("risk_level", 1), {})
        story.append(Paragraph(f"<b>Weight Status:</b> {obesity_label}", bold_style))
        story.append(Paragraph(f"<b>Risk Score:</b> {obesity_risk_score}% (0% = Underweight, 100% = Obese)", normal_style))
        story.append(Paragraph(f"<b>Assessment Confidence:</b> {obesity_probability}%", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Assessment Details:</b>", bold_style))
        story.append(Paragraph(f"Your current weight status is classified as {obesity_label.lower()} based on BMI calculation. {obesity_advice.get('interpretation', 'Weight monitoring is recommended.')}", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Recommended Actions:</b>", bold_style))
        if 'Obese' in obesity_label:
            story.append(Paragraph("• <b>Schedule an appointment with a weight management specialist or bariatrician</b> for comprehensive assessment.", normal_style))
            story.append(Paragraph("• Consider structured weight loss programs including nutritional counseling and behavioral therapy.", normal_style))
        elif 'Overweight' in obesity_label:
            story.append(Paragraph("• <b>Consult with a nutritionist or dietitian</b> to develop personalized weight management plan.", normal_style))
            story.append(Paragraph("• Aim for 150+ minutes of moderate physical activity per week with a calorie deficit diet.", normal_style))
        elif 'Underweight' in obesity_label:
            story.append(Paragraph("• <b>Consult with a nutritionist</b> to ensure adequate calorie and protein intake.", normal_style))
            story.append(Paragraph("• Focus on balanced nutrition with micronutrient assessment.", normal_style))
        else:
            story.append(Paragraph("• <b>Maintain your current healthy weight</b> through balanced diet and regular exercise.", normal_style))
            story.append(Paragraph("• Continue with healthy lifestyle habits.", normal_style))
        
        story.append(Paragraph(f"<b>Medical Advice:</b> {obesity_advice.get('advice', 'Maintain healthy lifestyle with balanced diet and exercise.')}", normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # OVERALL HEALTH RECOMMENDATIONS
        story.append(Paragraph("Overall Health Recommendations", heading_style))
        story.append(Paragraph("<b>This is an AI-generated assessment. Consulting with your doctor is essential.</b>", bold_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>General Lifestyle Guidelines:</b>", bold_style))
        story.append(Paragraph("• <b>Physical Activity:</b> Aim for at least 150 minutes of moderate-intensity aerobic exercise per week (walking, swimming, cycling).", normal_style))
        story.append(Paragraph("• <b>Diet:</b> Focus on whole grains, lean proteins, fresh fruits and vegetables, and healthy fats. Reduce sugar and processed foods.", normal_style))
        story.append(Paragraph("• <b>Smoking:</b> If you smoke, seek professional support to quit immediately. Smoking significantly increases diabetes and cardiovascular risks.", normal_style))
        story.append(Paragraph("• <b>Stress Management:</b> Practice meditation, yoga, deep breathing, or other relaxation techniques daily.", normal_style))
        story.append(Paragraph("• <b>Sleep:</b> Ensure 7-8 hours of quality sleep every night for better metabolic health.", normal_style))
        story.append(Paragraph("• <b>Regular Health Checkups:</b> Schedule annual physical examinations and appropriate screening tests.", normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # IMPORTANT NOTE
        story.append(Paragraph("Important Disclaimer", heading_style))
        disclaimer_text = ("<b>IMPORTANT:</b> This assessment report is generated by an artificial intelligence system for informational and educational purposes only. "
                          "It is <b>NOT a medical diagnosis</b> and should <b>NOT be used as a substitute for professional medical advice, diagnosis, or treatment</b>. "
                          "<br/><br/>"
                          "Please consult with qualified healthcare professionals including your primary care physician, endocrinologist, or relevant specialists for:<br/>"
                          "• Proper medical evaluation and diagnosis<br/>"
                          "• Personalized treatment plans<br/>"
                          "• Blood tests and diagnostic procedures<br/>"
                          "• Medication prescription if needed<br/>"
                          "<br/>"
                          "The AI predictions are based on statistical models trained on health data and may not be 100% accurate. "
                          "Individual health outcomes depend on many factors not captured by this assessment.")
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#c0392b'),
            spaceAfter=8,
            leading=12,
            borderPadding=8,
            borderRadius=3
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        # Generate filename
        filename = f"Health_Assessment_{full_name.replace(' ', '_')}_{datetime.now().strftime('%d%m%Y')}.pdf"
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        app.logger.error(f"PDF generation error: {e}")
        return jsonify({"error": str(e)}), 400


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", message="Page not found"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html", message="Server error occurred"), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
