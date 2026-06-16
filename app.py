import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# 1. Load your Logistic Regression pickle file safely
MODEL_PATH = "logistic_pkl.pkl"
model = None
model_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model_error = f"Model Unpickling Error: {str(e)}."

# 2. Strict ordered list of EXACTLY 63 features matching your model's X layout
FEATURE_ORDER = [
    'Person_ID', 'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI', 'Country', 'Occupation', 
    'Marital_Status', 'Wake_Up_Time', 'Sleep_Time', 'Sleep_Duration_Hours', 'Sleep_Quality_Score', 
    'Number_of_Night_Awakenings', 'Weekend_Sleep_Difference_Hours', 'Nap_Frequency_Per_Week', 
    'Screen_Time_Before_Bed_Hours', 'Exercise_Frequency_Per_Week', 'Exercise_Duration_Minutes', 
    'Exercise_Type', 'Daily_Steps', 'Morning_Workout', 'Workout_Intensity', 'Gym_Member', 
    'Daily_Calorie_Intake', 'Water_Intake_Liters', 'Fruit_Intake_Per_Day', 'Vegetable_Intake_Per_Day', 
    'Protein_Intake_Grams', 'Sugary_Drinks_Per_Week', 'Fast_Food_Meals_Per_Week', 'Breakfast_Regularity_Score', 
    'Smoking_Status', 'Alcohol_Consumption', 'Stress_Level', 'Working_Hours_Per_Day', 'Sitting_Hours_Per_Day', 
    'Outdoor_Time_Hours', 'Social_Interaction_Score', 'Meditation_Practice', 'Resting_Heart_Rate', 
    'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Level', 'Blood_Sugar_Level', 'Energy_Level_Score', 
    'Fatigue_Level_Score', 'Immune_Health_Score', 'Mood_Score', 'Anxiety_Score', 'Depression_Risk_Score', 
    'Productivity_Score', 'Focus_Concentration_Score', 'Life_Satisfaction_Score', 'Obesity_Risk', 
    'Hypertension_Risk', 'Diabetes_Risk', 'Cardiovascular_Risk', 'Sleep_Disorder_Risk', 'Health_Score', 
    'Fitness_Level', 'Healthy_Aging_Score', 'Wellness_Category'
]

# Text interpretation mapping for your y target (Early Wakeup)
PREDICTION_LABELS = {
    "0": "Night Owl / Late Riser Routine predicted.",
    "1": "Early Wakeup Chronotype confirmed!"
}

# 3. Categorical user mapping UI values to baseline model label indices
MAPPINGS = {
    'Gender': {'Male': 0.0, 'Female': 1.0, 'Other': 2.0},
    'Country': {'United States': 0.0, 'India': 1.0, 'United Kingdom': 2.0, 'Canada': 3.0, 'Australia': 4.0},
    'Occupation': {'Engineer': 0.0, 'Doctor': 1.0, 'Teacher': 2.0, 'Manager': 3.0, 'Student': 4.0, 'Other': 5.0},
    'Marital_Status': {'Single': 0.0, 'Married': 1.0, 'Divorced': 2.0, 'Widowed': 3.0},
    'Exercise_Type': {'None': 0.0, 'Cardio': 1.0, 'Strength': 2.0, 'Yoga': 3.0, 'Mixed': 4.0},
    'Morning_Workout': {'No': 0.0, 'Yes': 1.0},
    'Workout_Intensity': {'None': 0.0, 'Low': 1.0, 'Moderate': 2.0, 'High': 3.0},
    'Gym_Member': {'No': 0.0, 'Yes': 1.0},
    'Smoking_Status': {'Never Smoked': 0.0, 'Former Smoker': 1.0, 'Current Smoker': 2.0},
    'Alcohol_Consumption': {'Never/Rarely': 0.0, 'Socially': 1.0, 'Frequently': 2.0},
    'Meditation_Practice': {'No': 0.0, 'Yes': 1.0},
    'Wellness_Category': {'Category 0': 0.0, 'Category 1': 1.0, 'Category 2': 2.0}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Early Wakeup Predictive Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-gradient: linear-gradient(135deg, #f4f7f6 0%, #eef2f5 100%); --card-bg: #ffffff; --primary-blue: #3b82f6; }
        body { background: var(--bg-gradient); color: #1e293b; font-family: 'Plus Jakarta Sans', sans-serif; min-height: 100vh; padding: 50px 0; }
        .main-card { background: var(--card-bg); border: none; border-radius: 24px; box-shadow: 0 20px 40px rgba(30, 41, 59, 0.04); overflow: hidden; max-width: 950px; margin: 0 auto; }
        .hero-banner { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: #ffffff; padding: 45px; text-align: center; }
        .form-section-title { font-size: 1.15rem; font-weight: 700; color: #1e3a8a; margin: 32px 0 20px 0; padding-bottom: 10px; border-bottom: 2px solid #f1f5f9; display: flex; align-items: center; gap: 10px; }
        .form-label { font-size: 0.85rem; font-weight: 600; color: #334155; margin-bottom: 6px; }
        .form-control, .form-select { border: 1px solid #e2e8f0; border-radius: 12px; padding: 11px 16px; font-size: 0.95rem; background-color: #f8fafc; }
        .btn-submit { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; border: none; border-radius: 14px; padding: 15px; font-size: 1.1rem; font-weight: 700; width: 100%; margin-top: 20px; }
        .result-panel { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #bfdbfe; border-radius: 20px; padding: 26px; margin-bottom: 35px; text-align: center; max-width: 950px; margin-left: auto; margin-right: auto; }
        .error-panel { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border: 1px solid #fca5a5; border-radius: 20px; padding: 26px; margin-bottom: 35px; text-align: center; max-width: 950px; margin-left: auto; margin-right: auto; color: #991b1b; }
    </style>
</head>
<body>
<div class="container">
    
    {% if error_msg %}
    <div class="error-panel shadow-sm">
        <div class="fw-bold"><i class="fa-solid fa-triangle-exclamation me-2"></i>Application Runtime Exception</div>
        <div>{{ error_msg }}</div>
    </div>
    {% endif %}

    {% if prediction %}
    <div class="result-panel shadow-sm">
        <div style="font-size: 0.9rem; font-weight:700; color:#1e40af;">PREDICTIVE CHRONOTYPE OUTPUT</div>
        <div style="font-size: 1.75rem; font-weight:800; color:#1e3a8a;"><i class="fa-regular fa-clock me-2"></i>{{ prediction }}</div>
    </div>
    {% endif %}

    <div class="card main-card">
        <div class="hero-banner">
            <h1>Early Wakeup Analytics Portal</h1>
            <p>Data matches your model's exact 63-feature footprint including Wellness Category.</p>
        </div>
        
        <div class="card-body p-4 p-md-5">
            <form method="POST" action="/">
                <div class="form-section-title"><i class="fa-solid fa-clock"></i> Core Time Features</div>
                <div class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Wake Up Time (24h Decimal Format, e.g. 5.5 or 9.0)</label>
                        <input type="number" step="0.1" name="Wake_Up_Time" class="form-control" value="5.5" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Sleep Time Night Hour (24h Format, e.g. 22.0)</label>
                        <input type="number" step="0.1" name="Sleep_Time" class="form-control" value="22.0" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Sleep Duration (Hours)</label>
                        <input type="number" step="0.1" name="Sleep_Duration_Hours" class="form-control" value="7.5" required>
                    </div>
                </div>

                <div class="form-section-title"><i class="fa-solid fa-chart-line"></i> Categorical Model Inputs</div>
                <div class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Wellness Category Selection</label>
                        <select name="Wellness_Category" class="form-select">
                            {% for label in mappings['Wellness_Category'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Gender Selection</label>
                        <select name="Gender" class="form-select">
                            {% for label in mappings['Gender'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Morning Workout Habit?</label>
                        <select name="Morning_Workout" class="form-select">
                            {% for label in mappings['Morning_Workout'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                </div>

                <div class="form-section-title"><i class="fa-solid fa-user-gear"></i> Personal Biometrics</div>
                <div class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Age (Years)</label>
                        <input type="number" name="Age" class="form-control" value="26" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Height Measurement (cm)</label>
                        <input type="number" name="Height_cm" class="form-control" value="172" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Weight Measurement (kg)</label>
                        <input type="number" name="Weight_kg" class="form-control" value="68" required>
                    </div>
                </div>

                <button type="submit" class="btn btn-submit">Evaluate Early Wakeup Model</button>
            </form>
        </div>
    </div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_result = None
    error_msg = model_error
    
    if request.method == "POST":
        if model is None:
            return render_template_string(HTML_TEMPLATE, mappings=MAPPINGS, prediction=None, error_msg="Model failed to load.")
            
        try:
            # Build basic array matching exactly 63 elements
            form_data = {feature: 0.0 for feature in FEATURE_ORDER}
            
            # Setup realistic base metrics for columns hidden from the web screen layout
            form_data['Person_ID'] = 1.0
            form_data['Country'] = 0.0
            form_data['Occupation'] = 0.0
            form_data['Marital_Status'] = 0.0
            form_data['Sleep_Quality_Score'] = 7.0
            form_data['Number_of_Night_Awakenings'] = 1.0
            form_data['Weekend_Sleep_Difference_Hours'] = 1.0
            form_data['Nap_Frequency_Per_Week'] = 0.0
            form_data['Screen_Time_Before_Bed_Hours'] = 1.0
            form_data['Exercise_Frequency_Per_Week'] = 3.0
            form_data['Exercise_Duration_Minutes'] = 30.0
            form_data['Exercise_Type'] = 0.0
            form_data['Daily_Steps'] = 8000.0
            form_data['Workout_Intensity'] = 1.0
            form_data['Gym_Member'] = 0.0
            form_data['Daily_Calorie_Intake'] = 2000.0
            form_data['Water_Intake_Liters'] = 2.5
            form_data['Fruit_Intake_Per_Day'] = 2.0
            form_data['Vegetable_Intake_Per_Day'] = 2.0
            form_data['Protein_Intake_Grams'] = 60.0
            form_data['Sugary_Drinks_Per_Week'] = 1.0
            form_data['Fast_Food_Meals_Per_Week'] = 1.0
            form_data['Breakfast_Regularity_Score'] = 8.0
            form_data['Smoking_Status'] = 0.0
            form_data['Alcohol_Consumption'] = 0.0
            form_data['Stress_Level'] = 4.0
            form_data['Working_Hours_Per_Day'] = 8.0
            form_data['Sitting_Hours_Per_Day'] = 6.0
            form_data['Outdoor_Time_Hours'] = 2.0
            form_data['Social_Interaction_Score'] = 7.0
            form_data['Meditation_Practice'] = 0.0
            form_data['Resting_Heart_Rate'] = 70.0
            form_data['Systolic_BP'] = 120.0
            form_data['Diastolic_BP'] = 80.0
            form_data['Cholesterol_Level'] = 180.0
            form_data['Blood_Sugar_Level'] = 85.0
            form_data['Energy_Level_Score'] = 7.0
            form_data['Fatigue_Level_Score'] = 3.0
            form_data['Immune_Health_Score'] = 80.0
            form_data['Mood_Score'] = 7.0
            form_data['Anxiety_Score'] = 3.0
            form_data['Depression_Risk_Score'] = 2.0
            form_data['Productivity_Score'] = 7.0
            form_data['Focus_Concentration_Score'] = 7.0
            form_data['Life_Satisfaction_Score'] = 7.0
            form_data['Obesity_Risk'] = 0.0
            form_data['Hypertension_Risk'] = 0.0
            form_data['Diabetes_Risk'] = 0.0
            form_data['Cardiovascular_Risk'] = 0.0
            form_data['Sleep_Disorder_Risk'] = 0.0
            form_data['Health_Score'] = 80.0
            form_data['Fitness_Level'] = 75.0
            form_data['Healthy_Aging_Score'] = 80.0

            # Override parameters using values submitted via web form elements
            for key in request.form:
                val = request.form[key]
                if key in MAPPINGS:
                    form_data[key] = float(MAPPINGS[key].get(val, 0.0))
                else:
                    form_data[key] = float(val) if val else 0.0
            
            # Auto evaluate BMI computation metrics
            if form_data['Height_cm'] > 0:
                form_data['BMI'] = form_data['Weight_kg'] / ((form_data['Height_cm'] / 100) ** 2)
                
            # Align features into the exact structural footprint layout order
            ordered_input = [float(form_data[feature]) for feature in FEATURE_ORDER]
            final_features = np.array([ordered_input], dtype=np.float64)
            
            # Predict
            raw_pred = model.predict(final_features)[0]
            pred_str = str(int(raw_pred))
            
            # Format text outputs cleanly
            prediction_result = PREDICTION_LABELS.get(pred_str, f"Unknown Class Label: {pred_str}")
            
        except Exception as e:
            error_msg = f"Prediction Pipeline Failure: {str(e)}"

    return render_template_string(HTML_TEMPLATE, mappings=MAPPINGS, prediction=prediction_result, error_msg=error_msg)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
