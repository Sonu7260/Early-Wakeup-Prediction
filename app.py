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
    model_error = f"Model Unpickling Error: {str(e)}. Check your scikit-learn version match."

# 2. Strict ordered list of all 56 features expected by your model 
FEATURE_ORDER = [
    'Person_ID', 'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI', 'Country', 'Occupation', 
    'Marital_Status', 'Wake_Up_Time', 'Sleep_Time', 'Sleep_Duration_Hours', 'Sleep_Quality_Score', 
    'Number_of_Night_Awakenings', 'Weekend_Sleep_Difference_Hours', 'Nap_Frequency_Per_Week', 
    'Screen_Time_Before_Bed_Hours', 'Exercise_Frequency_Per_Week', 'Exercise_Duration_Minutes', 
    'Exercise_Type', 'Daily_Steps', 'Morning_Workout', 'Workout_Intensity', 'Gym_Member', 
    'Daily_Calorie_Intake', 'Water_Intake_L', 'Fast_Regularity_Score', 'Smoking_Status', 
    'Alcohol_Consumption', 'Stress_Level', 'Working_Hours_Per_Day', 'Sitting_Hours_Per_Day', 
    'Outdoor_Time_Hours', 'Social_Interaction_Score', 'Meditation_Practice', 'Resting_Heart_Rate', 
    'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Level', 'Blood_Sugar_Level', 'Energy_Level_Score', 
    'Fatigue_Level_Score', 'Immune_Health_Score', 'Mood_Score', 'Anxiety_Score', 'Depression_Risk_Score', 
    'Productivity_Score', 'Focus_Concentration_Score', 'Life_Satisfaction_Score', 'Obesity_Risk', 
    'Hypertension_Risk', 'Diabetes_Risk', 'Cardiovascular_Risk', 'Sleep_Disorder_Risk', 
    'Health_Score', 'Fitness_Level', 'Healthy_Aging_Score'
]

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
    'Meditation_Practice': {'No': 0.0, 'Yes': 1.0}
}

# Dynamic fallback options for any category not directly included on form page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wellness Intelligence Analytics</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-gradient: linear-gradient(135deg, #f4f7f6 0%, #eef2f5 100%); --card-bg: #ffffff; --primary-blue: #3b82f6; }
        body { background: var(--bg-gradient); color: #1e293b; font-family: 'Plus Jakarta Sans', sans-serif; min-height: 100vh; padding: 50px 0; }
        .main-card { background: var(--card-bg); border: none; border-radius: 24px; box-shadow: 0 20px 40px rgba(30, 41, 59, 0.04); overflow: hidden; max-width: 950px; margin: 0 auto; }
        .hero-banner { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: #ffffff; padding: 45px; text-align: center; }
        .form-section-title { font-size: 1.15rem; font-weight: 700; color: #1e3a8a; margin: 32px 0 20px 0; padding-bottom: 10px; border-bottom: 2px solid #f1f5f9; display: flex; align-items: center; gap: 10px; }
        .form-label { font-size: 0.85rem; font-weight: 600; color: #334155; margin-bottom: 6px; }
        .form-control, .form-select { border: 1px solid #e2e8f0; border-radius: 12px; padding: 11px 16px; font-size: 0.95rem; background-color: #f8fafc; }
        .btn-submit { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; border: none; border-radius: 14px; padding: 15px; font-size: 1.1rem; font-weight: 700; width: 100%; margin-top: 20px; }
        .result-panel { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border: 1px solid #a7f3d0; border-radius: 20px; padding: 26px; margin-bottom: 35px; text-align: center; max-width: 950px; margin-left: auto; margin-right: auto; }
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
        <div style="font-size: 0.9rem; font-weight:700; color:#065f46;">DIAGNOSTIC VISUALIZATION RESULT</div>
        <div style="font-size: 1.85rem; font-weight:800; color:#064e3b;">Raw Model Class Output: {{ prediction }}</div>
    </div>
    {% endif %}

    <div class="card main-card">
        <div class="hero-banner">
            <h1>Dynamic Wellness Analytics Portal</h1>
            <p>Input data values safely. Unsubmitted database features will fallback to default zeroes to protect processing structure.</p>
        </div>
        
        <div class="card-body p-4 p-md-5">
            <form method="POST" action="/">
                <div class="form-section-title"><i class="fa-solid fa-user-gear"></i> Demographic Data</div>
                <div class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Age Evaluation (Years)</label>
                        <input type="number" name="Age" class="form-control" value="32" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Gender Selection</label>
                        <select name="Gender" class="form-select">
                            {% for label in mappings['Gender'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Country Region</label>
                        <select name="Country" class="form-select">
                            {% for label in mappings['Country'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Occupation Field</label>
                        <select name="Occupation" class="form-select">
                            {% for label in mappings['Occupation'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Marital Status</label>
                        <select name="Marital_Status" class="form-select">
                            {% for label in mappings['Marital_Status'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                </div>

                <div class="form-section-title"><i class="fa-solid fa-heart-pulse"></i> Biometric Metrics</div>
                <div class="row g-3">
                    <div class="col-md-6">
                        <label class="form-label">Height Measurement (cm)</label>
                        <input type="number" name="Height_cm" class="form-control" value="172" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label">Weight Mass (kg)</label>
                        <input type="number" name="Weight_kg" class="form-control" value="68" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Systolic Pressure (mmHg)</label>
                        <input type="number" name="Systolic_BP" class="form-control" value="120" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Diastolic Pressure (mmHg)</label>
                        <input type="number" name="Diastolic_BP" class="form-control" value="80" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Resting Heart Beats (BPM)</label>
                        <input type="number" name="Resting_Heart_Rate" class="form-control" value="72" required>
                    </div>
                </div>

                <div class="form-section-title"><i class="fa-solid fa-person-running"></i> Lifestyle Routines</div>
                <div class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Sleep Quantity (Hours)</label>
                        <input type="number" step="0.1" name="Sleep_Duration_Hours" class="form-control" value="7.5" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Sleep Quality Score (1-10)</label>
                        <input type="number" name="Sleep_Quality_Score" class="form-control" min="1" max="10" value="8" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Stress Scale (1-10)</label>
                        <input type="number" name="Stress_Level" class="form-control" min="1" max="10" value="4" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Smoking Status</label>
                        <select name="Smoking_Status" class="form-select">
                            {% for label in mappings['Smoking_Status'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Alcohol Profile</label>
                        <select name="Alcohol_Consumption" class="form-select">
                            {% for label in mappings['Alcohol_Consumption'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Exercise Type</label>
                        <select name="Exercise_Type" class="form-select">
                            {% for label in mappings['Exercise_Type'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                </div>

                <button type="submit" class="btn btn-submit">Evaluate Wellness Metrics</button>
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
            return render_template_string(HTML_TEMPLATE, mappings=MAPPINGS, prediction=None, error_msg="Model file failed to load initially.")
            
        try:
            # Initialize all 56 features to 0.0 to prevent scikit-learn from encountering strings/NaNs
            form_data = {feature: 0.0 for feature in FEATURE_ORDER}
            
            # Map input elements securely
            for key in request.form:
                val = request.form[key]
                if key in MAPPINGS:
                    form_data[key] = float(MAPPINGS[key].get(val, 0.0))
                else:
                    form_data[key] = float(val) if val else 0.0
            
            # Handle standard calculated metrics
            if form_data['Height_cm'] > 0:
                form_data['BMI'] = form_data['Weight_kg'] / ((form_data['Height_cm'] / 100) ** 2)
                
            # Align exactly into sequential float array
            ordered_input = [float(form_data[feature]) for feature in FEATURE_ORDER]
            final_features = np.array([ordered_input], dtype=np.float64)
            
            # Run the prediction
            raw_pred = model.predict(final_features)[0]
            prediction_result = str(raw_pred)
            
        except Exception as e:
            error_msg = f"Prediction Pipeline Failure: {str(e)}"

    return render_template_string(HTML_TEMPLATE, mappings=MAPPINGS, prediction=prediction_result, error_msg=error_msg)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
