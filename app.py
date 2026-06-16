import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# 1. Load your Logistic Regression pickle file
MODEL_PATH = "logistic_pkl.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. Ordered list of all 56 features expected by your specific model
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
    'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'Country': {'United States': 0, 'India': 1, 'United Kingdom': 2, 'Canada': 3, 'Australia': 4},
    'Occupation': {'Engineer': 0, 'Doctor': 1, 'Teacher': 2, 'Manager': 3, 'Student': 4, 'Other': 5},
    'Marital_Status': {'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3},
    'Exercise_Type': {'None': 0, 'Cardio': 1, 'Strength': 2, 'Yoga': 3, 'Mixed': 4},
    'Morning_Workout': {'No': 0, 'Yes': 1},
    'Workout_Intensity': {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
    'Gym_Member': {'No': 0, 'Yes': 1},
    'Smoking_Status': {'Never Smoked': 0, 'Former Smoker': 1, 'Current Smoker': 2},
    'Alcohol_Consumption': {'Never/Rarely': 0, 'Socially': 1, 'Frequently': 2},
    'Meditation_Practice': {'No': 0, 'Yes': 1}
}

# Target output string mapping dictionary for Wellness_Category classifications
OUTPUT_CLASSES = {
    0: "Highly Unhealthy / Critical",
    1: "At Risk / Action Required",
    2: "Moderate / Sub-optimal Wellness",
    3: "Healthy / Optimal Vitality"
}

# 4. Comprehensive HTML and CSS template
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
        :root {
            --bg-gradient: linear-gradient(135deg, #f4f7f6 0%, #eef2f5 100%);
            --card-bg: #ffffff;
            --primary-blue: #3a86ff;
            --secondary-dark: #1e293b;
            --accent-green: #06d6a0;
            --border-soft: #e2e8f0;
            --text-muted: #64748b;
        }

        body {
            background: var(--bg-gradient);
            color: var(--secondary-dark);
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            min-height: 100vh;
            padding: 50px 0;
        }

        .dashboard-container {
            max-width: 980px;
            margin: 0 auto;
        }

        .main-card {
            background: var(--card-bg);
            border: none;
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(30, 41, 59, 0.04);
            overflow: hidden;
        }

        .hero-banner {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: #ffffff;
            padding: 45px;
            text-align: center;
            border-bottom: 4px solid var(--accent-green);
        }

        .hero-banner h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }

        .hero-banner p {
            color: rgba(255, 255, 255, 0.85);
            font-size: 1rem;
            margin-bottom: 0;
        }

        .form-section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1e3a8a;
            margin: 32px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #f1f5f9;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .form-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 6px;
        }

        .form-control, .form-select {
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 11px 16px;
            font-size: 0.95rem;
            background-color: #f8fafc;
            transition: all 0.2s ease-in-out;
        }

        .form-control:focus, .form-select:focus {
            background-color: #fff;
            border-color: var(--primary-blue);
            box-shadow: 0 0 0 4px rgba(58, 134, 255, 0.15);
            outline: none;
        }

        .btn-submit {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 15px;
            font-size: 1.1rem;
            font-weight: 700;
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3);
            transition: all 0.2s ease;
        }

        .btn-submit:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        }

        .result-panel {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 1px solid #a7f3d0;
            border-radius: 20px;
            padding: 26px;
            margin-bottom: 35px;
            text-align: center;
            animation: slideDown 0.4s ease-out;
        }

        .result-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #065f46;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .result-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #064e3b;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-15px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container dashboard-container">
    
    {% if prediction %}
    <div class="result-panel shadow-sm">
        <div class="result-title"><i class="fa-solid fa-square-poll-vertical me-2"></i>Classification Diagnostics</div>
        <div class="result-value">{{ prediction }}</div>
    </div>
    {% endif %}

    <div class="card main-card">
        <div class="hero-banner">
            <h1>Dynamic Wellness Analytics</h1>
            <p>Enter your behavioral profiles via clean categorical dropdown mappings directly connected to your ML pipeline.</p>
        </div>
        
        <div class="card-body p-4 p-md-5">
            <form method="POST" action="/">
                
                <div class="form-section-title">
                    <i class="fa-solid fa-user-gear"></i> Demographic Configuration
                </div>
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
                        <label class="form-label">Marital Relations</label>
                        <select name="Marital_Status" class="form-select">
                            {% for label in mappings['Marital_Status'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                </div>

                <div class="form-section-title">
                    <i class="fa-solid fa-heart-pulse"></i> Biometric Dimensions
                </div>
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
                        <input type="number" name="Systolic_BP" class="form-control" value="118" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Diastolic Pressure (mmHg)</label>
                        <input type="number" name="Diastolic_BP" class="form-control" value="76" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Resting Heart Beats (BPM)</label>
                        <input type="number" name="Resting_Heart_Rate" class="form-control" value="68" required>
                    </div>
                </div>

                <div class="form-section-title">
                    <i class="fa-solid fa-person-running"></i> Lifestyle & Routine Profiles
                </div>
                <div class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Sleep Quantity (Hours)</label>
                        <input type="number" step="0.1" name="Sleep_Duration_Hours" class="form-control" value="7.8" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Sleep Quality Score (1-10)</label>
                        <input type="number" name="Sleep_Quality_Score" class="form-control" min="1" max="10" value="8" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Perceived Stress Scale (1-10)</label>
                        <input type="number" name="Stress_Level" class="form-control" min="1" max="10" value="3" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Tobacco/Smoking Habits</label>
                        <select name="Smoking_Status" class="form-select">
                            {% for label in mappings['Smoking_Status'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Alcohol Profile Matrix</label>
                        <select name="Alcohol_Consumption" class="form-select">
                            {% for label in mappings['Alcohol_Consumption'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Primary Exercise Modality</label>
                        <select name="Exercise_Type" class="form-select">
                            {% for label in mappings['Exercise_Type'] %}<option value="{{ label }}">{{ label }}</option>{% endfor %}
                        </select>
                    </div>
                </div>

                <div class="mt-5">
                    <button type="submit" class="btn btn-submit w-100">
                        <i class="fa-solid fa-wand-magic-sparkles me-2"></i>Evaluate Wellness Metrics
                    </button>
                </div>
            </form>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_result = None
    
    if request.method == "POST":
        if model is None:
            return render_template_string(HTML_TEMPLATE, mappings=MAPPINGS, prediction="System Error: Model file missing.")
            
        # Initialize an input dictionary array covering all expected features to safe neutral defaults
        form_data = {feature: 5.0 for feature in FEATURE_ORDER}
        
        # Override values present from user interface fields
        for key in request.form:
            val = request.form[key]
            
            # Convert human text options directly into index variables 
            if key in MAPPINGS:
                form_data[key] = float(MAPPINGS[key].get(val, 0))
            else:
                try:
                    form_data[key] = float(val) if val else 0.0
                except ValueError:
                    form_data[key] = 0.0
        
        # Dynamically calculate underlying engineering inputs (BMI Formula)
        if form_data['Height_cm'] > 0:
            form_data['BMI'] = form_data['Weight_kg'] / ((form_data['Height_cm'] / 100) ** 2)
            
        # Match features against strict array sequence configurations
        ordered_input = [form_data[feature] for feature in FEATURE_ORDER]
        final_features = np.array([ordered_input])
        
        # Query our Logistic Regression model
        raw_pred = model.predict(final_features)[0]
        
        # Clean classification label to text display mapping
        prediction_result = OUTPUT_CLASSES.get(int(raw_pred), f"Class Profile {raw_pred}")

    return render_template_string(HTML_TEMPLATE, mappings=MAPPINGS, prediction=prediction_result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
