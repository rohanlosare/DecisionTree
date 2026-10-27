from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Decision Tree model
with open('decision.pkl', 'rb') as file:
    model = pickle.load(file)

# HTML template embedded directly inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Placement Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-9">
                <div class="card shadow border-0 rounded-4 p-4 bg-white">
                    <h2 class="text-center mb-4 text-primary fw-bold">🎓 Campus Placement Prediction System</h2>
                    <form action="/predict" method="POST">
                        <div class="row g-3">
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">Gender</label>
                                <select name="gender" class="form-select">
                                    <option value="Male" {% if form_data and form_data.gender == 'Male' %}selected{% endif %}>Male</option>
                                    <option value="Female" {% if form_data and form_data.gender == 'Female' %}selected{% endif %}>Female</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">10th % (ssc_p)</label>
                                <input type="number" step="0.01" name="ssc_p" class="form-control" value="{{ form_data.ssc_p if form_data else '70.0' }}" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">10th Board (ssc_b)</label>
                                <select name="ssc_b" class="form-select">
                                    <option value="Central">Central</option>
                                    <option value="Others">Others</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">12th % (hsc_p)</label>
                                <input type="number" step="0.01" name="hsc_p" class="form-control" value="{{ form_data.hsc_p if form_data else '65.0' }}" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">12th Board (hsc_b)</label>
                                <select name="hsc_b" class="form-select">
                                    <option value="Central">Central</option>
                                    <option value="Others">Others</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">12th Specialization (hsc_s)</label>
                                <select name="hsc_s" class="form-select">
                                    <option value="Commerce">Commerce</option>
                                    <option value="Science">Science</option>
                                    <option value="Arts">Arts</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">Degree % (degree_p)</label>
                                <input type="number" step="0.01" name="degree_p" class="form-control" value="{{ form_data.degree_p if form_data else '68.0' }}" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">Degree Type (degree_t)</label>
                                <select name="degree_t" class="form-select">
                                    <option value="Sci&Tech">Sci&Tech</option>
                                    <option value="Comm&Mgmt">Comm&Mgmt</option>
                                    <option value="Others">Others</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">Work Experience (workex)</label>
                                <select name="workex" class="form-select">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">Employability Test % (etest_p)</label>
                                <input type="number" step="0.01" name="etest_p" class="form-control" value="{{ form_data.etest_p if form_data else '75.0' }}" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">MBA Specialization</label>
                                <select name="specialisation" class="form-select">
                                    <option value="Mkt&HR">Mkt&HR</option>
                                    <option value="Mkt&Fin">Mkt&Fin</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label fw-semibold">MBA % (mba_p)</label>
                                <input type="number" step="0.01" name="mba_p" class="form-control" value="{{ form_data.mba_p if form_data else '62.0' }}" required>
                            </div>
                        </div>
                        <div class="mt-4 text-center">
                            <button type="submit" class="btn btn-success btn-lg w-100 rounded-pill shadow-sm">Predict Placement Status</button>
                        </div>
                    </form>

                    {% if prediction_text %}
                        <div class="alert mt-4 text-center fw-bold fs-5 shadow-sm {% if 'Placed!' in prediction_text %}alert-success{% else %}alert-danger{% endif %}">
                            {{ prediction_text }}
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features required by the model structure[cite: 1]
        gender = 1 if request.form.get('gender') == 'Male' else 0
        ssc_p = float(request.form.get('ssc_p'))
        ssc_b = 1 if request.form.get('ssc_b') == 'Central' else 0
        hsc_p = float(request.form.get('hsc_p'))
        hsc_b = 1 if request.form.get('hsc_b') == 'Central' else 0
        
        hsc_s_map = {"Commerce": 0, "Science": 1, "Arts": 2}
        hsc_s = hsc_s_map.get(request.form.get('hsc_s'), 0)
        
        degree_p = float(request.form.get('degree_p'))
        
        degree_t_map = {"Sci&Tech": 0, "Comm&Mgmt": 1, "Others": 2}
        degree_t = degree_t_map.get(request.form.get('degree_t'), 0)
        
        workex = 1 if request.form.get('workex') == 'Yes' else 0
        etest_p = float(request.form.get('etest_p'))
        
        spec_map = {"Mkt&HR": 0, "Mkt&Fin": 1}
        specialisation = spec_map.get(request.form.get('specialisation'), 0)
        
        mba_p = float(request.form.get('mba_p'))

        # Feature vector matching model parameters[cite: 1]
        features = np.array([[
            gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, 
            degree_p, degree_t, workex, etest_p, specialisation, mba_p
        ]])

        prediction = model.predict(features)[0]
        
        result_text = "Candidate is likely to be Placed! 🎉" if prediction == 1 or str(prediction).lower() in ['placed', '1'] else "Candidate is unlikely to be Placed. ⚠️"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=result_text, form_data=request.form)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error in prediction: {str(e)}", form_data=request.form)

if __name__ == "__main__":
    app.run(debug=True)