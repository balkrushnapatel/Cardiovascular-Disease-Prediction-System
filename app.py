# import numpy as np
# import pickle
# import os
# from flask import Flask, request, render_template

# app = Flask(__name__)

# # Load Model/Scaler
# def load_model():
#     # Check if we need to merge chunks
#     if not os.path.exists('model.pkl') and os.path.exists('model.pkl.part0'):
#         print("Merging model chunks...")
#         with open('model.pkl', 'wb') as outfile:
#             chunk = 0
#             while True:
#                 chunk_filename = f"model.pkl.part{chunk}"
#                 if os.path.exists(chunk_filename):
#                     with open(chunk_filename, 'rb') as infile:
#                         outfile.write(infile.read())
#                     chunk += 1
#                 else:
#                     break
#         print("Model merged successfully.")

#     try:
#         with open('model.pkl', 'rb') as m:
#             model = pickle.load(m)
#         with open('scaler.pkl', 'rb') as s:
#             scaler = pickle.load(s)
#         return model, scaler
#     except FileNotFoundError:
#         print("Warning: model.pkl or scaler.pkl not found. Predictions will fail.")
#         return None, None

# model, scaler = load_model()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/features')
# def features():
#     return render_template('features.html')

# @app.route('/about-model')
# def about_model():
#     return render_template('about-model.html')

# @app.route('/faq')
# def faq():
#     return render_template('faq.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             # Extract raw form data
#             age_yrs = float(request.form['age_yrs'])
#             gender = int(request.form['gender'])
#             height = float(request.form['height'])
#             weight = float(request.form['weight'])
#             ap_hi = float(request.form['ap_hi'])
#             ap_lo = float(request.form['ap_lo'])
#             chol = int(request.form['chol'])
#             gluc = int(request.form['gluc'])
#             smoke = int(request.form['smoke'])
#             alco = int(request.form['alco'])
#             active = int(request.form['active'])

#             # Calculate BMI: weight (kg) / (height (m) ^ 2)
#             # Height is in cm, convert to meters
#             bmi = weight / ((height / 100) ** 2)

#             # Feature alignment: gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years, bmi
#             input_data = [
#                 gender,
#                 ap_hi,
#                 ap_lo,
#                 chol,
#                 gluc,
#                 smoke,
#                 alco,
#                 active,
#                 age_yrs,
#                 bmi
#             ]

#             final_features = np.array([input_data])
            
#             # Scale features
#             if scaler:
#                 final_features = scaler.transform(final_features)
            
#             # Predict
#             if model:
#                 prediction = model.predict(final_features)
#                 # Get probability
#                 prob = model.predict_proba(final_features)[0][1] * 100
                
#                 result_text = "High Risk Detected" if prediction[0] == 1 else "Low Risk Detected"
#                 result_class = "danger" if prediction[0] == 1 else "success" # for css styling
                
#                 return render_template('result.html', 
#                                      result=result_text, 
#                                      probability=f"{prob:.1f}%",
#                                      result_class=result_class,
#                                      bmi=f"{bmi:.2f}")
#             else:
#                 return render_template('result.html', result="Model not loaded", probability="N/A", result_class="warning")

#         except Exception as e:
#             return render_template('result.html', result=f"Error: {str(e)}", probability="N/A", result_class="danger")

#     return render_template('predict.html')

# if __name__ == "__main__":
#     app.run(debug=True)

import streamlit as st
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# ----------------- CACHED MODEL & SCALER LOADER -----------------
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"

    # Reconstruct the model if original file is missing
    if not os.path.exists(model_path):
        chunk_num = 0
        chunk_files = []

        while os.path.exists(f"{model_path}.part{chunk_num}"):
            chunk_files.append(f"{model_path}.part{chunk_num}")
            chunk_num += 1

        if not chunk_files:
            raise FileNotFoundError("Neither model.pkl nor model chunks were found.")

        with open(model_path, "wb") as output:
            for chunk_file in chunk_files:
                with open(chunk_file, "rb") as part:
                    output.write(part.read())

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()


# ----------------- NAVIGATION SIDEBAR -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predict Risk", "Home", "Features", "About Model", "FAQ"])


# ----------------- PAGE: PREDICT RISK -----------------
if page == "Predict Risk":
    st.title("❤️ Cardiovascular Disease Risk Predictor")
    st.write("Enter the patient's clinical and demographic measurements below to generate a real-time risk assessment.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Demographics & Vitals")
            age_yrs = st.number_input("Age (Years)", min_value=1.0, max_value=120.0, value=50.0, step=1.0)
            gender = st.selectbox("Gender", options=[(1, "Female"), (2, "Male")], format_func=lambda x: x[1])[0]
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0, step=0.5)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
            ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=50.0, max_value=250.0, value=120.0, step=1.0)
            ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30.0, max_value=160.0, value=80.0, step=1.0)

        with col2:
            st.subheader("Clinical & Lifestyle Factors")
            chol = st.selectbox(
                "Cholesterol Level", 
                options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")],
                format_func=lambda x: x[1]
            )[0]
            gluc = st.selectbox(
                "Glucose Level", 
                options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")],
                format_func=lambda x: x[1]
            )[0]
            smoke = st.selectbox("Smoking Habits", options=[(0, "Non-Smoker"), (1, "Smoker")], format_func=lambda x: x[1])[0]
            alco = st.selectbox("Alcohol Intake", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
            active = st.selectbox("Physical Activity", options=[(0, "Inactive"), (1, "Active")], format_func=lambda x: x[1])[0]

        submit_btn = st.form_submit_button("Predict Cardiovascular Risk", type="primary", use_container_width=True)

    if submit_btn:
        try:
            # BMI calculation: weight (kg) / (height (m) ^ 2)
            bmi = weight / ((height / 100.0) ** 2)

            # Strict 10-feature sequence matching training alignment
            input_data = [
                gender,
                ap_hi,
                ap_lo,
                chol,
                gluc,
                smoke,
                alco,
                active,
                age_yrs,
                bmi
            ]

            final_features = np.array([input_data])

            # Apply Scaler
            scaled_features = scaler.transform(final_features)

            # Predict & Probabilities
            prediction = model.predict(scaled_features)[0]
            prob = model.predict_proba(scaled_features)[0][1] * 100.0

            st.divider()
            st.subheader("Assessment Results")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Calculated BMI", f"{bmi:.2f}")
            metric_col2.metric("Disease Probability", f"{prob:.1f}%")

            if prediction == 1:
                metric_col3.error("High Risk Detected")
                st.error(f"⚠️ **High Risk Detected**: The model predicts an elevated likelihood of cardiovascular disease ({prob:.1f}% probability).")
            else:
                metric_col3.success("Low Risk Detected")
                st.success(f"✅ **Low Risk Detected**: The model predicts a low likelihood of cardiovascular disease ({prob:.1f}% probability).")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")


# ----------------- PAGE: HOME -----------------
elif page == "Home":
    st.title("Cardiovascular Health Intelligence Platform")
    st.write("""
    Welcome to the Cardiovascular Disease Prediction tool. 
    This application utilizes machine learning models trained on clinical patient data to identify individuals at potential risk of cardiovascular conditions.
    
    Use the navigation menu on the left to test inputs on the **Predict Risk** page.
    """)


# ----------------- PAGE: FEATURES -----------------
elif page == "Features":
    st.title("Model Feature Descriptions")
    st.markdown("""
    - **Age**: Patient's age in years.
    - **Gender**: Biological sex recorded (1 = Female, 2 = Male).
    - **Systolic / Diastolic Blood Pressure (ap_hi / ap_lo)**: Measured arterial blood pressures (mmHg).
    - **Cholesterol**: Serum cholesterol grouped into categories (Normal, Above Normal, Well Above Normal).
    - **Glucose**: Fasting blood glucose grouped into categories.
    - **Lifestyle**: Self-reported active status, smoking, and regular alcohol consumption.
    - **BMI**: Body Mass Index derived directly from height and weight.
    """)


# ----------------- PAGE: ABOUT MODEL -----------------
elif page == "About Model":
    st.title("About the Prediction Model")
    st.write("""
    - **Architecture**: Scikit-learn classification model paired with a feature standardizer (`scaler.pkl`).
    - **Preprocessing**: Automatic feature standard scaling followed by binary classification with probabilistic estimation.
    - **Chunk Handling**: Automatically reconstructs split binary model weights if chunked to bypass repository limits.
    """)


# ----------------- PAGE: FAQ -----------------
elif page == "FAQ":
    st.title("Frequently Asked Questions")
    st.markdown("""
    **Is this intended for direct clinical diagnosis?**  
    No. This tool is built as an educational and predictive screening tool, not a diagnostic medical replacement.

    **What does the calculated probability mean?**  
    The probability score reflects the confidence of the classification model that the provided feature vector corresponds to a positive cardiovascular case.
    """)
