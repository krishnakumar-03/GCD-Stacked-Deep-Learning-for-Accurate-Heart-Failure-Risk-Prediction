import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
model_dnn = load_model("dnn.h5")
model_cnn = load_model("cnn.h5")
model_gru = load_model("gru.h5")
scaler = joblib.load("scaler.pkl")

# Feature names and their safe ranges
feature_info = {
    "age": "Age (years) [Normal: < 50]",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest Pain Type [0–3] (0 = typical angina, 3 = asymptomatic)",
    "trestbps": "Resting BP [mm Hg] [Normal: 90–120]",
    "chol": "Cholesterol [mg/dL] [Normal: < 200]",
    "fbs": "Fasting Blood Sugar > 120 mg/dL (1 = true, 0 = false)",
    "restecg": "Resting ECG Results [0–2]",
    "thalach": "Max Heart Rate Achieved [Normal: 100–170]",
    "exang": "Exercise-induced Angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression [Normal: 0.0–2.0]",
    "slope": "Slope of ST Segment [0–2]",
    "ca": "Number of Major Vessels [0–3]",
    "thal": "Thalassemia [1 = normal, 2 = fixed defect, 3 = reversible defect]",
    "smoke": "Smoking (1 = yes, 0 = no)",
    "alcohol": "Alcohol Intake (1 = yes, 0 = no)"
}

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("Enter your health details below:")

# Create input fields dynamically
user_input = []
for key, label in feature_info.items():
    val = st.number_input(label, step=1.0 if key in ['oldpeak'] else 1)
    user_input.append(val)

# Prepare input
X = np.array(user_input).reshape(1, -1)
X_scaled = scaler.transform(X)
X_reshaped = X_scaled.reshape(1, 1, 15)  # For GRU (assuming GRU trained on this shape)

# Predict from each model
if st.button("Predict"):
    pred_dnn = model_dnn.predict(X_scaled)[0][0]
    pred_cnn = model_cnn.predict(X_scaled.reshape(1, 15, 1))[0][0]
    pred_gru = model_gru.predict(X_reshaped)[0][0]

    # Ensemble (average of all)
    avg_pred = np.mean([pred_dnn, pred_cnn, pred_gru])
    final_result = "🟢 No Heart Disease Detected" if avg_pred < 0.5 else "🔴 High Risk of Heart Disease"

    st.subheader("Prediction Result:")
    st.success(final_result)
    
    st.markdown(f"**DNN Prediction:** {pred_dnn:.2f}")
    st.markdown(f"**CNN Prediction:** {pred_cnn:.2f}")
    st.markdown(f"**GRU Prediction:** {pred_gru:.2f}")
    st.markdown(f"**Ensemble Average:** {avg_pred:.2f}")
