import streamlit as st
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
from datetime import datetime
from reportlab.pdfgen import canvas
from io import BytesIO
# Set page config
st.set_page_config(page_title="Arogya Mitra", page_icon="🩺", layout="centered")
# Set the title of the app
st.title("🩺 Arogya Mitra - AI Disease Predictor")

# Add app description
st.markdown("""
### 📘 About Arogya Mitra
**Arogya Mitra** is an AI-powered web application that predicts possible diseases based on the symptoms you select.  
It also provides:
- Disease description
- Recommended precautions
- Severity risk level
- PDF report generation
- Symptom severity visualization

---

### 🧾 App Instructions
1. ✅ Select symptoms from the dropdown list.
2. 🔍 Click on **Predict Disease**.
3. 📖 View the predicted disease with description and precautions.
4. 📉 Check the severity score and graph.
5. 📄 Download a PDF report of the prediction.

---
""")

genai.configure(api_key=st.secrets["gemini_api_key"])

# Load the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit UI
# --- Symptom Selection ---
st.subheader("Select Your Symptoms 👇")
all_symptoms = ["Fever", "Cough", "Headache", "Fatigue", "Chest Pain", "Shortness of Breath", "Nausea", "Sore Throat"]
selected_symptoms = st.multiselect("Choose symptoms you are experiencing:", all_symptoms)


prompt = st.text_area("Enter your question:")

if st.button("Generate Response"):
    if prompt.strip():
        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(prompt)
                st.success("Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")




# Load only the Random Forest model and other required objects
rf_model = joblib.load('models/random_forest_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
all_symptoms = joblib.load('models/all_symptoms.pkl')

desc_df = pd.read_csv('symptom_Description.csv')
prec_df = pd.read_csv('symptom_precaution.csv')
severity_df = pd.read_csv('Symptom-severity.csv')

# App title
st.markdown("<h1 style='text-align: center; color: darkblue;'>🩺 AI Healthcare Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Empowered by Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Symptom input section
st.markdown("### 🤒 Select Your Symptoms")
selected_symptoms = st.multiselect("Choose from the list below:", all_symptoms)

# --- Utility: Calculate Severity Weight ---
def compute_severity_weight(symptom_list):
    weights = []
    for sym in symptom_list:
        row = severity_df[severity_df['Symptom'].str.lower().str.strip() == sym.lower().strip()]
        if not row.empty:
            weights.append(row['weight'].values[0])
        else:
            weights.append(1)  # default weight
    return sum(weights)

# Predict button
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Create input vector
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
        input_vector = np.array(input_vector).reshape(1, -1)

        # Predict using Random Forest
        pred_encoded = rf_model.predict(input_vector)[0]
        disease = label_encoder.inverse_transform([pred_encoded])[0]

        # Show prediction
        st.success(f"🧬 **Predicted Disease:** {disease}")
        st.markdown("---")

        # Description
        description_row = desc_df[desc_df['Disease'].str.lower() == disease.lower()]
        if not description_row.empty:
            st.markdown("### 🧾 Disease Description")
            st.info(description_row['Description'].values[0])
        else:
            st.warning("❌ No description found.")

        # Severity
        severity_score = compute_severity_weight(selected_symptoms)
        if severity_score >= 12:
            level = "🔴 High"
        elif severity_score >= 7:
            level = "🟠 Moderate"
        else:
            level = "🟢 Low"
        st.info(f"🧪 Severity Score: **{severity_score}** — Risk Level: **{level}**")

        # Precautions
        precaution_row = prec_df[prec_df['Disease'].str.lower() == disease.lower()]
        if not precaution_row.empty:
            st.markdown("### 💊 Recommended Precautions")
            for i in range(1, 5):
                precaution = precaution_row[f'Precaution_{i}'].values[0]
                if pd.notna(precaution):
                    st.write(f"✅ {precaution}")
        else:
            st.warning("❌ No precautions found.")

        # --- Visualize Results ---
        st.markdown("### 📊 Symptom Severity Visualization")
        symptom_weights = []
        for sym in selected_symptoms:
            row = severity_df[severity_df['Symptom'].str.lower().str.strip() == sym.lower().strip()]
            weight = row['weight'].values[0] if not row.empty else 1
            symptom_weights.append((sym, weight))

        viz_df = pd.DataFrame(symptom_weights, columns=["Symptom", "Severity Weight"])
        st.bar_chart(viz_df.set_index("Symptom"))

        # --- Generate PDF Report ---
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica", 12)

        c.drawString(100, 800, "🩺 AI Disease Prediction Report")
        c.drawString(100, 780, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(100, 760, f"Predicted Disease: {disease}")
        c.drawString(100, 740, f"Severity Score: {severity_score} ({level})")
        c.drawString(100, 720, "Selected Symptoms:")

        y = 700
        for sym in selected_symptoms:
            c.drawString(120, y, f"- {sym}")
            y -= 15
            if y < 100:
                c.showPage()
                y = 800

        c.save()
        buffer.seek(0)

        # --- Download button ---
        st.download_button(
            label="📄 Download PDF Report",
            data=buffer,
            file_name="disease_prediction_report.pdf",
            mime="application/pdf"
        )

# --- Footer ---
st.markdown("---")
st.caption("💡 Built with ❤️ using Streamlit & Google Gemini API")