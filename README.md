# arogya-mitra
# Arogya Mitra 🤖💊

Arogya Mitra is a **Healthcare AI Assistant** built using Google's **Gemini Pro (Generative AI)** model via `google-generativeai`. It helps users understand symptoms and suggests preliminary medical advice — just like a digital health companion. Built with `Streamlit`, it's easy to use and deploy!
This is an AI-based healthcare assistant that uses machine learning models like **Random Forest** and **Support Vector Machine (SVM)** to predict possible diseases based on user-input symptoms.

---

## 📌 Dataset

The dataset used in this project was sourced from Kaggle:

🔗 [Disease Symptom Description Dataset – Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

This dataset consists of **4 CSV files** that include:

- Symptoms to disease mapping
- Disease descriptions
- Precautions
- Medications

---

## 🧠 Models Used

| Model           | Accuracy |
|----------------|----------|
| Random Forest  | ✅ 1.0    |
| SVM            | ⚠️ 0.6885 |

**Random Forest** gave perfect classification, whereas **SVM** performance varied across classes.

---

## 📊 Performance Summary

### 🔹 Random Forest
- Accuracy: **1.0**
- Classification Report: Precision, Recall, F1-score all = 1.00
- Confusion Matrix: No misclassifications

### 🔸 SVM
- Accuracy: ~**0.69**
- Lower performance on minority classes
- Cross-Validation Accuracy: **~0.71**
- Best hyperparameters from GridSearchCV:
  - `C = 0.1`
  - `kernel = linear`
  - `gamma = scale`

---

## 📂 Project Structure

Healthcare-AI-Assistant/
├── healthcare_ai_assistant.py # Main Python file
├── dataset/ # Contains 4 Kaggle CSV files
├── README.md # Project documentation
├── requirements.txt # Python dependencies



---

## 💻 How to Run

Make sure you have Python 3 installed. Then:

```bash
pip install -r requirements.txt
python healthcare_ai_assistant.py

⚠️ Note
This is a prototype system for educational purposes only.

It should not be used for real medical diagnosis.

Use of a more diverse and medically verified dataset is needed for production-level deployment.



---

## 🚀 Live App

👉 [Click here to use Arogya Mitra](https://arogya-mitra-eehgwotehrpz6pkhncubga.streamlit.app/)

---

## 🌟 Features

- Accepts user symptoms as input
- Generates AI-based suggestions using Google Gemini Pro
- User-friendly Streamlit UI
- Lightweight and easy to deploy
- PDF report generation (optional)

---

## 🧰 Tech Stack

| Tool | Description |
|------|-------------|
| 🐍 Python | Core programming language |
| 📚 Streamlit | Frontend UI |
| 🤖 Google Generative AI | LLM used (Gemini Pro) |
| 📦 scikit-learn, pandas, numpy | Data handling and ML backend |
| 📝 reportlab | PDF generation |
| 💼 joblib | Model serialization (if applicable) |

---

## 🖼️ Screenshots

![Screenshot](https://github.com/user-attachments/assets/1b240845-95ae-4498-be7b-e7113b8a0118)
