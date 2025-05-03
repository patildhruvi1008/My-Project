import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Branch Predictor", layout="centered")

st.title("🎓 Branch Predictor using JEE / MHT-CET Scores")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is None:
    st.warning("⚠️ Please upload a dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Encode categorical variables
label_encoders = {}
for col in ['Category', 'Branch']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Choose exam type
exam_type = st.selectbox("Select Exam Type", ["JEE", "MHT-CET"])
score_column = "JEE Marks" if exam_type == "JEE" else "MHT-CET Marks"
max_score = 300 if exam_type == "JEE" else 200

# Train model
X = df[['Category', score_column, '10th Marks', '12th Marks']]
y = df['Branch']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Inputs
category = st.selectbox("Select Student Category", label_encoders['Category'].classes_)
category_encoded = label_encoders['Category'].transform([category])[0]

exam_score = st.slider(f"{exam_type} Marks (out of {max_score})", 0.0, float(max_score), step=1.0)
tenth_marks = st.slider("10th Marks (%)", 0.0, 100.0, step=1.0)
twelfth_marks = st.slider("12th Marks (%)", 0.0, 100.0, step=1.0)

# Validate cutoffs
valid = True
if exam_type == "MHT-CET":
    if category.lower() == "general" and exam_score < 90:
        st.error("General category requires at least 90 in MHT-CET.")
        valid = False
    elif category.lower() != "general" and exam_score < 80:
        st.error("Reserved categories require at least 80 in MHT-CET.")
        valid = False
elif exam_type == "JEE":
    if category.lower() == "general" and exam_score < 90:
        st.error("General category requires at least 90 in JEE.")
        valid = False

# Predict
if st.button("Predict Branch") and valid:
    input_data = pd.DataFrame([[category_encoded, 0, tenth_marks, twelfth_marks]],
                              columns=['Category', 'JEE Marks', '10th Marks', '12th Marks'])
    input_data[score_column] = exam_score

    probabilities = model.predict_proba(input_data)[0]
    branch_names = label_encoders['Branch'].inverse_transform(range(len(probabilities)))
    results = list(zip(branch_names, probabilities))
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🎯 Predicted Branch Preferences:")
    for i, (branch, prob) in enumerate(results, 1):
        st.write(f"**{i}. {branch}** — {prob:.2%}")
