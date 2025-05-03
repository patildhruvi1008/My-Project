import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import io

# Sample data
SAMPLE_DATA = """Category,JEE Marks,MHT-CET Marks,10th Marks,12th Marks,Branch
General,250,180,95,90,Computer Science
OBC,200,160,85,82,Information Technology
SC,180,150,80,75,Mechanical
General,260,175,92,88,Computer Science
General,220,165,90,85,Electronics
OBC,190,155,82,80,Civil
SC,170,145,78,72,Electrical
General,240,170,94,87,Computer Science
OBC,210,160,86,83,Information Technology
General,230,165,91,86,Electronics
SC,175,148,79,74,Mechanical
General,255,178,93,89,Computer Science
OBC,205,158,84,81,Civil
SC,185,152,81,76,Electrical
General,245,172,92,88,Information Technology
OBC,195,156,83,79,Mechanical
General,235,168,90,87,Electronics
SC,180,150,80,75,Civil
General,250,175,94,89,Computer Science
OBC,215,162,87,84,Electrical"""

# Load and train model
df = pd.read_csv(io.StringIO(SAMPLE_DATA))

label_encoders = {}
for col in ['Category', 'Branch']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[['Category', 'JEE Marks', 'MHT-CET Marks', '10th Marks', '12th Marks']]
y = df['Branch']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Branch Predictor (JEE / MHT-CET)")

exam_type = st.selectbox("Select Exam Type", ["JEE", "MHT-CET"])
category = st.selectbox("Student Category", label_encoders["Category"].classes_.tolist())
exam_score = st.number_input(f"Enter {exam_type} Score", min_value=0.0, max_value=300.0 if exam_type == "JEE" else 200.0)
tenth_marks = st.number_input("10th Marks (%)", min_value=0.0, max_value=100.0)
twelfth_marks = st.number_input("12th Marks (%)", min_value=0.0, max_value=100.0)

if st.button("Predict Branch"):
    # Validations
    if exam_type == "MHT-CET":
        if category.lower() == "general" and exam_score < 90:
            st.error("General category requires at least 90 marks in MHT-CET")
            st.stop()
        elif category.lower() != "general" and exam_score < 80:
            st.error("Reserved categories require at least 80 marks in MHT-CET")
            st.stop()
    elif exam_type == "JEE":
        if category.lower() == "general" and exam_score < 90:
            st.error("General category requires at least 90 marks in JEE")
            st.stop()

    input_data = pd.DataFrame([[0, 0, 0, tenth_marks, twelfth_marks]],
        columns=['Category', 'JEE Marks', 'MHT-CET Marks', '10th Marks', '12th Marks'])
    
    input_data['Category'] = label_encoders['Category'].transform([category])[0]
    input_data[f'{exam_type} Marks'] = exam_score

    prediction = model.predict(input_data)[0]
    branch = label_encoders['Branch'].inverse_transform([prediction])[0]
    st.success(f"🎓 Recommended Branch: **{branch}**")
