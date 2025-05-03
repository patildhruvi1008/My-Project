import os
import pandas as pd
import logging
import io
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Sample dataset as a CSV string (since we can't use the original file path)
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

# Model and label encoders
model = None
label_encoders = {}

def load_and_train_model():
    global model, label_encoders
    
    try:
        # Load the dataset from the CSV string
        df = pd.read_csv(io.StringIO(SAMPLE_DATA))
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['Category', 'Branch']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Train model for both exam types
        X = df[['Category', 'JEE Marks', 'MHT-CET Marks', '10th Marks', '12th Marks']]
        y = df['Branch']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        logger.info("Model trained successfully")
        return True
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def index():
    # Get the list of available categories from the label encoder
    if label_encoders and 'Category' in label_encoders:
        categories = label_encoders['Category'].classes_.tolist()
    else:
        categories = []
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        exam_type = request.form.get('exam_type')
        student_category = request.form.get('category')
        exam_score = float(request.form.get('exam_score'))
        tenth_marks = float(request.form.get('tenth_marks'))
        twelfth_marks = float(request.form.get('twelfth_marks'))
        
        # Validate inputs
        errors = []
        
        # Set max score based on exam type
        max_score = 300 if exam_type == "JEE" else 200
        score_column = "JEE Marks" if exam_type == "JEE" else "MHT-CET Marks"
        
        # Check score range
        if not (0 <= exam_score <= max_score):
            errors.append(f"Exam score must be between 0 and {max_score}")
        
        # Check cutoff requirements
        if exam_type == "MHT-CET":
            if student_category.lower() == "general" and exam_score < 90:
                errors.append("General category requires at least 90 marks in MHT-CET")
            elif student_category.lower() != "general" and exam_score < 80:
                errors.append("Reserved categories require at least 80 marks in MHT-CET")
        elif exam_type == "JEE":
            if student_category.lower() == "general" and exam_score < 90:
                errors.append("General category requires at least 90 marks in JEE")
        
        # Check 10th and 12th marks
        if not (0 <= tenth_marks <= 100):
            errors.append("10th marks must be between 0 and 100")
        if not (0 <= twelfth_marks <= 100):
            errors.append("12th marks must be between 0 and 100")
            
        if errors:
            for error in errors:
                flash(error, 'danger')
            return redirect(url_for('index'))
        
        # Encode category
        encoded_category = label_encoders['Category'].transform([student_category])[0]
        
        # Create input data
        input_data = pd.DataFrame([[encoded_category, 0, 0, tenth_marks, twelfth_marks]],
                               columns=['Category', 'JEE Marks', 'MHT-CET Marks', '10th Marks', '12th Marks'])
        
        # Set the appropriate score
        input_data[score_column] = exam_score
        
        # Predict
        predicted_branch_encoded = model.predict(input_data)[0]
        predicted_branch = label_encoders['Branch'].inverse_transform([predicted_branch_encoded])[0]
        
        # Store result in session
        session['prediction_result'] = {
            'branch': predicted_branch,
            'category': student_category,
            'exam_type': exam_type,
            'exam_score': exam_score,
            'tenth_marks': tenth_marks,
            'twelfth_marks': twelfth_marks
        }
        
        return redirect(url_for('result'))
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash(f"An error occurred: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/result')
def result():
    # Get prediction result from session
    prediction_result = session.get('prediction_result', None)
    if not prediction_result:
        flash("No prediction result found. Please submit the form first.", 'warning')
        return redirect(url_for('index'))
    
    return render_template('result.html', result=prediction_result)

# Initialize model when the app starts
def initialize():
    with app.app_context():
        load_and_train_model()

# Initialize model at startup
initialize()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
