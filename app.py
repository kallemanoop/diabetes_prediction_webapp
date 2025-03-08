import streamlit as st
import numpy as np
import joblib
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


selected = option_menu(None, ["Home", "Train"])
icons=["house", "cloud-upload"]

if selected=="Home":
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.markdown(
    """
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 20px; }
        .stTextInput>div>div>input { font-size: 18px; padding: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.write("### Enter patient details to predict diabetes.")

# Sidebar
    st.sidebar.image("https://i.ibb.co/RkVyKtkc/1-nwwkyz1wpa-RZg-Ag-Bg3h41-Q.jpg", use_container_width=True)
    st.sidebar.markdown("## About the Model")
    st.sidebar.info("This AI model predicts whether a patient has diabetes based on medical data.")

# Input Fields with Columns for Better Layout
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0)
        insulin = st.number_input("Insulin", min_value=0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    with col2:
        glucose = st.number_input("Glucose", min_value=0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
        age = st.number_input("Age", min_value=0, step=1)

# Predict Button with Animation
    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
    
    # Show Output with a Progress Bar
        with st.spinner('Analyzing...'):
            st.progress(100)
    
    # Display Result in a Card
        if prediction[0] == 1:
            st.error("**Diabetic**")
        else:
            st.success("**Non-Diabetic**")

if selected=="Train":
    st.write("Upload a CSV file and train a Random Forest model.")
    uploaded_file=st.file_uploader("Upload your input file", type=["csv"])

    if uploaded_file is not None:
        data=pd.read_csv(uploaded_file)

    if st.button("Train Model"):
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=25)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        class_report_rf = classification_report(y_test, y_pred_rf)
        joblib.dump(rf_model, 'diabetes_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        # st.write("Model and Scaler saved successfully!")
        # st.write("Random Forest Model Accuracy:", accuracy_rf)
        # st.write("Classification Report:\n", class_report_rf)
        TP = np.sum((y_test == 1) & (y_pred_rf == 1))
        TN = np.sum((y_test == 0) & (y_pred_rf == 0))
        FP = np.sum((y_test == 0) & (y_pred_rf == 1))
        FN = np.sum((y_test == 1) & (y_pred_rf == 0))

        # Compute Metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Display Metrics in Columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"<h3 style='text-align: center; color: black;'>Accuracy</h3><h2 style='text-align: center; color: #4CAF50;'>{accuracy:.4f}</h2>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h3 style='text-align: center; color: black;'>Precision</h3><h2 style='text-align: center; color: #FF5733;'>{precision:.4f}</h2>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<h3 style='text-align: center; color: black;'>Recall</h3><h2 style='text-align: center; color: #3498DB;'>{recall:.4f}</h2>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<h3 style='text-align: center; color: black;'>F1-Score</h3><h2 style='text-align: center; color: #9B59B6;'>{f1_score:.4f}</h2>", unsafe_allow_html=True)

    

