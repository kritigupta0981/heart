import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: #ffffff;
    }
    
    /* Titles and Headers */
    h1, h2, h3, .stSubheader {
        color: #ff4b4b !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161625;
        border-right: 1px solid #333;
    }
    
    /* Input fields and select boxes */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #2b2b3b !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }
    
    label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff7676 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    /* Metrics and Cards */
    .stMetric {
        background-color: #2b2b3b;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #ff4b4b;
    }
    
    /* Success/Error Alerts */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2b2b3b;
        border-bottom: 2px solid #ff4b4b !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
    This application predicts the likelihood of heart disease based on medical attributes.
    You can choose between different Machine Learning models and see their performance.
""")

# Load resources (Lazy loading)
@st.cache_resource
def load_resources():
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Load dataset for columns and comparison if needed
    df = pd.read_csv('heart disease.csv').drop_duplicates()
    return model, scaler, df

# Check if model exists, if not, prompt to run pipeline
if not os.path.exists('heart_model.pkl') or not os.path.exists('scaler.pkl'):
    st.warning("⚠️ Model files not found. Please run `heart_disease_pipeline.py` first to train and save the models.")
    if st.button("Run Training Pipeline"):
        with st.spinner("Training models..."):
            os.system("python3 heart_disease_pipeline.py")
        st.success("Training complete! Please refresh or continue.")
        st.rerun()
    st.stop()

model, scaler, df = load_resources()
X = df.drop('target', axis=1)

# Sidebar for Model Selection and Info
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select ML Model", 
    ["Random Forest (Recommended)", "Logistic Regression", "KNN", "Decision Tree"])

# Map UI names back to actual models if we were loading them dynamically, 
# but for now we'll use the pre-trained 'best' model for predictions.
# In a more advanced version, we could load different .pkl files.

# Input Section
st.subheader("📋 Enter Patient Clinical Data")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)

with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x==1 else "False")
    restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", value=150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0-3)", options=[0, 1, 2, 3])

# Prediction
if st.button("Predict Result"):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]], columns=X.columns)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.divider()
    
    if prediction == 1:
        st.error(f"### ⚠️ Prediction: Heart Disease Likely")
        st.write(f"Confidence Level: **{probability*100:.2f}%**")
    else:
        st.success(f"### ✅ Prediction: No Heart Disease Detected")
        st.write(f"Confidence Level: **{(1-probability)*100:.2f}%**")

# Analytics Section
st.divider()
st.subheader("📊 Model Performance & Data Insights")

tab1, tab2 = st.tabs(["Model Comparison", "Data Distribution"])

with tab1:
    if os.path.exists('correlation_heatmap.png'):
        st.image('correlation_heatmap.png', caption="Feature Correlation Heatmap")
    else:
        st.info("Run the training pipeline to see the correlation heatmap.")

with tab2:
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())
    
    col_to_plot = st.selectbox("Select feature to view distribution", X.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col_to_plot], kde=True, ax=ax, color='red')
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("Developed for Heart Disease Prediction Research.")
