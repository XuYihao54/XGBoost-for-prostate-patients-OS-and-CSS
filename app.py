import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import matplotlib.pyplot as plt
import pickle

# Page configuration
st.set_page_config(
    page_title="Elderly Prostate Cancer Patient OS and CSS Prediction Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("🏥 Elderly Prostate Cancer Patient OS and CSS Prediction Calculator")
st.markdown("""
This tool is based on XGBoost machine learning models developed using SEER database data to predict 8-year Overall Survival and Cancer-Specific Survival for prostate cancer patients.
Please fill in the following patient information to obtain prediction results.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This prediction tool is constructed based on the following features:
    - Clinical features: Age, PSA value, Days from diagnosis to treatment
    - Tumor features: T stage, N stage, Gleason score
    - Treatment information: Primary site surgery and other site surgery, Radiation, Chemotherapy
    - Metastasis status: Bone metastasis, Liver metastasis, Lung metastasis, Brain metastasis
    - Socioeconomic factors: Income, Marital status, etc.
    """)
    
    st.header("Notes")
    st.markdown("""
    - This tool is for clinical decision support only and cannot replace professional medical judgment
    - Prediction results are based on historical data, actual outcomes may vary due to individual differences
    - Please ensure the accuracy of input data when using
    """)

# Load model function
@st.cache_resource
def load_models():
    """Load pre-trained XGBoost models and Platt parameters"""
    try:
        # Load OS XGBoost model
        xgb_model_os = xgb.Booster()
        xgb_model_os.load_model("C:/Users/1747351679/Desktop/前列腺癌/Web_Calculator_Assets/OS-xgboost_model.model")
        
        # Load OS Platt parameters
        with open("C:/Users/1747351679/Desktop/前列腺癌/Web_Calculator_Assets/OS-platt_params.json", 'r') as f:
            platt_params_os = json.load(f)
        
        # Load CSS XGBoost model
        xgb_model_css = xgb.Booster()
        xgb_model_css.load_model("C:/Users/1747351679/Desktop/前列腺癌/Web_Calculator_Assets/CSS-xgboost_model.model")
        
        # Load CSS Platt parameters
        with open("C:/Users/1747351679/Desktop/前列腺癌/Web_Calculator_Assets/CSS-platt_params.json", 'r') as f:
            platt_params_css = json.load(f)
        
        return xgb_model_os, platt_params_os, xgb_model_css, platt_params_css
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None, None

# Platt scaling function
def platt_scale(raw_pred, intercept, coefficient):
    """Apply Platt scaling"""
    # Avoid numerical issues
    raw_pred = np.clip(raw_pred, 1e-15, 1 - 1e-15)
    
    # Calculate logit
    logit = np.log(raw_pred / (1 - raw_pred))
    
    # Apply Platt transformation
    scaled_logit = intercept + coefficient * logit
    scaled_prob = 1 / (1 + np.exp(-scaled_logit))
    
    return scaled_prob

# Load models
xgb_model_os, platt_params_os, xgb_model_css, platt_params_css = load_models()

# Feature definition and encoding mapping
feature_config = {
    "Days to treatment": {
        "type": "numeric",
        "min": 0,
        "max": 731,
        "mean": 72.3214,
        "median": 57
    },
    "PSA value": {
        "type": "numeric",
        "min": 0.1,
        "max": 98,
        "mean": 15.8897,
        "median": 7.5
    },
    "Age": {
        "type": "numeric",
        "min": 65,
        "max": 90,
        "mean": 71.9225,
        "median": 71
    },
    "Race": {
        "type": "categorical",
        "levels": ["Black", "Others", "White"],  # 修正：按照R的实际顺序
        "mapping": {"Black": 1, "Others": 2, "White": 3}  # 修正：按照R的实际映射
    },
    "Marital status": {
        "type": "categorical",
        "levels": ["Married (including common law)", "Unmarried"],
        "mapping": {"Married (including common law)": 1, "Unmarried": 2}
    },
    "Histological grade": {
        "type": "categorical",
        "levels": ["Grade I&Grade II", "Grade III&Grade IV"],  # 修正：按照R的实际顺序
        "mapping": {"Grade I&Grade II": 1, "Grade III&Grade IV": 2}  # 修正：按照R的实际映射
    },
    "T stage": {
        "type": "categorical",
        "levels": ["T1", "T2", "T3", "T4", "TX"],
        "mapping": {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "TX": 5}
    },
    "N stage": {
        "type": "categorical",
        "levels": ["N0", "N1", "NX"],
        "mapping": {"N0": 1, "N1": 2, "NX": 3}
    },
    "Gleason Score": {
        "type": "categorical",
        "levels": ["≤6", "≥9", "7", "8"],  # 修正：按照R的实际顺序
        "mapping": {"≤6": 1, "≥9": 2, "7": 3, "8": 4}  # 修正：按照R的实际映射
    },
    "Surg.Prim": {
        "type": "categorical",
        "levels": ["Done", "None"],
        "mapping": {"Done": 1, "None": 2}
    },
    "Surg.Oth": {
        "type": "categorical",
        "levels": ["Done", "None"],
        "mapping": {"Done": 1, "None": 2}
    },
    "Radiation": {
        "type": "categorical",
        "levels": ["No/Unknown", "Yes"],
        "mapping": {"No/Unknown": 1, "Yes": 2}
    },
    "Chemotherapy": {
        "type": "categorical",
        "levels": ["No/Unknown", "Yes"],
        "mapping": {"No/Unknown": 1, "Yes": 2}
    },
    "Bone metastasis": {
        "type": "categorical",
        "levels": ["No", "Yes"],
        "mapping": {"No": 1, "Yes": 2}
    },
    "Liver metastasis": {
        "type": "categorical",
        "levels": ["No", "Yes"],
        "mapping": {"No": 1, "Yes": 2}
    },
    "Lung metastasis": {
        "type": "categorical",
        "levels": ["No", "Yes"],
        "mapping": {"No": 1, "Yes": 2}
    },
    "Brain metastasis": {
        "type": "categorical",
        "levels": ["No", "Yes"],
        "mapping": {"No": 1, "Yes": 2}
    },
    "Income": {
        "type": "categorical",
        "levels": ["$120,000+", "$40,000 - $79,999", "$800,000 - $119,999", "< $40,000"],  # 修正：按照R的实际顺序
        "mapping": {
            "$120,000+": 1, 
            "$40,000 - $79,999": 2, 
            "$800,000 - $119,999": 3, 
            "< $40,000": 4
        }  # 修正：按照R的实际映射
    }
}

# Feature order (consistent with model training)
feature_order_os = [
    "Days to treatment",
    "PSA value",
    "Age",
    "Race",
    "Marital status",
    "Histological grade",
    "T stage",
    "N stage",
    "Gleason Score",
    "Surg.Prim",
    "Surg.Oth",
    "Radiation",
    "Chemotherapy",
    "Bone metastasis",
    "Liver metastasis",
    "Lung metastasis",
    "Income"
]

feature_order_css = [
    "Race",
    "Marital status",
    "Histological grade",
    "T stage",
    "N stage",
    "Gleason Score",
    "Surg.Prim",
    "Surg.Oth",
    "Radiation",
    "Chemotherapy",
    "Bone metastasis",
    "Liver metastasis",
    "Lung metastasis",
    "Brain metastasis"
]

# Create input form
st.header("Patient Information Input")

# Use column layout to organize input fields
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic Information and Clinical Features")
    days_to_treatment = st.slider(
        "Days from diagnosis to treatment",
        min_value=0,
        max_value=731,
        value=57,
        help="Days from diagnosis to treatment"
    )
    
    psa_value = st.slider(
        "PSA value",
        min_value=0.1,
        max_value=98.0,
        value=7.5,
        step=0.1,
        help="Prostate-Specific Antigen value"
    )
    
    age = st.slider(
        "Age",
        min_value=65,
        max_value=90,
        value=71,
        help="Patient age"
    )
    
    race = st.selectbox(
        "Race",
        options=["White", "Others", "Black"],
        help="Patient race"
    )
    
    marital_status = st.selectbox(
        "Marital status",
        options=["Married (including common law)", "Unmarried"],
        help="Patient marital status"
    )
    
    income = st.selectbox(
        "Income level",
        options=["< $40,000", "$40,000 - $79,999", "$800,000 - $119,999", "$120,000+"],
        help="Patient income level"
    )

with col2:
    st.subheader("Tumor Characteristics")
    histological_grade = st.selectbox(
        "Histological grade",
        options=["Grade III&Grade IV", "Grade I&Grade II"],
        help="Tumor histological grade"
    )
    
    t_stage = st.selectbox(
        "T stage",
        options=["T1", "T2", "T3", "T4", "TX"],
        help="Tumor T stage"
    )
    
    n_stage = st.selectbox(
        "N stage",
        options=["N0", "N1", "NX"],
        help="Lymph node N stage"
    )
    
    gleason_score = st.selectbox(
        "Gleason Score",
        options=["≤6", "7", "8", "≥9"],
        help="Gleason score"
    )

with col3:
    st.subheader("Treatment Information and Metastasis Status")
    surg_prim = st.selectbox(
        "Primary surgery (Surg.Prim)",
        options=["Done", "None"],
        help="Whether primary surgery was performed"
    )
    
    surg_oth = st.selectbox(
        "Other surgery (Surg.Oth)",
        options=["Done", "None"],
        help="Whether other surgery was performed"
    )
    
    radiation = st.selectbox(
        "Radiation",
        options=["No/Unknown", "Yes"],
        help="Whether radiation was administered"
    )
    
    chemotherapy = st.selectbox(
        "Chemotherapy",
        options=["No/Unknown", "Yes"],
        help="Whether chemotherapy was administered"
    )
    
    bone_metastasis = st.selectbox(
        "Bone metastasis",
        options=["No", "Yes"],
        help="Presence of bone metastasis"
    )
    
    liver_metastasis = st.selectbox(
        "Liver metastasis",
        options=["No", "Yes"],
        help="Presence of liver metastasis"
    )
    
    lung_metastasis = st.selectbox(
        "Lung metastasis",
        options=["No", "Yes"],
        help="Presence of lung metastasis"
    )
    
    brain_metastasis = st.selectbox(
        "Brain metastasis",
        options=["No", "Yes"],
        help="Presence of brain metastasis"
    )

# Create prediction function
def predict_survival(input_data, model, platt_params):
    """Make predictions using the loaded model"""
    try:
        # Convert to DMatrix format (required by XGBoost)
        dmatrix = xgb.DMatrix(input_data.reshape(1, -1))
        
        # Make prediction using XGBoost model
        raw_pred = model.predict(dmatrix)[0]
        
        # Apply Platt scaling for calibration
        calibrated_pred = platt_scale(
            raw_pred, 
            platt_params['intercept'], 
            platt_params['coefficient']
        )
        
        return calibrated_pred
    except Exception as e:
        st.error(f"Error occurred during prediction: {str(e)}")
        return None

# Create submit button
if st.button("Predict 8-Year Mortality Probability", type="primary"):
    # Verify models loaded successfully
    if xgb_model_os is None or platt_params_os is None or xgb_model_css is None or platt_params_css is None:
        st.error("Model loading failed, cannot make predictions. Please check if model files exist.")
    else:
        # Convert input data to format required by models
        input_dict = {
            "Days to treatment": days_to_treatment,
            "PSA value": psa_value,
            "Age": age,
            "Race": feature_config["Race"]["mapping"][race],
            "Marital status": feature_config["Marital status"]["mapping"][marital_status],
            "Histological grade": feature_config["Histological grade"]["mapping"][histological_grade],
            "T stage": feature_config["T stage"]["mapping"][t_stage],
            "N stage": feature_config["N stage"]["mapping"][n_stage],
            "Gleason Score": feature_config["Gleason Score"]["mapping"][gleason_score],
            "Surg.Prim": feature_config["Surg.Prim"]["mapping"][surg_prim],
            "Surg.Oth": feature_config["Surg.Oth"]["mapping"][surg_oth],
            "Radiation": feature_config["Radiation"]["mapping"][radiation],
            "Chemotherapy": feature_config["Chemotherapy"]["mapping"][chemotherapy],
            "Bone metastasis": feature_config["Bone metastasis"]["mapping"][bone_metastasis],
            "Liver metastasis": feature_config["Liver metastasis"]["mapping"][liver_metastasis],
            "Lung metastasis": feature_config["Lung metastasis"]["mapping"][lung_metastasis],
            "Brain metastasis": feature_config["Brain metastasis"]["mapping"][brain_metastasis],
            "Income": feature_config["Income"]["mapping"][income]
        }
        
        # Create arrays according to feature order
        input_array_os = np.array([input_dict[feature] for feature in feature_order_os])
        input_array_css = np.array([input_dict[feature] for feature in feature_order_css])
        
        # Make predictions
        with st.spinner("Calculating prediction results..."):
            # Use model output directly as mortality probability
            os_mortality_prob = predict_survival(input_array_os, xgb_model_os, platt_params_os)
            css_mortality_prob = predict_survival(input_array_css, xgb_model_css, platt_params_css)
            
            if os_mortality_prob is not None and css_mortality_prob is not None:
                if css_mortality_prob > os_mortality_prob:
                    os_mortality_prob = min(os_mortality_prob + css_mortality_prob, 1.0)

            if os_mortality_prob is not None and css_mortality_prob is not None:
                # Display prediction results
                st.success("Prediction completed!")
                
                # Calculate survival probabilities (1 - mortality probability)
                os_survival_prob = 1 - os_mortality_prob
                css_survival_prob = 1 - css_mortality_prob
                
                # Use two-column layout to display results
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    # Add CSS styling for better display
                    st.markdown("""
                    <style>
                    .big-font {
                        font-size: 20px !important;
                        font-weight: bold !important;
                        color: #1f77b4;
                        margin-bottom: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<p class="big-font">8-Year Overall Survival Probability</p>', unsafe_allow_html=True)
                    st.metric(
                        label="",
                        value=f"{os_survival_prob:.2%}",
                        help="Overall survival probability prediction result after Platt scaling calibration"
                    )
                    # Add explanatory text and show mortality probability as secondary information
                    st.caption(f"Probability of survival from any cause within 8 years after diagnosis (Mortality probability: {os_mortality_prob:.2%})")
                
                with res_col2:
                    st.markdown("""
                    <style>
                    .big-font-css {
                        font-size: 20px !important;
                        font-weight: bold !important;
                        color: #d62728;
                        margin-bottom: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<p class="big-font-css">8-Year Cancer-Specific Survival Probability</p>', unsafe_allow_html=True)
                    st.metric(
                        label="",
                        value=f"{css_survival_prob:.2%}",
                        help="Cancer-specific survival probability prediction result after Platt scaling calibration"
                    )
                    # Add explanatory text and show mortality probability as secondary information
                    st.caption(f"Probability of survival from prostate cancer within 8 years after diagnosis (Mortality probability: {css_mortality_prob:.2%})")
            
            # Explanatory text
            st.info("""
            **Result Interpretation:**
            - This prediction is based on historical data, actual results may vary due to individual differences
            - This tool is for clinical decision support only and should not replace professional medical judgment
            """)
            
            # Risk level assessment
            st.subheader("Risk Level Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os_mortality_prob >= 0.6:
                    risk_level = "High Risk"
                    risk_color = "red"
                elif os_mortality_prob >= 0.3:
                    risk_level = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_level = "Low Risk"
                    risk_color = "green"
                
                st.markdown(f"""
                **Overall Mortality Risk:** <span style="color:{risk_color}; font-weight:bold; font-size:18px">{risk_level}</span>
                """, unsafe_allow_html=True)
            
            with col2:
                if css_mortality_prob >= 0.5:
                    risk_level = "High Risk"
                    risk_color = "red"
                elif css_mortality_prob >= 0.2:
                    risk_level = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_level = "Low Risk"
                    risk_color = "green"
                
                st.markdown(f"""
                **Cancer-Specific Mortality Risk:** <span style="color:{risk_color}; font-weight:bold; font-size:18px">{risk_level}</span>
                """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
    }
    </style>
    <div class="footer">
        This tool is for healthcare professionals' reference only and does not replace professional medical judgment. Prediction results are based on historical data models, actual results may vary due to individual differences.
    </div>
    """,
    unsafe_allow_html=True
)