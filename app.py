# app.py - Smartphone Addiction Predictor with Real Model
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="wide"
)

# Load the model and features
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except:
        st.error("❌ Model not found. Please run the notebook first to save the model.")
        return None, None

model, feature_names = load_model()

# Title
st.title("📱 Smartphone Addiction Predictor")
st.markdown("---")

# Sidebar - User Inputs
st.sidebar.header("👤 User Profile")

# Demographics
st.sidebar.subheader("📋 Demographics")
age = st.sidebar.slider("Age", 18, 60, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
gender_map = {"Male": 1, "Female": 0, "Other": 2}
gender_encoded = gender_map[gender]

stress = st.sidebar.selectbox("Stress Level", ["Low", "Medium", "High"])
stress_map = {"Low": 0, "Medium": 1, "High": 2}
stress_encoded = stress_map[stress]

academic = st.sidebar.selectbox("Academic Work Impact", ["No", "Yes"])
academic_encoded = 1 if academic == "Yes" else 0

# Usage Metrics
st.sidebar.subheader("📊 Daily Usage")
screen_time = st.sidebar.slider("Daily Screen Time (hours)", 0.0, 15.0, 7.5, 0.5)
social_media = st.sidebar.slider("Social Media (hours)", 0.0, 10.0, 3.0, 0.5)
gaming = st.sidebar.slider("Gaming (hours)", 0.0, 8.0, 1.5, 0.5)
work_study = st.sidebar.slider("Work/Study (hours)", 0.0, 10.0, 4.0, 0.5)
sleep = st.sidebar.slider("Sleep (hours)", 3.0, 10.0, 7.0, 0.5)
notifications = st.sidebar.slider("Notifications per day", 0, 300, 100, 10)
app_opens = st.sidebar.slider("App Opens per day", 0, 200, 80, 10)
weekend = st.sidebar.slider("Weekend Screen Time (hours)", 0.0, 16.0, 9.0, 0.5)

# Main content
st.header("📊 Prediction Results")

# Display user summary
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📱 Daily Screen Time", f"{screen_time:.1f} hrs")
    st.metric("🌙 Sleep", f"{sleep:.1f} hrs")

with col2:
    st.metric("📱 Social Media", f"{social_media:.1f} hrs")
    st.metric("📊 Stress", stress)

with col3:
    st.metric("📱 Weekend Usage", f"{weekend:.1f} hrs")
    st.metric("📋 Academic Impact", academic)

st.markdown("---")

# Predict button
if st.button("🔮 Predict Addiction", type="primary"):
    
    if model is None:
        st.error("❌ Model not loaded. Please save the model from the notebook first.")
    else:
        # Create user data with ALL 12 features
        user_data = pd.DataFrame({
            'age': [age],
            'daily_screen_time_hours': [screen_time],
            'social_media_hours': [social_media],
            'gaming_hours': [gaming],
            'work_study_hours': [work_study],
            'sleep_hours': [sleep],
            'notifications_per_day': [notifications],
            'app_opens_per_day': [app_opens],
            'weekend_screen_time': [weekend],
            'gender_encoded': [gender_encoded],
            'stress_level_encoded': [stress_encoded],
            'academic_work_impact_encoded': [academic_encoded]
        })
        
        # Make prediction
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0]
        
        # Display result
        if prediction == 1:
            result = "🔴 ADDICTED"
            confidence = probability[1] * 100
            color = "#ff4444"
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 30px; border-radius: 15px; text-align: center; border: 3px solid {color}">
                <h1 style="color: {color}; font-size: 48px;">{result}</h1>
                <p style="font-size: 24px;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show risk factors
            st.subheader("⚠️ Risk Factors")
            risk_factors = []
            if screen_time > 8:
                risk_factors.append("⚠️ High daily screen time (>8 hours)")
            if social_media > 4:
                risk_factors.append("⚠️ High social media usage (>4 hours)")
            if sleep < 6:
                risk_factors.append("⚠️ Insufficient sleep (<6 hours)")
            if app_opens > 150:
                risk_factors.append("⚠️ Frequent app switching (>150 opens)")
            for factor in risk_factors:
                st.write(f"• {factor}")
                
            st.subheader("💡 Recommendations")
            st.write("""
            - 📱 Set daily screen time limits (aim for < 6 hours)
            - 📱 Limit social media to < 2 hours per day
            - 😴 Aim for 7-8 hours of sleep
            - 📱 Take regular breaks (20-20-20 rule)
            - 🏃‍♂️ Engage in physical activities
            """)
            
        else:
            result = "🟢 NOT ADDICTED"
            confidence = probability[0] * 100
            color = "#44bb44"
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 30px; border-radius: 15px; text-align: center; border: 3px solid {color}">
                <h1 style="color: {color}; font-size: 48px;">{result}</h1>
                <p style="font-size: 24px;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("✅ Healthy usage patterns detected!")
            
            st.subheader("💡 Recommendations")
            st.write("""
            - ✅ Continue maintaining healthy usage patterns
            - 📱 Keep screen time under 6 hours
            - 😴 Maintain good sleep hygiene
            - 📱 Practice mindful phone usage
            """)

# Feature Importance Section
st.markdown("---")
st.header("📊 Feature Importance")

if model is not None:
    # Get feature importance from the model
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Display as bar chart
    st.bar_chart(importance_df.set_index('Feature'))
    
    # Show table
    st.subheader("Top 5 Most Important Features")
    st.dataframe(importance_df.head(5))

# Footer
st.markdown("---")
st.caption("🔬 Smartphone Addiction Predictor v1.0 | Powered by Random Forest (93.28% accuracy)")