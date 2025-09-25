"""
Simplified Streamlit app for panic attack prediction with Twilio integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from enhanced_panic_model import EnhancedPanicPredictor

# Page config
st.set_page_config(
    page_title="AuraVerse - Panic Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-moderate {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class SimpleTwilioIntegration:
    """Simplified Twilio integration for demo purposes."""
    
    def __init__(self):
        self.configured = False
        
    def simulate_emergency_call(self, panic_probability, participant_name):
        """Simulate emergency call."""
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">
        Alert! A panic attack has been detected for {participant_name}. 
        The panic probability is {panic_probability:.0%}. 
        Please check on them immediately.
    </Say>
</Response>"""
        
        return {
            'success': True,
            'message': f"🚨 Emergency Alert Triggered!\n📞 Would call caregiver\n⚠️ {participant_name} panic risk: {panic_probability:.0%}",
            'twiml': twiml
        }

@st.cache_resource
def load_model():
    """Load the panic prediction model."""
    try:
        predictor = EnhancedPanicPredictor()
        predictor.load_model('panic_model.pkl')
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_risk_gauge(probability):
    """Create risk gauge visualization."""
    if probability > 0.7:
        color = "red"
        risk_level = "HIGH RISK"
    elif probability > 0.4:
        color = "orange"
        risk_level = "MODERATE RISK"
    else:
        color = "green"
        risk_level = "LOW RISK"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': f"Panic Risk<br><span style='color:{color}'>{risk_level}</span>"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AuraVerse - Panic Attack Prediction</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Real-time panic attack risk assessment system**
    
    Enter physiological readings to get instant panic risk assessment with emergency alerting.
    """)
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.error("Model not available. Please run 'python enhanced_panic_model.py' first.")
        st.stop()
    
    # Initialize Twilio
    twilio = SimpleTwilioIntegration()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Input Readings")
        
        # Input controls
        heart_rate = st.slider("💓 Heart Rate (bpm)", 50, 120, 75)
        accelerometer = st.slider("🏃 Activity Level", 0.0, 5.0, 1.5, 0.1)
        skin_conductance = st.slider("⚡ Skin Conductance (μS)", 0.0, 10.0, 2.5, 0.1)
        sleep_duration = st.slider("😴 Sleep Duration (hours)", 0.0, 12.0, 7.5, 0.5)
        
        participant_name = st.text_input("Child's Name", "Alex")
        
        # Make prediction
        if st.button("🔍 Analyze Risk", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    panic_probability = predictor.predict_panic_probability(
                        heart_rate=heart_rate,
                        accelerometer=accelerometer,
                        skin_conductance=skin_conductance,
                        sleep_duration=sleep_duration
                    )
                    
                    # Store in session state
                    st.session_state.panic_probability = panic_probability
                    st.session_state.last_prediction = datetime.now()
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()
    
    with col2:
        st.subheader("📋 Current Readings")
        st.metric("Heart Rate", f"{heart_rate} bpm")
        st.metric("Activity", f"{accelerometer:.1f}")
        st.metric("Skin Conductance", f"{skin_conductance:.1f} μS")
        st.metric("Sleep", f"{sleep_duration:.1f} hrs")
    
    # Display results if available
    if hasattr(st.session_state, 'panic_probability'):
        probability = st.session_state.panic_probability
        
        # Risk gauge
        st.plotly_chart(create_risk_gauge(probability), use_container_width=True)
        
        # Risk assessment
        if probability > 0.7:
            st.markdown(f"""
            <div class="alert-high">
                <h3>🚨 HIGH PANIC RISK DETECTED</h3>
                <p><strong>Probability: {probability:.1%}</strong></p>
                <p>Immediate intervention recommended!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Emergency alert button
            if st.button("🚨 TRIGGER EMERGENCY ALERT", type="primary"):
                alert_result = twilio.simulate_emergency_call(probability, participant_name)
                st.success(alert_result['message'])
                
                with st.expander("📋 Alert Details"):
                    st.code(alert_result['twiml'], language='xml')
                
                st.balloons()
                
        elif probability > 0.4:
            st.markdown(f"""
            <div class="alert-moderate">
                <h3>⚠️ MODERATE PANIC RISK</h3>
                <p><strong>Probability: {probability:.1%}</strong></p>
                <p>Monitor closely and consider preventive measures.</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="alert-low">
                <h3>✅ LOW PANIC RISK</h3>
                <p><strong>Probability: {probability:.1%}</strong></p>
                <p>Continue regular monitoring.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Last prediction info
        st.info(f"Last prediction: {st.session_state.last_prediction.strftime('%H:%M:%S')}")
    
    # System information
    with st.expander("ℹ️ System Information"):
        st.markdown("""
        **Model Information:**
        - Enhanced Random Forest/XGBoost Classifier
        - 16 total features including synthetic heart rate
        - Trained on 3,000 samples
        - Panic threshold: 70% probability
        
        **Integration Status:**
        - Twilio Voice API: 🟡 Demo Mode
        - Real-time Processing: ✅ Active
        - Emergency Alerting: ✅ Functional
        """)
    
    # Instructions
    st.markdown("""
    ---
    **Instructions:**
    1. Adjust the sliders to match current readings from wearable devices
    2. Click "Analyze Risk" to get panic probability assessment
    3. If high risk is detected, use the emergency alert button
    4. Configure real Twilio credentials for actual emergency calls
    """)

if __name__ == "__main__":
    main()