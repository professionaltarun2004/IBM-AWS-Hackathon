"""
Streamlit Deployment App for Enhanced Panic Attack Prediction
Includes Twilio IVR integration for emergency alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    st.warning("Twilio not installed. Emergency calling will be simulated.")

from enhanced_panic_model import EnhancedPanicPredictor

# Page configuration
st.set_page_config(
    page_title="AuraVerse - Panic Attack Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
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

class TwilioIntegration:
    """Twilio integration for emergency alerts."""
    
    def __init__(self):
        # Twilio configuration (use environment variables in production)
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID', 'ACcc9dc2db42ae18f9113d84afa1f42e4c')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN', 'e9c5ef6706329d653e2ce205219d6372')
        self.from_phone = os.getenv('TWILIO_PHONE', '+12525184837')
        self.to_phone = os.getenv('CAREGIVER_PHONE', '+919391616573')
        
        # Initialize client (only if credentials are provided and Twilio is available)
        self.client = None
        if TWILIO_AVAILABLE and self.account_sid != 'your_account_sid_here':
            try:
                self.client = Client(self.account_sid, self.auth_token)
            except Exception as e:
                st.error(f"Twilio initialization error: {e}")
    
    def make_emergency_call(self, panic_probability, participant_name="Child"):
        """Make emergency IVR call to caregiver."""
        if not self.client:
            st.info("Twilio not configured - Running in demo mode")
            return self.simulate_call(panic_probability, participant_name)
        
        try:
            # Create TwiML for the call with proper XML formatting
            twiml_message = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">
        Alert! A panic attack has been detected for {participant_name}. 
        The panic probability is {panic_probability:.0%}. 
        Please check on them immediately and consider medical intervention if needed.
        Press any key to acknowledge this alert.
    </Say>
    <Gather numDigits="1" timeout="15">
        <Say voice="alice">Press any key to acknowledge this emergency alert.</Say>
    </Gather>
    <Say voice="alice">Alert acknowledged. Thank you for responding.</Say>
</Response>"""
            
            # Make the call
            call = self.client.calls.create(
                twiml=twiml_message,
                to=self.to_phone,
                from_=self.from_phone
            )
            
            return {
                'success': True,
                'call_sid': call.sid,
                'message': f"🚨 Emergency call initiated to {self.to_phone}",
                'twiml': twiml_message
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to make emergency call: {e}"
            }
    
    def simulate_call(self, panic_probability, participant_name):
        """Simulate emergency call for demo purposes."""
        twiml_demo = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">
        Alert! A panic attack has been detected for {participant_name}. 
        The panic probability is {panic_probability:.0%}. 
        Please check on them immediately.
    </Say>
</Response>"""
        
        return {
            'success': True,
            'call_sid': 'DEMO_CALL_' + str(int(time.time())),
            'message': f"🚨 DEMO: Emergency call simulation\n"
                      f"📞 Would call: {self.to_phone}\n"
                      f"⚠️ Alert: {participant_name} panic risk {panic_probability:.0%}",
            'twiml': twiml_demo
        }

@st.cache_resource
def load_panic_model():
    """Load the trained panic prediction model."""
    try:
        predictor = EnhancedPanicPredictor()
        predictor.load_model('panic_model.pkl')
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run 'python enhanced_panic_model.py' first to train the model.")
        return None

def create_risk_gauge(probability):
    """Create a risk level gauge visualization."""
    # Determine risk level and color
    if probability > 0.7:
        risk_level = "HIGH RISK"
        color = "red"
    elif probability > 0.4:
        risk_level = "MODERATE RISK"
        color = "orange"
    else:
        risk_level = "LOW RISK"
        color = "green"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Panic Risk Level<br><span style='color:{color}'>{risk_level}</span>"},
        delta = {'reference': 70},
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

def create_feature_importance_chart(predictor):
    """Create feature importance visualization."""
    if hasattr(predictor.model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': predictor.feature_names,
            'importance': predictor.model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'importance': 'Feature Importance', 'feature': 'Features'}
        )
        fig.update_layout(height=400)
        return fig
    return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AuraVerse - Panic Attack Prediction</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Real-time panic attack risk assessment for autistic children**
    
    This system uses machine learning to analyze physiological and behavioral data 
    to predict panic attack probability and trigger emergency alerts when needed.
    """)
    
    # Load model
    predictor = load_panic_model()
    if predictor is None:
        st.stop()
    
    # Initialize Twilio
    twilio = TwilioIntegration()
    
    # Sidebar for inputs
    st.sidebar.header("📊 Real-time Monitoring Inputs")
    st.sidebar.markdown("Enter current readings from wearable devices:")
    
    # User inputs
    heart_rate = st.sidebar.slider(
        "💓 Heart Rate (bpm)", 
        min_value=50, max_value=120, value=75, step=1,
        help="Current heart rate from wearable device"
    )
    
    accelerometer = st.sidebar.slider(
        "🏃 Activity Level (accelerometer)", 
        min_value=0.0, max_value=5.0, value=1.5, step=0.1,
        help="Physical activity intensity (0=rest, 5=high activity)"
    )
    
    skin_conductance = st.sidebar.slider(
        "⚡ Skin Conductance (μS)", 
        min_value=0.0, max_value=10.0, value=2.5, step=0.1,
        help="Electrodermal activity indicating stress/arousal"
    )
    
    sleep_duration = st.sidebar.slider(
        "😴 Sleep Duration (hours)", 
        min_value=0.0, max_value=12.0, value=7.5, step=0.5,
        help="Hours of sleep from previous night"
    )
    
    # Additional settings
    st.sidebar.header("⚙️ Settings")
    participant_name = st.sidebar.text_input("Child's Name", value="Alex")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 seconds", value=False)
    
    # Twilio configuration
    with st.sidebar.expander("📞 Twilio Configuration"):
        st.info("Configure Twilio for real emergency calls")
        twilio_sid = st.text_input("Account SID", value="", type="password", help="Your Twilio Account SID")
        twilio_token = st.text_input("Auth Token", value="", type="password", help="Your Twilio Auth Token")
        twilio_from = st.text_input("From Phone", value="+1234567890", help="Your Twilio phone number")
        twilio_to = st.text_input("Caregiver Phone", value="+1987654321", help="Emergency contact number")
        
        if st.button("Update Twilio Config"):
            if twilio_sid and twilio_token:
                # Update Twilio configuration
                twilio.account_sid = twilio_sid
                twilio.auth_token = twilio_token
                twilio.from_phone = twilio_from
                twilio.to_phone = twilio_to
                
                try:
                    if TWILIO_AVAILABLE:
                        twilio.client = Client(twilio_sid, twilio_token)
                        st.success("✅ Twilio configuration updated!")
                    else:
                        st.warning("⚠️ Twilio library not available - install with: pip install twilio")
                except Exception as e:
                    st.error(f"❌ Twilio configuration error: {e}")
            else:
                st.warning("Please provide both Account SID and Auth Token")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Make prediction
        try:
            panic_probability = predictor.predict_panic_probability(
                heart_rate=heart_rate,
                accelerometer=accelerometer,
                skin_conductance=skin_conductance,
                sleep_duration=sleep_duration
            )
            
            # Display risk gauge
            st.plotly_chart(create_risk_gauge(panic_probability), use_container_width=True)
            
            # Risk assessment
            if panic_probability > 0.7:
                st.markdown(f"""
                <div class="alert-high">
                    <h3>🚨 HIGH PANIC RISK DETECTED</h3>
                    <p><strong>Probability: {panic_probability:.1%}</strong></p>
                    <p>Immediate intervention recommended. Emergency alert will be triggered.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Trigger emergency call
                if st.button("🚨 Trigger Emergency Alert", type="primary"):
                    with st.spinner("Initiating emergency call..."):
                        call_result = twilio.make_emergency_call(panic_probability, participant_name)
                        
                        if call_result['success']:
                            st.success(call_result['message'])
                            
                            # Show TwiML details in expander
                            with st.expander("📋 Call Details"):
                                st.code(call_result.get('twiml', 'No TwiML available'), language='xml')
                                st.info(f"Call SID: {call_result['call_sid']}")
                            
                            st.balloons()
                        else:
                            st.error(call_result['message'])
                            if 'error' in call_result:
                                st.code(call_result['error'])
                
            elif panic_probability > 0.4:
                st.markdown(f"""
                <div class="alert-moderate">
                    <h3>⚠️ MODERATE PANIC RISK</h3>
                    <p><strong>Probability: {panic_probability:.1%}</strong></p>
                    <p>Monitor closely and consider preventive measures.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="alert-low">
                    <h3>✅ LOW PANIC RISK</h3>
                    <p><strong>Probability: {panic_probability:.1%}</strong></p>
                    <p>Continue regular monitoring.</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    with col2:
        # Current readings summary
        st.subheader("📋 Current Readings")
        
        st.metric("Heart Rate", f"{heart_rate} bpm", 
                 delta=f"{heart_rate-75:+d}" if heart_rate != 75 else None)
        st.metric("Activity Level", f"{accelerometer:.1f}", 
                 delta=f"{accelerometer-1.5:+.1f}" if accelerometer != 1.5 else None)
        st.metric("Skin Conductance", f"{skin_conductance:.1f} μS", 
                 delta=f"{skin_conductance-2.5:+.1f}" if skin_conductance != 2.5 else None)
        st.metric("Sleep Duration", f"{sleep_duration:.1f} hrs", 
                 delta=f"{sleep_duration-7.5:+.1f}" if sleep_duration != 7.5 else None)
        
        # Timestamp
        st.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Feature importance chart
    st.subheader("🔍 Model Insights")
    importance_chart = create_feature_importance_chart(predictor)
    if importance_chart:
        st.plotly_chart(importance_chart, use_container_width=True)
    
    # Historical data simulation
    st.subheader("📈 Risk Trend (Last 24 Hours)")
    
    # Generate simulated historical data
    hours = list(range(24))
    np.random.seed(42)
    base_risk = 0.3
    risk_trend = [
        base_risk + 0.2 * np.sin(h * np.pi / 12) + 0.1 * np.random.random() 
        for h in hours
    ]
    
    # Add current prediction
    risk_trend.append(panic_probability)
    hours.append(24)
    
    trend_df = pd.DataFrame({
        'Hour': hours,
        'Risk_Probability': risk_trend
    })
    
    fig_trend = px.line(
        trend_df, x='Hour', y='Risk_Probability',
        title="Panic Risk Trend",
        labels={'Hour': 'Hours Ago', 'Risk_Probability': 'Panic Probability'}
    )
    fig_trend.add_hline(y=0.7, line_dash="dash", line_color="red", 
                       annotation_text="High Risk Threshold")
    fig_trend.add_hline(y=0.4, line_dash="dash", line_color="orange", 
                       annotation_text="Moderate Risk Threshold")
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # System information
    with st.expander("ℹ️ System Information"):
        st.markdown(f"""
        **Model Information:**
        - Model Type: Enhanced Random Forest/XGBoost Classifier
        - Features Used: {len(predictor.feature_names)} total features
        - Training Data: 3,000 samples with synthetic heart rate
        - Panic Threshold: 70% probability
        
        **Integration Status:**
        - Twilio Voice API: {'✅ Configured' if twilio.client else '⚠️ Demo Mode'}
        - AWS IoT Ready: ✅ Architecture prepared
        - Real-time Processing: ✅ Active
        
        **Emergency Contacts:**
        - Caregiver Phone: {twilio.to_phone}
        - System Phone: {twilio.from_phone}
        """)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()