import streamlit as st
import tensorflow as tf
import numpy as np  
from PIL import Image
import pandas as pd
import datetime
import requests

# Load the trained model
def load_model():
    return tf.keras.models.load_model('trained_plant_disease_model.keras')

model = load_model()

def model_prediction(image):
    image = image.resize((128, 128))  # Resize image to match model input size
    input_arr = np.array(image) / 255.0  # Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand dimensions to match model input shape
    predictions = model.predict(input_arr)
    confidence = np.max(predictions) * 100  # Convert confidence to percentage
    return np.argmax(predictions), confidence



# Disease treatment recommendations
disease_treatments = {
    'Potato___Early_blight': {
        'Organic Treatment': 'Neem oil spray can help manage the disease.',
        'Chemical Treatment': 'Use Chlorothalonil fungicide for effective control.'
    },
    'Potato___Late_blight': {
        'Organic Treatment': 'Apply copper-based fungicides for natural control.',
        'Chemical Treatment': 'Use Metalaxyl-based fungicides for strong protection.'
    },
    'Potato___Healthy': {
        'Organic Treatment': 'No treatment needed. Maintain proper crop care.',
        'Chemical Treatment': 'No chemical treatment required.'
    }
}

# Sidebar with UI enhancements
st.sidebar.title("🌿 Plant Disease Detection System")
st.sidebar.markdown("Helping farmers identify and treat plant diseases effectively.")
app_mode = st.sidebar.radio('📌 Navigation', ['🏠 Home', '🔍 Disease Recognition', '📜 Prediction History'])

# Display an image for UI enhancement
st.image("Disease.png", use_column_width=True)

# Create a DataFrame to store predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=['Date', 'Time', 'Predicted Class', 'Confidence'])

if app_mode == '🏠 Home':
    st.title("🌿 Welcome to the Plant Disease Detection System")
    st.write("This system helps farmers identify diseases in their crops and suggests treatments.")
    st.write("Navigate to the Disease Recognition page to get started!")


elif app_mode == '🔍 Disease Recognition':
    st.header('🦠 Disease Recognition for Potato Leaves')
    test_image = st.file_uploader('📸 Upload an image', type=['jpg', 'png', 'jpeg'])
    
    if test_image:
        image = Image.open(test_image)
        st.image(image, use_column_width=True, caption='Uploaded Image')

        if st.button('🔍 Predict Disease'):
            result_index, confidence = model_prediction(image)
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']
            predicted_class = class_name[result_index]
            
            st.success(f'✅ Model predicts: **{predicted_class}**')
            st.info(f'🎯 Confidence Score: **{confidence:.2f}%**')
            
            # Display disease treatment suggestions with UI enhancements
            treatments = disease_treatments.get(predicted_class, {})
            st.subheader("💊 Treatment Suggestions")
            st.write(f"**🌱 Organic Treatment:** {treatments.get('Organic Treatment', 'Not available')}")
            st.write(f"**🧪 Chemical Treatment:** {treatments.get('Chemical Treatment', 'Not available')}")
            
            # Save prediction to session state
            now = datetime.datetime.now()
            new_entry = pd.DataFrame({
                'Date': [now.strftime('%Y-%m-%d')],
                'Time': [now.strftime('%H:%M:%S')],
                'Predicted Class': [predicted_class],
                'Confidence': [f'{confidence:.2f}%']
            })
            st.session_state.predictions = pd.concat([st.session_state.predictions, new_entry], ignore_index=True)

elif app_mode == '📜 Prediction History':
    st.header('📋 Your Past Predictions')
    if not st.session_state.predictions.empty:
        st.dataframe(st.session_state.predictions)
    else:
        st.write("📭 No predictions made yet.")
def clear_predictions():
    st.session_state.predictions = pd.DataFrame(columns=['Date', 'Time', 'Predicted Class', 'Confidence'])

st.button('📝 Clear Prediction History', on_click=clear_predictions)



