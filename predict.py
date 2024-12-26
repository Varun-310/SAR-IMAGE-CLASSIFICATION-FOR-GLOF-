from tensorflow.keras.utils import img_to_array, load_img  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import requests
import json

# Load the model
model = load_model('glof_cnn_model.h5')  # Replace with your actual model path
API_KEY = "AIzaSyD10232-XJuWGt5zFyrf38cesmwJ1dMi8w"
DATABASE_URL = "https://glof-ff2e6-default-rtdb.asia-southeast1.firebasedatabase.app"

# Function to send the probability to Firebase
def send_probability_to_firebase_api(probability):
    # Firebase endpoint for updating the "prediction" key
    url = f"{DATABASE_URL}/GLOF_Predictions/prediction.json?auth={API_KEY}"

    # Data to send
    data = {
        "value": int(probability),
    }

    # Send data to Firebase
    response = requests.put(url, data=json.dumps(data))

    if response.status_code == 200:
        print("Data sent successfully to Firebase.")
    else:
        print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image in RGB mode and resize it
    image = load_img(image_path, color_mode='rgb', target_size=(128, 128))  # RGB mode
    image_array = img_to_array(image)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to predict and return the highest probability
def predict_highest_probability(image_path):
    preprocessed_image = preprocess_image(image_path)
    probabilities = model.predict(preprocessed_image)[0]  # Get probabilities
    highest_probability = np.max(probabilities) * 100  # Get highest probability and convert to percentage
    return highest_probability

# Example usage
image_path = 'D:\\SAR\\preprocessed_2023-06-06-00_00_2023-12-06-23_59_Sentinel-1_IW_VV+VH_IW_-_VH_[dB_gamma0].jpg'  # Replace with your image path
highest_probability = predict_highest_probability(image_path)

# Print the final output
print(f"Probability of GLOF: {highest_probability:.2f}%")
send_probability_to_firebase_api(highest_probability)
