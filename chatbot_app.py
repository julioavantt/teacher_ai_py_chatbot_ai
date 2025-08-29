import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import random
import pickle
import streamlit as st
import requests
import re
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer and geolocator
lemmatizer = WordNetLemmatizer()
geolocator = Nominatim(user_agent="chatbot_clima")

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Load saved words and classes
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

# Function to predict intent
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get coordinates from location name
def get_coordinates(location, api_key):
    # Try Nominatim first
    try:
        loc = geolocator.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    
    # Fallback to OpenWeatherMap geocoding API for countries or ambiguous locations
    try:
        geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": location,
            "limit": 1,
            "appid": api_key
        }
        response = requests.get(geo_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            return data[0]["lat"], data[0]["lon"]
        else:
            return None, None
    except requests.RequestException:
        return None, None

# Function to extract location and period from user input
def extract_location_and_period(sentence):
    sentence = sentence.lower()
    # Improved pattern to capture only city or country, stopping before period keywords
    location_pattern = r"(?:en|de)\s+([a-záéíóúñ\s]+?)(?=\s*(?:hoy|mañana|esta semana|próximos?\s*\d*\s*(?:días?|meses?|semanas?)?|para\s+(?:hoy|mañana|esta semana|próximos?\s*\d*\s*(?:días?|meses?|semanas?))?|$))"
    period_pattern = r"(hoy|mañana|esta semana|próximos?\s*(\d+)\s*(días?|meses?|semanas?)|para\s+(?:hoy|mañana|esta semana|próximos?\s*(\d+)\s*(días?|meses?|semanas?)))"
    
    location_match = re.search(location_pattern, sentence)
    period_match = re.search(period_pattern, sentence)
    
    location = location_match.group(1).strip() if location_match else None
    period = period_match.group(1) if period_match and period_match.group(1) else "hoy"
    days = None
    if period_match and period_match.group(2):
        days = int(period_match.group(2)) if period_match.group(3).startswith("día") else None
        if period_match.group(3).startswith("semana"):
            days = int(period_match.group(2)) * 7
        elif period_match.group(3).startswith("mes"):
            days = int(period_match.group(2)) * 30
    elif period == "esta semana":
        days = 7  # Assume "esta semana" means 7 days
    
    # Debug print to verify extraction
    print(f"Location: {location}, Period: {period}, Days: {days}")
    return location, period, days

# Function to get weather data from OpenWeatherMap API
def get_weather_data(latitude, longitude, intent_tag, days=None, api_key=os.getenv("API_KEY_WHEATHER")):
    base_url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
        "units": "metric",  # Use metric units for temperature in Celsius
        "lang": "es"  # Spanish language for descriptions
    }
    
    # For historical data (if needed for intent_tag == "pronostico_largo" and days > 8)
    if intent_tag == "pronostico_largo" and days and days > 8:
        return handle_historical_data(latitude, longitude, days, api_key)
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# Function to handle historical data (if needed for long-term forecasts)
def handle_historical_data(latitude, longitude, days, api_key):
    # OpenWeatherMap's One Call API only supports up to 8 days forecast.
    # For longer periods, use historical API (separate endpoint).
    # Historical data requires: https://api.openweathermap.org/data/3.0/onecall/timemachine
    # Note: This requires a paid subscription for full historical data access.
    if days > 8:
        return None  # Placeholder: Indicate that long-term forecasts beyond 8 days are not supported
    return None

# Function to format weather data
def format_weather_response(data, intent_tag, location, period, days=None):
    if not data:
        return f"Lo siento, no pude obtener los datos del clima para {location}."
    
    # OpenWeatherMap weather condition codes (simplified mapping)
    weather_codes = {
        200: "Tormenta con lluvia ligera",
        201: "Tormenta con lluvia",
        202: "Tormenta con lluvia fuerte",
        500: "Lluvia ligera",
        501: "Lluvia moderada",
        502: "Lluvia fuerte",
        600: "Nieve ligera",
        601: "Nieve",
        602: "Nieve fuerte",
        800: "Cielo despejado",
        801: "Pocas nubes",
        802: "Nubes dispersas",
        803: "Nubes rotas",
        804: "Cielo nublado"
    }
    
    if intent_tag == "clima_actual":
        current = data.get("current", {})
        temp = current.get("temp", "desconocida")
        precip = current.get("rain", {}).get("1h", 0)  # Rainfall in the last hour (mm)
        weather_code = current.get("weather", [{}])[0].get("id", 0)
        weather_desc = weather_codes.get(weather_code, current.get("weather", [{}])[0].get("description", "Condición desconocida"))
        return f"El clima actual en {location} es {weather_desc} con una temperatura de {temp}°C y precipitación de {precip} mm."
    
    elif intent_tag == "pronostico_corto":
        daily = data.get("daily", [])
        if not daily:
            return f"No hay datos de pronóstico disponibles para {location}."
        if period == "mañana":
            tomorrow = daily[1]
            temp_max = tomorrow.get("temp", {}).get("max", "desconocida")
            temp_min = tomorrow.get("temp", {}).get("min", "desconocida")
            precip = tomorrow.get("rain", 0)
            weather_code = tomorrow.get("weather", [{}])[0].get("id", 0)
            weather_desc = weather_codes.get(weather_code, tomorrow.get("weather", [{}])[0].get("description", "Condición desconocida"))
            return f"Mañana en {location} se espera {weather_desc} con temperaturas entre {temp_min}°C y {temp_max}°C, y precipitación de {precip} mm."
        else:
            # Short-term forecast (up to 8 days with One Call API)
            days = min(days or 7, 8)  # Default to 7 for "esta semana", max 8
            response = f"Pronóstico para los próximos {days} días en {location}:\n"
            for i in range(days):
                date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                day_data = daily[i]
                temp_max = day_data.get("temp", {}).get("max", "desconocida")
                temp_min = day_data.get("temp", {}).get("min", "desconocida")
                precip = day_data.get("rain", 0)
                weather_code = day_data.get("weather", [{}])[0].get("id", 0)
                weather_desc = weather_codes.get(weather_code, day_data.get("weather", [{}])[0].get("description", "Condición desconocida"))
                response += f"- {date}: {weather_desc}, {temp_min}°C - {temp_max}°C, precipitación {precip} mm\n"
            return response
    
    elif intent_tag == "pronostico_largo":
        daily = data.get("daily", [])
        if not daily:
            return f"No hay datos de pronóstico a largo plazo disponibles para {location}."
        days = min(days or 8, 8)  # One Call API supports up to 8 days
        response = f"Pronóstico a largo plazo para {location} (hasta {days} días):\n"
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            day_data = daily[i]
            temp_max = day_data.get("temp", {}).get("max", "desconocida")
            temp_min = day_data.get("temp", {}).get("min", "desconocida")
            precip = day_data.get("rain", 0)
            weather_code = day_data.get("weather", [{}])[0].get("id", 0)
            weather_desc = weather_codes.get(weather_code, day_data.get("weather", [{}])[0].get("description", "Condición desconocida"))
            response += f"- {date}: {weather_desc}, {temp_min}°C - {temp_max}°C, precipitación {precip} mm\n"
        if days > 8:
            response += "Nota: Los pronósticos a más de 8 días no están disponibles con la API actual. Contacta con OpenWeatherMap para datos históricos o a largo plazo."
        return response
    
    elif intent_tag == "clima_general":
        current = data.get("current", {})
        temp = current.get("temp", "desconocida")
        precip = current.get("rain", {}).get("1h", 0)
        weather_code = current.get("weather", [{}])[0].get("id", 0)
        weather_desc = weather_codes.get(weather_code, current.get("weather", [{}])[0].get("description", "Condición desconocida"))
        return f"El clima general en {location} hoy es {weather_desc} con una temperatura promedio de {temp}°C y precipitación de {precip} mm."
    
    return "Lo siento, no pude procesar los datos del clima."

# Function to get response
def get_response(ints, intents_json, user_input, api_key=os.getenv("API_KEY_WHEATHER")):
    tag = ints[0]['intent'] if ints else 'sin_respuesta'
    if tag in ["clima_actual", "pronostico_corto", "pronostico_largo", "clima_general"]:
        location, period, days = extract_location_and_period(user_input)
        if not location:
            return "Por favor, especifica una ciudad o país para consultar el clima."
        latitude, longitude = get_coordinates(location, api_key)
        if latitude is None or longitude is None:
            return f"No pude encontrar la ubicación {location}. Intenta con otra ciudad o país."
        weather_data = get_weather_data(latitude, longitude, tag, days, api_key)
        return format_weather_response(weather_data, tag, location, period, days)
    
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Lo siento, no entendí eso."

# Streamlit app
st.title("Mi Chatbot del Clima")
st.write("Pregúntame sobre el clima actual o el pronóstico en cualquier ciudad o país. Los pronósticos están disponibles hasta 8 días.")

# Initialize session state for chat history and input
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Create a form for input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Tú:", value=st.session_state.input_text, key="input_field")
    submit_button = st.form_submit_button("Enviar ➡️")

# Handle form submission
if submit_button and user_input.strip():
    # Predict intent and get response
    intents_pred = predict_class(user_input, model)
    response = get_response(intents_pred, intents, user_input)
    
    # Add to chat history
    st.session_state.chat_history.append(("Tú", user_input))
    st.session_state.chat_history.append(("Chatbot", response))
    
    # Clear the input field
    st.session_state.input_text = ""

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "Tú":
        st.markdown(f"**{speaker}**: {message}")
    else:
        st.markdown(f"**{speaker}**: {message}")