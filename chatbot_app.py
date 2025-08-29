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
def get_coordinates(location):
    try:
        loc = geolocator.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
        return None, None
    except:
        return None, None

# Function to extract location and period from user input
def extract_location_and_period(sentence):
    sentence = sentence.lower()
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
    
    print(f"Location: {location}, Period: {period}, Days: {days}")
    return location, period, days

# Function to get weather data from Open-Meteo API
def get_weather_data(latitude, longitude, intent_tag, days=None):
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "timezone": "auto",
        "forecast_days": min(days or 7, 16)  # Open-Meteo supports up to 16 days
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# Function to format weather data
def format_weather_response(data, intent_tag, location, period, days=None):
    if not data:
        return f"Lo siento, no pude obtener los datos del clima para {location}. Intenta de nuevo más tarde."
    
    # WMO weather codes (used by Open-Meteo)
    weather_codes = {
        0: "Cielo despejado",
        1: "Mayormente despejado",
        2: "Parcialmente nublado",
        3: "Cielo nublado",
        51: "Llovizna ligera",
        53: "Llovizna moderada",
        55: "Llovizna intensa",
        61: "Lluvia ligera",
        63: "Lluvia moderada",
        65: "Lluvia fuerte",
        71: "Nieve ligera",
        73: "Nieve moderada",
        75: "Nieve fuerte",
        80: "Chubascos ligeros",
        81: "Chubascos moderados",
        82: "Chubascos fuertes",
        95: "Tormenta"
    }
    
    if intent_tag == "clima_actual":
        current = data.get("current", {})
        temp = current.get("temperature_2m", "desconocida")
        precip = current.get("precipitation", 0)
        weather_code = current.get("weather_code", 0)
        weather_desc = weather_codes.get(weather_code, "Condición desconocida")
        return f"El clima actual en {location} es {weather_desc} con una temperatura de {temp}°C y precipitación de {precip} mm."
    
    elif intent_tag == "pronostico_corto":
        daily = data.get("daily", {})
        if not daily:
            return f"No hay datos de pronóstico disponibles para {location}."
        if period == "mañana":
            temp_max = daily.get("temperature_2m_max", [])[1]
            temp_min = daily.get("temperature_2m_min", [])[1]
            precip = daily.get("precipitation_sum", [])[1]
            weather_code = daily.get("weather_code", [])[1]
            weather_desc = weather_codes.get(weather_code, "Condición desconocida")
            return f"Mañana en {location} se espera {weather_desc} con temperaturas entre {temp_min}°C y {temp_max}°C, y precipitación de {precip} mm."
        else:
            days = min(days or 7, 16)  # Default to 7 for "esta semana", max 16
            response = f"Pronóstico para los próximos {days} días en {location}:\n"
            for i in range(days):
                date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                temp_max = daily.get("temperature_2m_max", [])[i]
                temp_min = daily.get("temperature_2m_min", [])[i]
                precip = daily.get("precipitation_sum", [])[i]
                weather_code = daily.get("weather_code", [])[i]
                weather_desc = weather_codes.get(weather_code, "Condición desconocida")
                response += f"- {date}: {weather_desc}, {temp_min}°C - {temp_max}°C, precipitación {precip} mm\n"
            return response
    
    elif intent_tag == "pronostico_largo":
        daily = data.get("daily", {})
        if not daily:
            return f"No hay datos de pronóstico a largo plazo disponibles para {location}."
        days = min(days or 16, 16)  # Open-Meteo supports up to 16 days
        response = f"Pronóstico a largo plazo para {location} (hasta {days} días):\n"
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            temp_max = daily.get("temperature_2m_max", [])[i]
            temp_min = daily.get("temperature_2m_min", [])[i]
            precip = daily.get("precipitation_sum", [])[i]
            weather_code = daily.get("weather_code", [])[i]
            weather_desc = weather_codes.get(weather_code, "Condición desconocida")
            response += f"- {date}: {weather_desc}, {temp_min}°C - {temp_max}°C, precipitación {precip} mm\n"
        return response
    
    elif intent_tag == "clima_general":
        current = data.get("current", {})
        temp = current.get("temperature_2m", "desconocida")
        precip = current.get("precipitation", 0)
        weather_code = current.get("weather_code", 0)
        weather_desc = weather_codes.get(weather_code, "Condición desconocida")
        return f"El clima general en {location} hoy es {weather_desc} con una temperatura promedio de {temp}°C y precipitación de {precip} mm."
    
    return "Lo siento, no pude procesar los datos del clima. Intenta de nuevo."

# Function to get response
def get_response(ints, intents_json, user_input):
    tag = ints[0]['intent'] if ints else 'sin_respuesta'
    if tag in ["clima_actual", "pronostico_corto", "pronostico_largo", "clima_general"]:
        location, period, days = extract_location_and_period(user_input)
        if not location:
            return "Por favor, especifica una ciudad o país para consultar el clima."
        latitude, longitude = get_coordinates(location)
        if latitude is None or longitude is None:
            return f"No pude encontrar la ubicación {location}. Intenta con otra ciudad o país."
        weather_data = get_weather_data(latitude, longitude, tag, days)
        return format_weather_response(weather_data, tag, location, period, days)
    
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Lo siento, no entendí eso."

# Streamlit app
st.title("Mi Chatbot del Clima")
st.write("Pregúntame sobre el clima actual o el pronóstico en cualquier ciudad o país. Los pronósticos están disponibles hasta 16 días con Open-Meteo.")

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
    intents_pred = predict_class(user_input, model)
    response = get_response(intents_pred, intents, user_input)
    
    st.session_state.chat_history.append(("Tú", user_input))
    st.session_state.chat_history.append(("Chatbot", response))
    
    st.session_state.input_text = ""

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "Tú":
        st.markdown(f"**{speaker}**: {message}")
    else:
        st.markdown(f"**{speaker}**: {message}")