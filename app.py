import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Chargement du modèle
model = load_model(r'C:\Users\oumar\OneDrive\Bureau\fuit\Image_classify.keras')

# Liste des catégories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
            'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
            'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 
            'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Titre de l'application
st.header('Image Classification Model')

# Utilisation de Streamlit pour télécharger une image
uploaded_image = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Ouvrir l'image téléchargée avec PIL (Python Imaging Library)
    image = Image.open(uploaded_image)

    # Redimensionner l'image à la taille d'entrée du modèle (100x100)
    img_height = 100
    img_width = 100
    image = image.resize((img_width, img_height))

    # Afficher l'image téléchargée
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Convertir l'image en tableau numpy
    img_arr = np.array(image)

    # Ajouter une dimension pour simuler un batch (1 image)
    img_bat = np.expand_dims(img_arr, axis=0)

    # Normalisation des pixels de l'image (si nécessaire pour votre modèle)
    img_bat = img_bat / 255.0  # Si votre modèle a été formé avec des images normalisées

    # Prédiction
    predict = model.predict(img_bat)

    # Calculer la probabilité
    score = tf.nn.softmax(predict)

    # Affichage du résultat de la prédiction
    st.write(f"Fruit/Végétal détecté : {data_cat[np.argmax(score)]}")
    st.write(f"Précision : {np.max(score)*100:.2f}%")
