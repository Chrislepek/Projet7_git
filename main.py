
import streamlit as st
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
import threading


# Chemin  vers les fichiers
CSV_PATH = './small_data.csv'
SCALER_PATH  = './scaler.joblib'
MODEL_PATH = './model.joblib'

# === Chargement des ressources ===
@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
scaler = load_scaler()
model = load_model()



#Fonction  pour récupérer les données d'un client

def get_client_data(client_id: int):
    client_row = df[df['SK_ID_CURR'] == client_id]

    if client_row.empty:
        return None  # Client ID non trouvé
   
    return client_row # Extraire les features


# ------------------------------Interface Streamlit-----------------------

st.title("Prédiction de Défaut de Client")

# === Entrée utilisateur ===
client_id = st.number_input("Entrez le Client ID", min_value=0, step=1)


# === Bouton : Prédiction ===

if st.button("Prédire"):
    client_features = get_client_data(client_id)

    if client_features is None:

        st.error(f"Client {client_id} non trouvé")
    else:
        # Préparer les données pour la prédiction
            # Convertir les données en tableau 2D avec numpy
        client_features_2d = np.array(client_features).reshape(1, -1)

        # Préparer les données pour la prédiction
        scaled_features = scaler.transform(client_features_2d)

        # Prédiction du modèle
        proba = model.predict_proba(scaled_features)
        print(proba)  # Probabilité de la classe "défaut"
        proba = proba[:, 1]

        # Appliquer le seuil optimisé (par exemple, 0.06)
        seuil = 0.07
        classe = "Accepté" if proba <= seuil else "Refusé"
        
        st.write(f"Client ID: {client_id}")
        st.write(f"Probabilité de défaut: {proba[0]:.4f}")
        st.write(f"Classe prédite: {classe}")
   