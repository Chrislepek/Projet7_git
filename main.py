from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
import threading


# Chemin  vers les fichiers
CSV_PATH = './small_data.csv'
SCALER_PATH  = './scaler.joblib'
MODEL_PATH = './model.joblib'

# Charge les données
df = pd.read_csv(CSV_PATH)
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# FastAPI app
app = FastAPI()


#Fonction  pour récupérer les données d'un client
def get_client_data(client_id: int):
    client_row = df[df['SK_ID_CURR'] == client_id]

    if client_row.empty:
        return None  # Client ID non trouvé

    # Extraire les features
    features = client_row
    return features


@app.get("/predict/{client_id}")
def predict(client_id: int):
    # Vérifier si le client existe
    client_features = get_client_data(client_id)

    if client_features is None:
        raise HTTPException(status_code=404, detail="Client not found")

    # Préparer les données pour la prédiction
    # Convertir les données en tableau 2D avec numpy
    client_features_2d = np.array(client_features).reshape(1, -1)
    scaled_features = scaler.transform(client_features_2d)

    # Prédiction du modèle
    proba = model.predict_proba(scaled_features)
    print(proba)  # Probabilité de la classe "défaut"
    proba = proba[:, 1]

    # Appliquer le seuil optimisé
    seuil = 0.07
    classe = "Accepté" if proba <= seuil else "Refusé"

    return {"client_id": client_id, "probabilité": proba[0], "classe": classe}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)