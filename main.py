from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict
import os
import logging
import math
import json
import io
import base64
from fastapi.responses import JSONResponse, FileResponse
import matplotlib.pyplot as plt
import shap
import requests
from PIL import Image

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Chemins vers les fichiers
DATA_PATH = './small_data.csv'
CLIENT_INFO_PATH = './appli_train_small.csv'  # Nouveau fichier pour les informations originelles des clients liés à small_data
SCALER_PATH = './scaler.joblib'
MODEL_PATH = './model.joblib'


colonnes_2keep = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY','OCCUPATION_TYPE']



# Fonctions pour charger les données, le scaler et le modèle
def load_data():
    return pd.read_csv(DATA_PATH)

def load_client_info():
    try:
        if not os.path.exists(CLIENT_INFO_PATH):
            raise FileNotFoundError(f"Fichier introuvable : {CLIENT_INFO_PATH}")
        return pd.read_csv(CLIENT_INFO_PATH, usecols=colonnes_2keep)
    except Exception as e:
        print(f"[ERREUR] Échec du chargement de client_info_df : {e}")
        return pd.DataFrame(columns=colonnes_2keep)

def load_scaler():
    return joblib.load(SCALER_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

#nettoyage des données 
def replace_nan_with_none(data):
    if isinstance(data, dict):
        return {k: replace_nan_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_none(v) for v in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data 

# Charge les données
df = load_data()
client_info_df = load_client_info()
scaler = load_scaler()
model = load_model()

#préparation explainer
df_shap_ready = df.astype(float)
explainer = shap.Explainer(model.predict_proba, df_shap_ready)

# FastAPI app
app = FastAPI()


# Fonction pour récupérer les données d'un client
def get_client_data(client_id: int):
    client_row = df[df['SK_ID_CURR'] == client_id]
    if client_row.empty:
        return None  # Client ID non trouvé
    return client_row

# Fonction pour récupérer les informations descriptives d'un client
def get_client_info(client_id: int):
    if client_info_df.empty:
        return None
    logger.debug(f"Recherche des informations pour le client {client_id}")
    client_info = client_info_df[client_info_df['SK_ID_CURR'] == client_id]
    if client_info.empty:
        return None  # Client ID non trouvé
    print(f"[DEBUG] Client info avant transformation : {client_info}")
    result = client_info.iloc[0].to_dict()
    logger.debug(f"Informations client sous forme de dictionnaire : {result}")
    print(f"[DEBUG] Client info après transformation en dictionnaire : {result}")
    return result


# Fonction pour récupérer l'importance des features locales


@app.get("/predict/{client_id}")
def predict(client_id: int):
    client_features = get_client_data(client_id)
    if client_features is None:
        raise HTTPException(status_code=404, detail="Client not found du predict")
    client_features_2d = np.array(client_features).reshape(1, -1)
    scaled_features = scaler.transform(client_features_2d)
    proba = model.predict_proba(scaled_features)
    proba = proba[:, 1]
    seuil = 0.07
    classe = "Accepté" if proba <= seuil else "Refusé"
    return {"client_id": client_id, "probabilité": proba[0], "classe": classe}


@app.get("/client_info/{client_id}")
def client_info(client_id: int):
    try:
        client_info = get_client_info(client_id)
        if client_info is None:
            raise HTTPException(status_code=404, detail="Client not found mais ça passe")
        clean_client_info = replace_nan_with_none(client_info)
        return clean_client_info
    except Exception as e:
        return {"error": str(e)}

@app.get("/feature_importance/global")

async def get_image():
    pix_path = os.path.abspath('./Global_Feature_importance.jpg')
    if not os.path.exists(pix_path):
        return {"error": "Image not found"}
    return FileResponse(pix_path, media_type='image/jpeg')

'''def get_global_shap_values():
    # preparation des feature global importance
    background = shap.sample(df, 20, random_state=42)
    explainer = shap.Explainer(model.predict_proba, background)
    shap_values = explainer(df)
    # Conversion des valeurs SHAP en liste pour la réponse JSON
    shap_values_list = shap_values.values[..., 1].tolist()
    feature_names = df.columns.tolist()
    #return {"shap_values": shap_values_list, "feature_names": feature_names}
    return {background}'''



@app.get("/feature_importance/local/{client_id}")
async def get_local_feature_importance(client_id: int):
    try:
        # 1. transformer les données client
        client_data = get_client_data(client_id)
        X_client_scaled = scaler.transform(client_data)

        # 2. Calculer les valeurs SHAP
        shap_values = explainer(X_client_scaled)

        # 3. Créer l'explication SHAP pour le client
        explanation = shap.Explanation(
            values=shap_values.values[0][:, 1],
            base_values=shap_values.base_values[0][1],
            data=X_client_scaled[0],
            feature_names=client_data.columns
        )

        # 4. Générer le graphique waterfall
        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        img_path = f"shap_local_{client_id}.png"
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()

        # 5. Retourner l'image en réponse
        return FileResponse(img_path, media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

##MAJ des données clients

class ClientUpdate(BaseModel):
    features: Dict[str, float]

@app.put("/update_client/{client_id}")
def update_client(client_id: int, update: ClientUpdate):
    global df, client_info_df
    client_row = df[df['SK_ID_CURR'] == client_id]
    if client_row.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    for feature, value in update.features.items():
        df.loc[df['SK_ID_CURR'] == client_id, feature] = value
        if feature in client_info_df.columns:
            client_info_df.loc[client_info_df['SK_ID_CURR'] == client_id, feature] = value
    df.to_csv(DATA_PATH, index=False)
    client_info_df.to_csv(CLIENT_INFO_PATH, index=False)
    return {"message": "Client updated successfully"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)