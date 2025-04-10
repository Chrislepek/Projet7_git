import sys
import os
import pytest
import pandas as pd
import numpy as np
import joblib

#répertoire parent au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import get_client_data, load_data, load_scaler, load_model  # Remplacez 'votre_module' par le nom de votre fichier Python

# Charger les données et les modèles une seule fois pour tous les tests
@pytest.fixture(scope="module")
def setup():
    df = load_data()
    scaler = load_scaler()
    model = load_model()
    return df, scaler, model

def test_get_client_data(setup):
    df, _, _ = setup
    # Test avec un client_id existant
    client_id = 100286
    client_data = get_client_data(client_id)
    assert client_data is not None, f"Client {client_id} non trouvé"

    # Test avec un client_id non existant
    client_id = 9999
    client_data = get_client_data(client_id)
    assert client_data is None, f"Client {client_id} trouvé alors qu'il ne devrait pas l'être"

def test_prediction_logic(setup):
    df, scaler, model = setup
    # Test avec un client_id existant
    client_id = 1
    client_data = get_client_data(client_id)

    if client_data is not None:
        client_features_2d = np.array(client_data).reshape(1, -1)
        scaled_features = scaler.transform(client_features_2d)
        proba = model.predict_proba(scaled_features)[:, 1]
        seuil = 0.07
        classe = "Accepté" if proba <= seuil else "Refusé"

        assert classe in ["Accepté", "Refusé"], f"Classe prédite inattendue: {classe}"