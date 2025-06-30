import os
import mlflow
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# =================================================================
# 1. CONFIGURATION POUR DAGSHUB
# =================================================================
# C'est la même configuration que dans train.py, mais avec des tokens
# pour que ça marche dans le pipeline GitHub Actions.
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/paulker194/mlops.mlflow'
# Dans le pipeline, ces valeurs viendront des "GitHub Secrets".
# Pour tester en local, vous pouvez les mettre ici temporairement.
os.environ['MLFLOW_TRACKING_USERNAME'] = 'paulker194'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'VOTRE_TOKEN_DAGSHUB_ICI' # IMPORTANT

# =================================================================
# 2. CHARGEMENT DU MEILLEUR MODÈLE DEPUIS DAGSHUB
# =================================================================
print("Connecting to DagsHub MLflow server...")
# On se connecte à l'expérience que vous avez utilisée
mlflow.set_experiment("Digits-Classification-Experiment")

# On récupère toutes les exécutions (runs) de cette expérience,
# triées par 'accuracy' en ordre décroissant.
runs = mlflow.search_runs(order_by=["metrics.accuracy DESC"])

# Le premier run de la liste est donc le meilleur.
best_run = runs.iloc[0]
best_run_id = best_run.run_id
print(f"Found best model in run: {best_run_id} with accuracy: {best_run['metrics.accuracy']:.4f}")

# On télécharge l'artefact (notre modèle) de ce run
local_model_path = mlflow.artifacts.download_artifacts(
    run_id=best_run_id,
    artifact_path="model" # C'est le dossier qu'on a spécifié dans log_artifact
)

# On charge le modèle en mémoire
model = joblib.load(os.path.join(local_model_path, "model.joblib"))
print("Model loaded successfully!")

# =================================================================
# 3. DÉFINITION DE L'API AVEC FASTAPI
# =================================================================
app = FastAPI(title="ML Model API for Digit Classification")

# On définit la structure des données d'entrée pour la prédiction
class PredictionInput(BaseModel):
    # Le jeu de données 'digits' a 64 features
    features: list[float]

@app.get("/")
def read_root():
    return {"message": f"Welcome! API is running. Best model from run_id '{best_run_id}' is loaded."}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # On convertit la liste en format numpy que le modèle comprend
        features_np = np.array(input_data.features).reshape(1, -1)
        
        # On fait la prédiction
        prediction = model.predict(features_np)
        
        # On retourne le résultat
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}